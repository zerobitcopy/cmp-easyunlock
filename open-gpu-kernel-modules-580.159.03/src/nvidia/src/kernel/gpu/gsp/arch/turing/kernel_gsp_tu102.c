/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*!
 * Provides TU102+ specific KernelGsp HAL implementations.
 */

#include "gpu/gsp/kernel_gsp.h"

#include "gpu/disp/kern_disp.h"
#include "gpu/fifo/kernel_fifo.h"
#include "gpu/mem_mgr/mem_mgr.h"
#include "gpu/mem_sys/kern_mem_sys.h"
#include "kernel/gpu/mc/kernel_mc.h"
#include "vgpu/rpc.h"
#include "rmgspseq.h"
#include "core/thread_state.h"
#include "os/os.h"
#include "nverror.h"
#include "nvrm_registry.h"
#include "crashcat/crashcat_report.h"

#include "published/turing/tu102/dev_gsp.h"
#include "published/turing/tu102/dev_gsp_addendum.h"
#include "published/turing/tu102/dev_riscv_pri.h"
#include "published/turing/tu102/dev_fbif_v4.h"
#include "published/turing/tu102/dev_falcon_v4.h"
#include "published/turing/tu102/dev_fb.h"  // for NV_PFB_PRI_MMU_WPR2_ADDR_HI
#include "published/turing/tu102/dev_fuse.h"
#include "published/turing/tu102/dev_ram.h"
#include "published/turing/tu102/dev_bus.h"
#include "published/ampere/ga100/dev_boot.h"
#include "published/ampere/ga100/dev_top.h"
#include "published/turing/tu102/dev_gc6_island.h"
#include "published/turing/tu102/dev_gc6_island_addendum.h"

#include "gpu/sec2/kernel_sec2.h"

#include "g_all_dcl_pb.h"
#include "lib/protobuf/prb.h"

static NvBool _kgspIsProcessorSuspended(OBJGPU *pGpu, void *pVoid);

#define GA100_STAGE1_REJOIN_SIGNATURE_SIZE  0x0000f800U
#define GA100_STAGE1_REJOIN_DMA_TARGET       0x00000800U
#define GA100_STAGE1_REJOIN_CANARY           0xfaceb13dU
#define GA100_STAGE1_REJOIN_ENTRY            0x000037b7U
#define GA100_STAGE1_FIVE_WRITE_ENTRY         0x000010b9U
#define GA100_STAGE1_FIVE_WRITE_BRIDGE        0x0000202dU
#define GA100_STAGE1_NATIVE_MAIN_RELEASE      0x000080d7U
#define GA100_STAGE1_NATIVE_RELEASE_DMA_SIZE  0x0000f7f0U
#define GA100_STAGE1_REJOIN_OFFSET(_dmem)    ((_dmem) - GA100_STAGE1_REJOIN_DMA_TARGET)

static NV_STATUS
_kgspPatchGa100Stage1WholeStackRejoin
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    GspFwWprMeta *pWprMeta,
    GSP_FIRMWARE *pGspFw
)
{
    NvU8 *pSignature;
    NvU64 signatureMemdescSize;
    NvBool bWholeStackRejoin;
    NvBool bFiveWriteNativeRelease;

    if (!gpuIsImplementation(pGpu, HAL_IMPL_GA100) ||
        (pKernelGsp->pSignatureMemdesc == NULL) ||
        (pGspFw->signatureSize != GA100_STAGE1_REJOIN_SIGNATURE_SIZE))
    {
        return NV_OK;
    }

    signatureMemdescSize = memdescGetSize(pKernelGsp->pSignatureMemdesc);
    NV_ASSERT_OR_RETURN(
        signatureMemdescSize >= pGspFw->signatureSize,
        NV_ERR_INVALID_STATE);

    pSignature = memdescMapInternal(
        pGpu, pKernelGsp->pSignatureMemdesc, TRANSFER_FLAGS_NONE);
    NV_ASSERT_OR_RETURN(pSignature != NULL, NV_ERR_INSUFFICIENT_RESOURCES);

    bWholeStackRejoin =
        MEM_RD32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xff5c)) ==
            GA100_STAGE1_REJOIN_ENTRY;
    bFiveWriteNativeRelease =
        (MEM_RD32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xff5c)) ==
            GA100_STAGE1_FIVE_WRITE_ENTRY) &&
        (MEM_RD32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xffd4)) ==
            GA100_STAGE1_FIVE_WRITE_BRIDGE) &&
        (MEM_RD32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xffe0)) ==
            GA100_STAGE1_NATIVE_MAIN_RELEASE) &&
        (MEM_RD32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xffe8)) ==
            GA100_STAGE1_REJOIN_CANARY) &&
        (MEM_RD32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xffec)) == 0U) &&
        (MEM_RD32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xfff4)) ==
            GA100_STAGE1_REJOIN_CANARY);

    // Only accept one of the two exact stage1 templates.  Other oversized
    // signatures retain their original payload bytes and aligned DMA length.
    if (!bWholeStackRejoin && !bFiveWriteNativeRelease)
    {
        memdescUnmapInternal(pGpu, pKernelGsp->pSignatureMemdesc, 0);
        return NV_OK;
    }

    if (bWholeStackRejoin)
    {
        // 0x4d4 ends in mpopaddret $r6.  Patch the restored register image
        // from this boot attempt's live WprMeta before rejoining at 0x37b7.
        MEM_WR32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xff3c),
                 NvU64_LO32(pWprMeta->sizeOfRadix3Elf));
        MEM_WR32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xff40),
                 NvU64_HI32(pWprMeta->gspFwWprStart));
        MEM_WR32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xff44),
                 NvU64_LO32(pWprMeta->bootBinOffset));
        MEM_WR32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xff48),
                 NvU64_LO32(pWprMeta->sizeOfBootloader));
        MEM_WR32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xff4c),
                 NvU64_LO32(pWprMeta->gspFwWprStart));
        MEM_WR32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xff50),
                 0x00000600U);
        MEM_WR32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xff54),
                 0x00000001U);
        MEM_WR32(pSignature + GA100_STAGE1_REJOIN_OFFSET(0xff58),
                 GA100_STAGE1_REJOIN_CANARY);
    }

    // The native-release template is stored as a 0xf800 signature so KMD can
    // identify its complete tail, but only DMA through D[0xffef].  D[0xffec]
    // is still forced to zero for scratch14, while D[0xfff0..0xffff] remains
    // the live main frame.  In particular this preserves the mutex3 token at
    // D[0xfff3], which is allocated afresh after a full reboot and therefore
    // must not be replayed from a previously captured snapshot.
    pWprMeta->sizeOfSignature = bFiveWriteNativeRelease ?
        GA100_STAGE1_NATIVE_RELEASE_DMA_SIZE : pGspFw->signatureSize;
    portAtomicMemoryFenceStore();

    if (bWholeStackRejoin)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_STAGE1_REJOIN_PATCH signature_size=0x%llx "
                  "signature_memdesc_size=0x%llx "
                  "r6_radix3_size=0x%08x r5_wpr_hi=0x%08x "
                  "r4_boot_bin=0x%08x r3_boot_size=0x%08x "
                  "r2_wpr_lo=0x%08x r1_meta=0x00000600 "
                  "ra=0x000037b7 dma_end=0x0000ffe4 "
                  "acr_mutex3_token=preserved\n",
                  pGspFw->signatureSize, signatureMemdescSize,
                  NvU64_LO32(pWprMeta->sizeOfRadix3Elf),
                  NvU64_HI32(pWprMeta->gspFwWprStart),
                  NvU64_LO32(pWprMeta->bootBinOffset),
                  NvU64_LO32(pWprMeta->sizeOfBootloader),
                  NvU64_LO32(pWprMeta->gspFwWprStart));
    }
    else
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_STAGE1_FIVE_WRITE_NATIVE_RELEASE "
                  "signature_size=0x%llx signature_memdesc_size=0x%llx "
                  "entry=0x000010b9 bridge=0x0000202d "
                  "native_release=0x000080d7 native_sp=0x0000ffe4 "
                  "dma_end=0x0000fff0 main_version=0x00000000 "
                  "acr_mutex3_token=preserved_live\n",
                  pGspFw->signatureSize, signatureMemdescSize);
    }

    memdescUnmapInternal(pGpu, pKernelGsp->pSignatureMemdesc, 0);
    return NV_OK;
}

#define GA100_FALCON_IDLESTATE                    0x0000004c
#define GA100_FALCON_IDLESTATE_BUSY_MASK          0x0000ffff
#define GA100_STAGE2_DMA_CONTEXT                  6
#define GA100_STAGE1_CLEAR_BLOCK_SIZE             (16 * 1024 * 1024)
#define GA100_VBIOS_FWSEC_SB_SCRATCH_INDEX         0x15
#define GA100_BOOT_SEQUENCE_STATE_MASK             0xfff00000U

#define GA100_SEC2_ENGCTL_PRIV_LEVEL_MASK          0x0084027c
#define GA100_SEC2_IMEM_PRIV_LEVEL_MASK            0x00840280
#define GA100_SEC2_DMEM_PRIV_LEVEL_MASK            0x00840284
#define GA100_SEC2_EXE_PRIV_LEVEL_MASK             0x0084028c
#define GA100_SEC2_IRQTMR_PRIV_LEVEL_MASK          0x00840290
#define GA100_SEC2_FALCON_SCTL                     0x00840240
#define GA100_SEC2_FALCON_SSTAT                    0x00840244
#define GA100_SEC2_FALCON_SCTL1                    0x00840250
#define GA100_SEC2_ENGINE                          0x008403c0
#define GA100_SEC2_RESET_PRIV_LEVEL_MASK           0x008403c4

#define GA100_SEC2_INTERNAL_I_1C000                0x00840700
#define GA100_SEC2_INTERNAL_I_1C300                0x0084070c
#define GA100_SEC2_INTERNAL_I_12000                0x00840480
#define GA100_SEC2_INTERNAL_I_12100                0x00840484
#define GA100_SEC2_INTERNAL_I_12400                0x00840490
#define GA100_SEC2_INTERNAL_I_12600                0x00840498

#define GA100_PTOP_DEVICE_INFO2_CONTINUE            NVBIT(31)
#define GA100_PTOP_DEVICE_INFO2_TYPE_SHIFT           24
#define GA100_PTOP_DEVICE_INFO2_TYPE_MASK            0x3f
#define GA100_PTOP_DEVICE_INFO2_INST_SHIFT           16
#define GA100_PTOP_DEVICE_INFO2_INST_MASK            0x0f
#define GA100_PTOP_DEVICE_INFO2_ADDR_MASK            0x00fff000
#define GA100_PTOP_DEVICE_INFO2_RESET_MASK           0x0000001f
#define GA100_PTOP_DEVICE_TYPE_SEC2                   0x0d

typedef struct
{
    KernelFalcon *pKernelFalcon;
} GA100_FALCON_IDLE_WAIT_DATA;

static NvBool
_kgspGa100FalconDmaAndXdIdle
(
    OBJGPU *pGpu,
    void *pVoid
)
{
    GA100_FALCON_IDLE_WAIT_DATA *pData =
        (GA100_FALCON_IDLE_WAIT_DATA *)pVoid;
    NvU32 dmaCmd = kflcnRegRead_HAL(
        pGpu, pData->pKernelFalcon, NV_PFALCON_FALCON_DMATRFCMD);
    NvU32 idleState = kflcnRegRead_HAL(
        pGpu, pData->pKernelFalcon, GA100_FALCON_IDLESTATE);

    return ((dmaCmd & NVBIT(1)) != 0) &&
           ((idleState & GA100_FALCON_IDLESTATE_BUSY_MASK) == 0);
}

static NV_STATUS
_kgspGa100WaitForFalconDmaAndXdIdle
(
    OBJGPU *pGpu,
    KernelFalcon *pKernelFalcon,
    const char *pName
)
{
    GA100_FALCON_IDLE_WAIT_DATA waitData = { pKernelFalcon };
    NV_STATUS status = gpuTimeoutCondWait(
        pGpu, _kgspGa100FalconDmaAndXdIdle, &waitData, NULL);

    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE: timeout waiting for %s DMA/XD idle: dma_cmd=0x%08x idle_state=0x%08x status=0x%x\n",
                  pName,
                  kflcnRegRead_HAL(pGpu, pKernelFalcon,
                                   NV_PFALCON_FALCON_DMATRFCMD),
                  kflcnRegRead_HAL(pGpu, pKernelFalcon,
                                   GA100_FALCON_IDLESTATE),
                  status);
    }

    return status;
}

static void
_kgspDumpGa100SecurityDomainState
(
    OBJGPU *pGpu,
    KernelFalcon *pGspFalcon,
    KernelFalcon *pSec2Falcon,
    const char *point
)
{
    NvU32 region6Addr = pSec2Falcon->fbifBase +
                        NV_PFALCON_FBIF_TRANSCFG(
                            GA100_STAGE2_DMA_CONTEXT);

    NV_PRINTF(LEVEL_ERROR,
              "GA100_SECURITY_DOMAIN_STATE point=%s\n",
              point);
    NV_PRINTF(LEVEL_ERROR,
              "GA100_SECURITY_DOMAIN_GFW point=%s plm05=0x%08x boot05=0x%08x sb_scratch15=0x%08x bsi_scratch14=0x%08x\n",
              point,
              GPU_REG_RD32(
                  pGpu,
                  NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK),
              GPU_REG_RD32(
                  pGpu,
                  NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0_GFW_BOOT),
              GPU_REG_RD32(
                  pGpu,
                  NV_PBUS_VBIOS_SCRATCH(
                      GA100_VBIOS_FWSEC_SB_SCRATCH_INDEX)),
              GPU_REG_RD32(pGpu, NV_PGC6_BSI_SECURE_SCRATCH_14));
    NV_PRINTF(LEVEL_ERROR,
              "GA100_SECURITY_DOMAIN_WPR point=%s lo=0x%08x hi=0x%08x region6@0x%08x=0x%08x\n",
              point,
              GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO),
              GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI),
              region6Addr, GPU_REG_RD32(pGpu, region6Addr));
    NV_PRINTF(LEVEL_ERROR,
              "GA100_SECURITY_DOMAIN_GSP point=%s cpuctl=0x%08x mailbox0=0x%08x mailbox1=0x%08x dmactl=0x%08x dmacmd=0x%08x idle=0x%08x\n",
              point,
              kflcnRegRead_HAL(
                  pGpu, pGspFalcon, NV_PFALCON_FALCON_CPUCTL),
              kflcnRegRead_HAL(
                  pGpu, pGspFalcon, NV_PFALCON_FALCON_MAILBOX0),
              kflcnRegRead_HAL(
                  pGpu, pGspFalcon, NV_PFALCON_FALCON_MAILBOX1),
              kflcnRegRead_HAL(
                  pGpu, pGspFalcon, NV_PFALCON_FALCON_DMACTL),
              kflcnRegRead_HAL(
                  pGpu, pGspFalcon, NV_PFALCON_FALCON_DMATRFCMD),
              kflcnRegRead_HAL(
                  pGpu, pGspFalcon, GA100_FALCON_IDLESTATE));
    NV_PRINTF(LEVEL_ERROR,
              "GA100_SECURITY_DOMAIN_SEC2 point=%s cpuctl=0x%08x mailbox0=0x%08x mailbox1=0x%08x sctl=0x%08x sstat=0x%08x sctl1=0x%08x engine=0x%08x dmactl=0x%08x dmacmd=0x%08x idle=0x%08x\n",
              point,
              kflcnRegRead_HAL(
                  pGpu, pSec2Falcon, NV_PFALCON_FALCON_CPUCTL),
              kflcnRegRead_HAL(
                  pGpu, pSec2Falcon, NV_PFALCON_FALCON_MAILBOX0),
              kflcnRegRead_HAL(
                  pGpu, pSec2Falcon, NV_PFALCON_FALCON_MAILBOX1),
              GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_SCTL),
              GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_SSTAT),
              GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_SCTL1),
              GPU_REG_RD32(pGpu, GA100_SEC2_ENGINE),
              kflcnRegRead_HAL(
                  pGpu, pSec2Falcon, NV_PFALCON_FALCON_DMACTL),
              kflcnRegRead_HAL(
                  pGpu, pSec2Falcon, NV_PFALCON_FALCON_DMATRFCMD),
              kflcnRegRead_HAL(
                  pGpu, pSec2Falcon, GA100_FALCON_IDLESTATE));
    NV_PRINTF(LEVEL_ERROR,
              "GA100_SECURITY_DOMAIN_SEC2_PLM point=%s engctl=0x%08x imem=0x%08x dmem=0x%08x exe=0x%08x irqtimer=0x%08x reset=0x%08x\n",
              point,
              GPU_REG_RD32(pGpu, GA100_SEC2_ENGCTL_PRIV_LEVEL_MASK),
              GPU_REG_RD32(pGpu, GA100_SEC2_IMEM_PRIV_LEVEL_MASK),
              GPU_REG_RD32(pGpu, GA100_SEC2_DMEM_PRIV_LEVEL_MASK),
              GPU_REG_RD32(pGpu, GA100_SEC2_EXE_PRIV_LEVEL_MASK),
              GPU_REG_RD32(pGpu, GA100_SEC2_IRQTMR_PRIV_LEVEL_MASK),
              GPU_REG_RD32(pGpu, GA100_SEC2_RESET_PRIV_LEVEL_MASK));
    NV_PRINTF(LEVEL_ERROR,
              "GA100_SECURITY_DOMAIN_INTERNAL point=%s I[0x1c000]=0x%08x I[0x1c300]=0x%08x I[0x12000]=0x%08x I[0x12100]=0x%08x I[0x12400]=0x%08x I[0x12600]=0x%08x\n",
              point,
              GPU_REG_RD32(pGpu, GA100_SEC2_INTERNAL_I_1C000),
              GPU_REG_RD32(pGpu, GA100_SEC2_INTERNAL_I_1C300),
              GPU_REG_RD32(pGpu, GA100_SEC2_INTERNAL_I_12000),
              GPU_REG_RD32(pGpu, GA100_SEC2_INTERNAL_I_12100),
              GPU_REG_RD32(pGpu, GA100_SEC2_INTERNAL_I_12400),
              GPU_REG_RD32(pGpu, GA100_SEC2_INTERNAL_I_12600));
}

static NV_STATUS
_kgspGetGa100Sec2ResetIndex
(
    OBJGPU *pGpu,
    KernelFalcon *pSec2Falcon,
    NvU32 *pResetIdx
)
{
    NvU32 cfg;
    NvU32 numRows;
    NvU32 row;
    NvU32 rowInDevice = 0;
    NvU32 type = 0;
    NvU32 inst = 0;
    NvU32 addr = 0;
    NvU32 resetIdx = 0;
    NvBool bHaveResetRow = NV_FALSE;

    NV_ASSERT_OR_RETURN(pSec2Falcon != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pResetIdx != NULL, NV_ERR_INVALID_ARGUMENT);

    cfg = GPU_REG_RD32(pGpu, NV_PTOP_DEVICE_INFO_CFG);
    numRows = DRF_VAL(_PTOP, _DEVICE_INFO_CFG, _NUM_ROWS, cfg);

    NV_PRINTF(LEVEL_ERROR,
              "GA100_PTOP: cfg=0x%08x num_rows=%u looking_for_sec2_base=0x%08x\n",
              cfg, numRows, pSec2Falcon->registerBase);

    for (row = 0; row < numRows; row++)
    {
        NvU32 data = GPU_REG_RD32(pGpu, NV_PTOP_DEVICE_INFO2(row));

        // Empty entries between devices do not start a new record.
        if ((data == 0) && (rowInDevice == 0))
            continue;

        switch (rowInDevice++)
        {
            case 0:
                type = (data >> GA100_PTOP_DEVICE_INFO2_TYPE_SHIFT) &
                       GA100_PTOP_DEVICE_INFO2_TYPE_MASK;
                inst = (data >> GA100_PTOP_DEVICE_INFO2_INST_SHIFT) &
                       GA100_PTOP_DEVICE_INFO2_INST_MASK;
                addr = 0;
                resetIdx = 0;
                bHaveResetRow = NV_FALSE;
                break;

            case 1:
                addr = data & GA100_PTOP_DEVICE_INFO2_ADDR_MASK;
                resetIdx = data & GA100_PTOP_DEVICE_INFO2_RESET_MASK;
                bHaveResetRow = NV_TRUE;
                break;

            default:
                break;
        }

        if ((data & GA100_PTOP_DEVICE_INFO2_CONTINUE) != 0)
            continue;

        if ((type == GA100_PTOP_DEVICE_TYPE_SEC2) && (inst == 0))
        {
            NV_PRINTF(LEVEL_ERROR,
                      "GA100_PTOP_SEC2: final_row=%u rows=%u addr=0x%08x reset_idx=%u have_reset_row=%u\n",
                      row, rowInDevice, addr, resetIdx, bHaveResetRow);

            if (!bHaveResetRow)
                return NV_ERR_INVALID_STATE;

            // PTOP reports the SEC engine's topology PRI address (0x87000 on
            // this GA100), not NV_PSEC's Falcon aperture (0x840000).  The
            // type/instance tuple is therefore authoritative for reset-index
            // discovery; keep both addresses in the log instead of requiring
            // them to match.
            if (addr != pSec2Falcon->registerBase)
            {
                NV_PRINTF(LEVEL_ERROR,
                          "GA100_PTOP_SEC2: topology PRI base 0x%08x differs from Falcon aperture 0x%08x (expected)\n",
                          addr, pSec2Falcon->registerBase);
            }

            *pResetIdx = resetIdx;
            return NV_OK;
        }

        rowInDevice = 0;
    }

    NV_PRINTF(LEVEL_ERROR,
              "GA100_PTOP: SEC2 type 0x%x instance 0 was not found in %u rows\n",
              GA100_PTOP_DEVICE_TYPE_SEC2, numRows);
    return NV_ERR_OBJECT_NOT_FOUND;
}

static NV_STATUS
_kgspResetGa100SecEngine
(
    OBJGPU *pGpu,
    KernelFalcon *pSec2Falcon
)
{
    KernelMc *pKernelMc = GPU_GET_KERNEL_MC(pGpu);
    NvU32 resetIdx;
    NvU32 secMask;
    NvU32 deviceBefore;
    NvU32 deviceDisabled;
    NvU32 deviceEnabled;
    NV_STATUS status;

    NV_ASSERT_OR_RETURN(pKernelMc != NULL, NV_ERR_INVALID_STATE);
    NV_ASSERT_OR_RETURN(pSec2Falcon != NULL, NV_ERR_INVALID_STATE);

    // GA100 moved per-engine reset controls from NV_PMC_ENABLE to the
    // reset-indexed NV_PMC_DEVICE_ENABLE register.  Resolve SEC2's reset
    // index from the device-info table instead of using the legacy SEC bit.
    status = _kgspGetGa100Sec2ResetIndex(
        pGpu, pSec2Falcon, &resetIdx);
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE: failed to resolve SEC2 reset index: 0x%x\n",
                  status);
        return status;
    }

    NV_ASSERT_OR_RETURN(resetIdx < 32, NV_ERR_OUT_OF_RANGE);
    secMask = DRF_IDX_DEF(
        _PMC, _DEVICE_ENABLE, _STATUS_BIT, resetIdx, _ENABLE);

    deviceBefore = GPU_REG_RD32(pGpu, NV_PMC_DEVICE_ENABLE(0));
    status = kmcWritePmcEnableReg_HAL(
        pGpu, pKernelMc, secMask, NV_FALSE, NV_TRUE);
    if (status != NV_OK)
        return status;
    deviceDisabled = GPU_REG_RD32(pGpu, NV_PMC_DEVICE_ENABLE(0));

    status = kmcWritePmcEnableReg_HAL(
        pGpu, pKernelMc, secMask, NV_TRUE, NV_TRUE);
    if (status != NV_OK)
        return status;
    deviceEnabled = GPU_REG_RD32(pGpu, NV_PMC_DEVICE_ENABLE(0));

    NV_PRINTF(LEVEL_ERROR,
              "GA100_SEC_PMC_RESET: reset_idx=%u device_enable=0x%08x->0x%08x->0x%08x sec_mask=0x%08x\n",
              resetIdx, deviceBefore, deviceDisabled, deviceEnabled,
              secMask);

    if ((deviceEnabled & secMask) == 0)
        return NV_ERR_INVALID_STATE;

    return NV_OK;
}

void
kgspConfigureFalcon_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    KernelFalconEngineConfig falconConfig;

    portMemSet(&falconConfig, 0, sizeof(falconConfig));

    falconConfig.registerBase       = DRF_BASE(NV_PGSP);
    falconConfig.riscvRegisterBase  = NV_FALCON2_GSP_BASE;
    falconConfig.fbifBase           = NV_PGSP_FBIF_BASE;
    falconConfig.bBootFromHs        = NV_FALSE;
    falconConfig.pmcEnableMask      = 0;
    falconConfig.bIsPmcDeviceEngine = NV_FALSE;
    falconConfig.physEngDesc        = ENG_GSP;

    // Enable CrashCat monitoring
    falconConfig.crashcatEngConfig.bEnable = NV_TRUE;
    falconConfig.crashcatEngConfig.pName = MAKE_NV_PRINTF_STR("GSP");
    falconConfig.crashcatEngConfig.errorId = GSP_ERROR;

    kflcnConfigureEngine(pGpu, staticCast(pKernelGsp, KernelFalcon), &falconConfig);
}

/*!
 * Check if the GSP is in debug mode
 *
 * @return whether the GSP is in debug mode or not
 */
NvBool
kgspIsDebugModeEnabled_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    NvU32 data;

    data = GPU_REG_RD32(pGpu, NV_FUSE_OPT_SECURE_GSP_DEBUG_DIS);

    return FLD_TEST_DRF(_FUSE, _OPT_SECURE_GSP_DEBUG_DIS, _DATA, _NO, data);
}

NV_STATUS
kgspAllocBootArgs_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    NvP64 pVa = NvP64_NULL;
    NvP64 pPriv = NvP64_NULL;
    NV_STATUS nvStatus = NV_OK;
    NvU64 flags = MEMDESC_FLAGS_NONE;

    flags |= MEMDESC_FLAGS_ALLOC_IN_UNPROTECTED_MEMORY;

    // GA100 Booter folds the low nibble of mailbox1 (the metadata address
    // high dword) into NV_PGC6_BSI_SECURE_SCRATCH_14[31:28] during its native
    // completion path.  Stage 1 of the two-stage flow must therefore use a
    // metadata DMA address below 4 GiB; otherwise it leaves binary-version 1
    // in protected AON/BSI scratch and the next Booter Load exits with 0x29.
    if (gpuIsImplementation(pGpu, HAL_IMPL_GA100))
        flags |= MEMDESC_FLAGS_ALLOC_32BIT_ADDRESSABLE;

    // Allocate WPR meta data
    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         memdescCreate(&pKernelGsp->pWprMetaDescriptor,
                                       pGpu, 0x1000, 0x1000,
                                       NV_TRUE, ADDR_SYSMEM, NV_MEMORY_CACHED,
                                       flags),
                        _kgspAllocBootArgs_exit_cleanup);

    memdescTagAlloc(nvStatus, NV_FB_ALLOC_RM_INTERNAL_OWNER_WPR_METADATA,
                    pKernelGsp->pWprMetaDescriptor);
    NV_ASSERT_OK_OR_GOTO(nvStatus, nvStatus,
                         _kgspAllocBootArgs_exit_cleanup);

    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         memdescMap(pKernelGsp->pWprMetaDescriptor, 0,
                                    memdescGetSize(pKernelGsp->pWprMetaDescriptor),
                                    NV_TRUE, NV_PROTECT_READ_WRITE,
                                    &pVa, &pPriv),
                         _kgspAllocBootArgs_exit_cleanup);

    pKernelGsp->pWprMeta = (GspFwWprMeta *)NvP64_VALUE(pVa);
    pKernelGsp->pWprMetaMappingPriv = pPriv;

    portMemSet(pKernelGsp->pWprMeta, 0, sizeof(*pKernelGsp->pWprMeta));

    //
    // Setup libos arguments memory
    //
    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         memdescCreate(&pKernelGsp->pLibosInitArgumentsDescriptor,
                                       pGpu,
                                       LIBOS_MEMORY_REGION_INIT_ARGUMENTS_MAX,
                                       LIBOS_MEMORY_REGION_INIT_ARGUMENTS_MAX,
                                       NV_TRUE, ADDR_SYSMEM, NV_MEMORY_UNCACHED,
                                       flags),
                         _kgspAllocBootArgs_exit_cleanup);

    memdescTagAlloc(nvStatus, NV_FB_ALLOC_RM_INTERNAL_OWNER_LIBOS_ARGS,
                    pKernelGsp->pLibosInitArgumentsDescriptor);
    NV_ASSERT_OK_OR_GOTO(nvStatus, nvStatus,
                         _kgspAllocBootArgs_exit_cleanup);

    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         memdescMap(pKernelGsp->pLibosInitArgumentsDescriptor, 0,
                                    memdescGetSize(pKernelGsp->pLibosInitArgumentsDescriptor),
                                    NV_TRUE, NV_PROTECT_READ_WRITE,
                                     &pVa, &pPriv),
                         _kgspAllocBootArgs_exit_cleanup);

    pKernelGsp->pLibosInitArgumentsCached = (LibosMemoryRegionInitArgument *)NvP64_VALUE(pVa);
    pKernelGsp->pLibosInitArgumentsMappingPriv = pPriv;

    portMemSet(pKernelGsp->pLibosInitArgumentsCached, 0, LIBOS_MEMORY_REGION_INIT_ARGUMENTS_MAX);

    // Setup bootloader arguments memory.
    NV_ASSERT(sizeof(GSP_ARGUMENTS_CACHED) <= 0x2000);

    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         memdescCreate(&pKernelGsp->pGspArgumentsDescriptor,
                                       pGpu, 0x1000, 0x1000,
                                       NV_TRUE, ADDR_SYSMEM, NV_MEMORY_CACHED,
                                       flags),
                         _kgspAllocBootArgs_exit_cleanup);

    memdescTagAlloc(nvStatus, NV_FB_ALLOC_RM_INTERNAL_OWNER_BOOTLOADER_ARGS,
                    pKernelGsp->pGspArgumentsDescriptor);
    NV_ASSERT_OK_OR_GOTO(nvStatus, nvStatus,
                         _kgspAllocBootArgs_exit_cleanup);

    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         memdescMap(pKernelGsp->pGspArgumentsDescriptor, 0,
                                    memdescGetSize(pKernelGsp->pGspArgumentsDescriptor),
                                    NV_TRUE, NV_PROTECT_READ_WRITE,
                                    &pVa, &pPriv),
                         _kgspAllocBootArgs_exit_cleanup);

    pKernelGsp->pGspArgumentsCached = (GSP_ARGUMENTS_CACHED *)NvP64_VALUE(pVa);
    pKernelGsp->pGspArgumentsMappingPriv = pPriv;

    portMemSet(pKernelGsp->pGspArgumentsCached, 0, sizeof(*pKernelGsp->pGspArgumentsCached));

    if (pGpu->getProperty(pGpu, PDB_PROP_GPU_ZERO_FB))
    {
        NvU32 heapSizeMB = 0;
        // Get the sysmem heap size override from the registry, or use default
        if (osReadRegistryDword(pGpu, NV_REG_STR_GSP_SYSMEM_HEAP_SIZE_MB, &heapSizeMB) != NV_OK)
        {
            heapSizeMB = NV_REG_STR_GSP_SYSMEM_HEAP_SIZE_MB_DEFAULT;
        }

        NV_ASSERT_OK_OR_GOTO(nvStatus,
                             memdescCreate(&pKernelGsp->pSysmemHeapDescriptor,
                                            pGpu, (NvU64)heapSizeMB << 20, 0,
                                            NV_FALSE, ADDR_SYSMEM, NV_MEMORY_UNCACHED,
                                            flags),
                             _kgspAllocBootArgs_exit_cleanup);

        memdescTagAlloc(nvStatus, NV_FB_ALLOC_RM_INTERNAL_OWNER_BOOTLOADER_ARGS,
                        pKernelGsp->pSysmemHeapDescriptor);
        NV_ASSERT_OK_OR_GOTO(nvStatus, nvStatus,
                            _kgspAllocBootArgs_exit_cleanup);

        NV_ASSERT_OR_GOTO(memdescCheckContiguity(pKernelGsp->pSysmemHeapDescriptor, AT_GPU),
                          _kgspAllocBootArgs_exit_cleanup);
    }

    return nvStatus;

_kgspAllocBootArgs_exit_cleanup:
    kgspFreeBootArgs_HAL(pGpu, pKernelGsp);
    return nvStatus;
}

void
kgspFreeBootArgs_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    // release wpr meta data resources
    if (pKernelGsp->pWprMeta != NULL)
    {
        memdescUnmap(pKernelGsp->pWprMetaDescriptor,
                     NV_TRUE,
                     (void *)pKernelGsp->pWprMeta,
                     pKernelGsp->pWprMetaMappingPriv);
        pKernelGsp->pWprMeta = NULL;
        pKernelGsp->pWprMetaMappingPriv = NULL;
    }
    if (pKernelGsp->pWprMetaDescriptor != NULL)
    {
        memdescFree(pKernelGsp->pWprMetaDescriptor);
        memdescDestroy(pKernelGsp->pWprMetaDescriptor);
        pKernelGsp->pWprMetaDescriptor = NULL;
    }

    // release libos init argument resources
    if (pKernelGsp->pLibosInitArgumentsCached != NULL)
    {
        memdescUnmap(pKernelGsp->pLibosInitArgumentsDescriptor,
                     NV_TRUE,
                     (void *)pKernelGsp->pLibosInitArgumentsCached,
                     pKernelGsp->pLibosInitArgumentsMappingPriv);
        pKernelGsp->pLibosInitArgumentsCached = NULL;
        pKernelGsp->pLibosInitArgumentsMappingPriv = NULL;
    }
    if (pKernelGsp->pLibosInitArgumentsDescriptor != NULL)
    {
        memdescFree(pKernelGsp->pLibosInitArgumentsDescriptor);
        memdescDestroy(pKernelGsp->pLibosInitArgumentsDescriptor);
        pKernelGsp->pLibosInitArgumentsDescriptor = NULL;
    }

    // release init argument page resources
    if (pKernelGsp->pGspArgumentsCached != NULL)
    {
        memdescUnmap(pKernelGsp->pGspArgumentsDescriptor,
                     NV_TRUE,
                     (void *)pKernelGsp->pGspArgumentsCached,
                     pKernelGsp->pGspArgumentsMappingPriv);
        pKernelGsp->pGspArgumentsCached = NULL;
        pKernelGsp->pGspArgumentsMappingPriv = NULL;
    }
    if (pKernelGsp->pGspArgumentsDescriptor != NULL)
    {
        memdescFree(pKernelGsp->pGspArgumentsDescriptor);
        memdescDestroy(pKernelGsp->pGspArgumentsDescriptor);
        pKernelGsp->pGspArgumentsDescriptor = NULL;
    }

    // Release radix3 version of GSP-RM ucode
    if (pKernelGsp->pGspUCodeRadix3Descriptor != NULL)
    {
        memdescFree(pKernelGsp->pGspUCodeRadix3Descriptor);
        memdescDestroy(pKernelGsp->pGspUCodeRadix3Descriptor);
        pKernelGsp->pGspUCodeRadix3Descriptor = NULL;
    }

    // Release signature memory
    if (pKernelGsp->pSignatureMemdesc != NULL)
    {
        memdescFree(pKernelGsp->pSignatureMemdesc);
        memdescDestroy(pKernelGsp->pSignatureMemdesc);
        pKernelGsp->pSignatureMemdesc = NULL;
    }

    // Release sysmem heap memory
    if (pKernelGsp->pSysmemHeapDescriptor != NULL)
    {
        memdescFree(pKernelGsp->pSysmemHeapDescriptor);
        memdescDestroy(pKernelGsp->pSysmemHeapDescriptor);
        pKernelGsp->pSysmemHeapDescriptor = NULL;
    }
}

/*!
 * Determine if GSP reload via SEC2 is completed.
 */
static NvBool
_kgspIsReloadCompleted
(
    OBJGPU  *pGpu,
    void    *pVoid
)
{
    NvU32 reg;

    reg = GPU_REG_RD32(pGpu, NV_PGC6_BSI_SECURE_SCRATCH_14);

    return FLD_TEST_DRF(_PGC6, _BSI_SECURE_SCRATCH_14, _BOOT_STAGE_3_HANDOFF, _VALUE_DONE, reg);
}

/*!
 * Set command queue head for CPU to GSP message queue
 *
 * @param[in]   pGpu            GPU object pointer
 * @param[in]   pKernelGsp      KernelGsp object pointer (not used)
 * @param[in]   queueIdx        index
 * @param[in]   value           value to set command queue head to.
 *
 * @return NV_OK if the operation was successful.
 */
NV_STATUS
kgspSetCmdQueueHead_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    NvU32   queueIdx,
    NvU32   value
)
{
    NV_ASSERT_OR_RETURN(queueIdx < NV_PGSP_QUEUE_HEAD__SIZE_1, NV_ERR_INVALID_ARGUMENT);

    // Write the value to the correct queue head.
    GPU_REG_WR32(pGpu, NV_PGSP_QUEUE_HEAD(queueIdx), value);

    return NV_OK;
}

/*!
 * Load entrypoint address of boot binary into mailbox regs.
 */
void
kgspProgramLibosBootArgsAddr_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    NvU64 addr =
        memdescGetPhysAddr(pKernelGsp->pLibosInitArgumentsDescriptor, AT_GPU, 0);

    GPU_REG_WR32(pGpu, NV_PGSP_FALCON_MAILBOX0, NvU64_LO32(addr));
    GPU_REG_WR32(pGpu, NV_PGSP_FALCON_MAILBOX1, NvU64_HI32(addr));
}

/*!
 * Prepare to boot GSP-RM
 *
 * This routine handles the prerequisites to booting GSP-RM that requires the API LOCK:
 *   - prepares boot binary image
 *   - prepares RISCV core to run GSP-RM
 *
 * Note that boot binary and GSP-RM images have already been placed
 * in the appropriate places by kgspPopulateWprMeta_HAL().
 *
 * Note that this routine is based on flcnBootstrapRiscvOS_GA102().
 *
 * @param[in]   pGpu            GPU object pointer
 * @param[in]   pKernelGsp      GSP object pointer
 * @param[in]   bootMode        GSP boot mode
 *
 * @return NV_OK if GSP-RM RISCV boot was successful.
 *         Appropriate NV_ERR_xxx value otherwise.
 */
NV_STATUS
kgspPrepareForBootstrap_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    KernelGspBootMode bootMode
)
{
    NV_STATUS     status;
    KernelFalcon *pKernelFalcon = staticCast(pKernelGsp, KernelFalcon);

    // Only for GSP client builds
    if (!IS_GSP_CLIENT(pGpu))
    {
        NV_PRINTF(LEVEL_ERROR, "IS_GSP_CLIENT is not set.\n");
        return NV_ERR_NOT_SUPPORTED;
    }

    if (!kflcnIsRiscvCpuEnabled_HAL(pGpu, pKernelFalcon))
    {
        NV_PRINTF(LEVEL_ERROR, "RISC-V core is not enabled.\n");
        return NV_ERR_NOT_SUPPORTED;
    }

    //
    // Prepare to execute FWSEC to setup FRTS if we have a FRTS region
    // Note: for resume and GC6 exit, FRTS is restored by Booter not FWSEC
    //
    if ((bootMode == KGSP_BOOT_MODE_NORMAL) &&
        (kgspGetFrtsSize_HAL(pGpu, pKernelGsp) > 0))
    {
        pKernelGsp->pPreparedFwsecCmd = portMemAllocNonPaged(sizeof(KernelGspPreparedFwsecCmd));
        status = kgspPrepareForFwsecFrts_HAL(pGpu, pKernelGsp,
                                             pKernelGsp->pFwsecUcode,
                                             pKernelGsp->pWprMeta->frtsOffset,
                                             pKernelGsp->pPreparedFwsecCmd);
        if (status != NV_OK)
        {
            portMemFree(pKernelGsp->pPreparedFwsecCmd);
            pKernelGsp->pPreparedFwsecCmd = NULL;
            return status;
        }
    }

    return NV_OK;
}

/*!
 * Obtain sysmem addr or arguments to be consumed by Booter Load.
 * Booter expects different arguments for normal boot, resume, and GC6 exit.
 *
 * @param[in]  bootMode  GSP boot mode
 */
static inline NvU64
_kgspGetBooterLoadArgs
(
    KernelGsp *pKernelGsp,
    KernelGspBootMode bootMode
)
{
    switch (bootMode)
    {
        case KGSP_BOOT_MODE_NORMAL:
            return memdescGetPhysAddr(pKernelGsp->pWprMetaDescriptor, AT_GPU, 0);
        case KGSP_BOOT_MODE_SR_RESUME:
            return memdescGetPhysAddr(pKernelGsp->pSRMetaDescriptor, AT_GPU, 0);
        case KGSP_BOOT_MODE_GC6_EXIT:
            return 0;
    }

    // unreachable
    NV_ASSERT_FAILED("unexpected GSP boot mode");
    return 0;
}

/*!
 * Boot GSP-RM.
 *
 * This routine handles the following:
 *   - starts the RISCV core and passes control to boot binary image
 *   - waits for GSP-RM to complete initialization
 *
 * Note that boot binary and GSP-RM images have already been placed
 * in the appropriate places by kgspPopulateWprMeta_HAL().
 *
 * Note that this routine is based on flcnBootstrapRiscvOS_GA102().
 *
 * Note that this routine can be called without the API lock for
 * parllel initialization.
 *
 * @param[in]   pGpu            GPU object pointer
 * @param[in]   pKernelGsp      GSP object pointer
 * @param[in]   bootMode        GSP boot mode
 *
 * @return NV_OK if GSP-RM RISCV boot was successful.
 *         Appropriate NV_ERR_xxx value otherwise.
 */
static NV_STATUS
_s_kgspBootstrapBeforeBooter_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    KernelGspBootMode bootMode
)
{
    NV_STATUS status;
    KernelFalcon *pKernelFalcon = staticCast(pKernelGsp, KernelFalcon);

    // Execute Scrubber if needed.
    if (((bootMode == KGSP_BOOT_MODE_SR_RESUME) ||
         (bootMode == KGSP_BOOT_MODE_NORMAL)) &&
        (pKernelGsp->pScrubberUcode != NULL))
    {
        NV_ASSERT_OK_OR_RETURN(
            kgspExecuteScrubberIfNeeded_HAL(pGpu, pKernelGsp));
    }

    // GA100 invokes this setup before each of its two Booter Load stages.
    if (bootMode == KGSP_BOOT_MODE_NORMAL)
    {
        if (kgspGetFrtsSize_HAL(pGpu, pKernelGsp) > 0)
        {
            NV_ASSERT_OR_RETURN(pKernelGsp->pPreparedFwsecCmd != NULL,
                                NV_ERR_INVALID_STATE);

            NV_ASSERT_OK_OR_RETURN(kflcnReset_HAL(pGpu, pKernelFalcon));

            status = kgspExecuteFwsec_HAL(pGpu, pKernelGsp,
                                          pKernelGsp->pPreparedFwsecCmd);
            portMemFree(pKernelGsp->pPreparedFwsecCmd);
            pKernelGsp->pPreparedFwsecCmd = NULL;

            NV_ASSERT_OK_OR_RETURN(status);
        }

        NV_ASSERT_OK_OR_RETURN(
            kflcnResetIntoRiscv_HAL(pGpu, pKernelFalcon));

        kgspProgramLibosBootArgsAddr_HAL(pGpu, pKernelGsp);
    }

    return NV_OK;
}

static NV_STATUS
_s_kgspBootstrapAfterBooter_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    KernelGspBootMode bootMode
)
{
    KernelFalcon *pKernelFalcon = staticCast(pKernelGsp, KernelFalcon);
    RM_RISCV_UCODE_DESC *pRiscvDesc = pKernelGsp->pGspRmBootUcodeDesc;

    kflcnRegWrite_HAL(pGpu, pKernelFalcon, NV_PFALCON_FALCON_OS,
                      pRiscvDesc->appVersion);

    if (kflcnIsRiscvActive_HAL(pGpu, pKernelFalcon) ||
        _kgspIsProcessorSuspended(pGpu, pKernelGsp))
    {
        NV_PRINTF(LEVEL_INFO, "GSP ucode loaded and RISCV started.\n");
    }
    else
    {
        NV_PRINTF(LEVEL_ERROR,
                  "Failed to boot GSP. cpu=0x%08x irq=0x%08x mbox0=0x%08x mbox1=0x%08x os=0x%08x rm=0x%08x bootvec=0x%08x dmactl=0x%08x hwcfg2=0x%08x dbg=0x%08x riscvActive=%d\n",
                  kflcnRegRead_HAL(pGpu, pKernelFalcon,
                                   NV_PFALCON_FALCON_CPUCTL),
                  kflcnRegRead_HAL(pGpu, pKernelFalcon,
                                   NV_PFALCON_FALCON_IRQSTAT),
                  kflcnRegRead_HAL(pGpu, pKernelFalcon,
                                   NV_PFALCON_FALCON_MAILBOX0),
                  kflcnRegRead_HAL(pGpu, pKernelFalcon,
                                   NV_PFALCON_FALCON_MAILBOX1),
                  kflcnRegRead_HAL(pGpu, pKernelFalcon,
                                   NV_PFALCON_FALCON_OS),
                  kflcnRegRead_HAL(pGpu, pKernelFalcon,
                                   NV_PFALCON_FALCON_RM),
                  kflcnRegRead_HAL(pGpu, pKernelFalcon,
                                   NV_PFALCON_FALCON_BOOTVEC),
                  kflcnRegRead_HAL(pGpu, pKernelFalcon,
                                   NV_PFALCON_FALCON_DMACTL),
                  kflcnRegRead_HAL(pGpu, pKernelFalcon,
                                   NV_PFALCON_FALCON_HWCFG2),
                  kflcnRegRead_HAL(pGpu, pKernelFalcon,
                                   NV_PFALCON_FALCON_DEBUGINFO),
                  kflcnIsRiscvActive_HAL(pGpu, pKernelFalcon));

        return NV_ERR_NOT_READY;
    }

    NV_PRINTF(LEVEL_INFO, "Waiting for GSP fw RM to be ready...\n");

    if (bootMode == KGSP_BOOT_MODE_NORMAL)
    {
        NV_ASSERT_OK_OR_RETURN(
            GspStatusQueueInit(pGpu, &pKernelGsp->pRpc->pMessageQueueInfo));
    }

    NV_ASSERT_OK_OR_RETURN(kgspWaitForRmInitDone(pGpu, pKernelGsp));

    NV_PRINTF(LEVEL_INFO, "GSP FW RM ready.\n");

    return NV_OK;
}

NV_STATUS
kgspBootstrap_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    KernelGspBootMode bootMode
)
{
    NV_STATUS status;

    NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
        _s_kgspBootstrapBeforeBooter_TU102(pGpu, pKernelGsp, bootMode));

    status = kgspExecuteBooterLoad_HAL(
        pGpu, pKernelGsp, _kgspGetBooterLoadArgs(pKernelGsp, bootMode));
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "failed to execute Booter Load (ucode for initial boot): 0x%x\n",
                  status);
        return status;
    }

    return _s_kgspBootstrapAfterBooter_TU102(pGpu, pKernelGsp, bootMode);
}

NV_STATUS
kgspTeardownGa100Stage1_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    NV_STATUS status;
    KernelGspPreparedFwsecCmd preparedFwsecSbCmd;
    KernelSec2 *pKernelSec2 = GPU_GET_KERNEL_SEC2(pGpu);
    KernelFalcon *pGspFalcon = staticCast(pKernelGsp, KernelFalcon);
    KernelFalcon *pSec2Falcon = staticCast(pKernelSec2, KernelFalcon);
    MemoryManager *pMemoryManager = GPU_GET_MEMORY_MANAGER(pGpu);
    GspFwWprMeta *pWprMeta = pKernelGsp->pWprMeta;
    MEMORY_DESCRIPTOR *pClearMemdesc = NULL;
    NvU32 region6Addr = pSec2Falcon->fbifBase +
                        NV_PFALCON_FBIF_TRANSCFG(GA100_STAGE2_DMA_CONTEXT);
    NvU32 region6Before;
    NvU32 region6AfterReset;
    NvU32 region6After;
    NvU32 secureScratch14Before;
    NvU32 secureScratch14Expected;
    NvU32 secureScratch14After;
    NvU32 wpr2LoBefore;
    NvU32 wpr2HiBefore;
    NvU32 wpr2LoAfter;
    NvU32 wpr2HiAfter;
    NvU64 clearBase;
    NvU64 clearLimit;
    NvU64 clearSize;
    NvBool bWpr2Cleared;

    NV_ASSERT_OR_RETURN(pWprMeta != NULL, NV_ERR_INVALID_STATE);

    _kgspDumpGa100SecurityDomainState(
        pGpu, pGspFalcon, pSec2Falcon,
        "STAGE1_POST_BEFORE_GSP_RESET");

    // 1. Stop GSP before changing any of the regions it may still access.
    status = kflcnReset_HAL(pGpu, pGspFalcon);
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE: failed to stop GSP after stage 1: 0x%x\n",
                  status);
        return status;
    }

    _kgspDumpGa100SecurityDomainState(
        pGpu, pGspFalcon, pSec2Falcon,
        "AFTER_GSP_RESET");

    // 2. Wait for both engines' Falcon DMA queues and external/XD paths.
    NV_CHECK_OK_OR_RETURN(
        LEVEL_ERROR,
        _kgspGa100WaitForFalconDmaAndXdIdle(pGpu, pGspFalcon, "GSP"));
    NV_CHECK_OK_OR_RETURN(
        LEVEL_ERROR,
        _kgspGa100WaitForFalconDmaAndXdIdle(pGpu, pSec2Falcon, "SEC2"));

    // 3. Restore the PreOsApps/security-domain state before attempting to
    // enter another SEC2 HS Booter.  A SEC2 Falcon engine reset alone does
    // not re-arm the secure I/O write at I[0x12000].
    NV_ASSERT_OR_RETURN(pKernelGsp->pFwsecUcode != NULL,
                        NV_ERR_INVALID_STATE);

    status = kgspPrepareForFwsecSb_HAL(
        pGpu, pKernelGsp, pKernelGsp->pFwsecUcode,
        &preparedFwsecSbCmd);
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE: failed to prepare FWSEC-SB after stage 1: 0x%x\n",
                  status);
        return status;
    }

    status = kgspExecuteFwsec_HAL(
        pGpu, pKernelGsp, &preparedFwsecSbCmd);
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE: failed to execute FWSEC-SB after stage 1: 0x%x\n",
                  status);
        return status;
    }

    _kgspDumpGa100SecurityDomainState(
        pGpu, pGspFalcon, pSec2Falcon,
        "AFTER_FWSEC_SB");

    NV_PRINTF(LEVEL_ERROR,
              "GA100_TWO_STAGE: FWSEC-SB completed after stage 1; resetting SEC engine before stage 2\n");

    status = _kgspResetGa100SecEngine(pGpu, pSec2Falcon);
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE: failed to reset complete SEC engine after stage 1: 0x%x\n",
                  status);
        return status;
    }

    _kgspDumpGa100SecurityDomainState(
        pGpu, pGspFalcon, pSec2Falcon,
        "AFTER_SEC_PMC_RESET");

    // 4. Follow the complete SEC engine reset with the normal SEC2 Falcon
    // reset, then clear FBIF DMA context/region entry 6.
    region6Before = GPU_REG_RD32(pGpu, region6Addr);
    status = kflcnReset_HAL(pGpu, pSec2Falcon);
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE: failed to reset SEC2 before clearing FBIF region entry 6: 0x%x\n",
                  status);
        return status;
    }

    _kgspDumpGa100SecurityDomainState(
        pGpu, pGspFalcon, pSec2Falcon,
        "AFTER_SEC2_RESET");

    region6AfterReset = GPU_REG_RD32(pGpu, region6Addr);
    GPU_REG_WR32(pGpu, region6Addr, 0);
    region6After = GPU_REG_RD32(pGpu, region6Addr);
    if (region6After != 0)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE: failed to clear SEC2 FBIF region entry 6 addr=0x%08x before=0x%08x after_reset=0x%08x after_write=0x%08x\n",
                  region6Addr, region6Before, region6AfterReset,
                  region6After);
        return NV_ERR_INVALID_STATE;
    }

    // A Falcon/SEC engine reset does not reset the AON/BSI Booter sequence
    // state.  Stage 1 leaves BOOTER_BINARY_VERSION=1 and the Booter Load
    // handoff DONE bit in NV_PGC6_BSI_SECURE_SCRATCH_14.  A second Booter
    // Load therefore exits at 0x80a5/0x1c75 with mailbox 0x29 before it can
    // acquire mutex3 or enter the real Load flow.  Re-arm only the upper
    // sequence fields and preserve CURRENT_VPR_RANGE_START_ADDR[19:0].
    secureScratch14Before = GPU_REG_RD32(
        pGpu, NV_PGC6_BSI_SECURE_SCRATCH_14);
    secureScratch14Expected =
        secureScratch14Before & ~GA100_BOOT_SEQUENCE_STATE_MASK;
    GPU_REG_WR32(pGpu, NV_PGC6_BSI_SECURE_SCRATCH_14,
                 secureScratch14Expected);
    secureScratch14After = GPU_REG_RD32(
        pGpu, NV_PGC6_BSI_SECURE_SCRATCH_14);
    if (secureScratch14After != secureScratch14Expected)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE_SCRATCH14_REARM_FAILED "
                  "before=0x%08x expected=0x%08x after=0x%08x "
                  "clear_mask=0x%08x\n",
                  secureScratch14Before, secureScratch14Expected,
                  secureScratch14After, GA100_BOOT_SEQUENCE_STATE_MASK);
        return NV_ERR_INVALID_STATE;
    }

    NV_PRINTF(LEVEL_ERROR,
              "GA100_TWO_STAGE_SCRATCH14_REARM before=0x%08x "
              "after=0x%08x cleared_sequence_bits=0x%08x "
              "preserved_vpr_start=0x%05x\n",
              secureScratch14Before, secureScratch14After,
              secureScratch14Before & GA100_BOOT_SEQUENCE_STATE_MASK,
              secureScratch14After & ~GA100_BOOT_SEQUENCE_STATE_MASK);

    // 5. Attempt the requested direct WPR2 bounds/mode clear.  Do not invoke
    // Booter Unload here: this experiment intentionally performs two Booter
    // Load executions back-to-back.  GA100 may keep these registers hardware
    // locked after the first Load, so record that state but continue to the
    // second Load rather than converting it into an unload flow.
    wpr2LoBefore = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO);
    wpr2HiBefore = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
    GPU_REG_WR32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI, 0);
    GPU_REG_WR32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO, 0);
    wpr2LoAfter = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO);
    wpr2HiAfter = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
    bWpr2Cleared = (wpr2LoAfter == 0) && (wpr2HiAfter == 0);
    if (!bWpr2Cleared)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE_WPR2_CLEAR_BLOCKED_CONTINUING: lo=0x%08x->0x%08x hi=0x%08x->0x%08x\n",
                  wpr2LoBefore, wpr2LoAfter,
                  wpr2HiBefore, wpr2HiAfter);
    }

    // 6. Clear the WPR heap and runtime/boot image range, preserving metadata
    // and FRTS.  Stage 2 will repopulate this range from fresh DMA sources.
    clearBase = pWprMeta->gspFwHeapOffset;
    clearLimit = pWprMeta->frtsOffset;
    NV_ASSERT_OR_RETURN(clearLimit > clearBase, NV_ERR_INVALID_STATE);
    clearSize = clearLimit - clearBase;

    if (bWpr2Cleared)
    {
        NV_CHECK_OK_OR_RETURN(
            LEVEL_ERROR,
            memdescCreate(&pClearMemdesc, pGpu, clearSize, 0, NV_TRUE,
                          ADDR_FBMEM, NV_MEMORY_UNCACHED,
                          MEMDESC_FLAGS_NONE));
        memdescDescribe(pClearMemdesc, ADDR_FBMEM, clearBase, clearSize);
        status = memmgrMemsetInBlocks(
            pMemoryManager, pClearMemdesc, 0, 0, clearSize,
            TRANSFER_FLAGS_PREFER_CE, GA100_STAGE1_CLEAR_BLOCK_SIZE);
        memdescDestroy(pClearMemdesc);
        if (status != NV_OK)
        {
            NV_PRINTF(LEVEL_ERROR,
                      "GA100_TWO_STAGE: failed to clear WPR heap/runtime range 0x%016llx-0x%016llx: 0x%x\n",
                      clearBase, clearLimit - 1, status);
            return status;
        }
    }
    else
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE: skipping host WPR heap/runtime clear while WPR2 remains hardware-locked range=0x%016llx-0x%016llx\n",
                  clearBase, clearLimit - 1);
    }

    NV_PRINTF(LEVEL_ERROR,
              "GA100_TWO_STAGE_TEARDOWN: gsp_stopped=1 dma_xd_idle=1 fwsec_sb=1 booter_unload=0 sec2_reset=1 region6=0x%08x->0x%08x->0x%08x wpr2_lo=0x%08x->0x%08x wpr2_hi=0x%08x->0x%08x wpr2_cleared=%u clear_performed=%u clear=0x%016llx-0x%016llx\n",
              region6Before, region6AfterReset, region6After,
              wpr2LoBefore, wpr2LoAfter, wpr2HiBefore, wpr2HiAfter,
              bWpr2Cleared, bWpr2Cleared,
              clearBase, clearLimit - 1);

    return NV_OK;
}

NV_STATUS
kgspBootstrapGa100Stage1_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    NV_STATUS status;

    NV_ASSERT_OR_RETURN(gpuIsImplementation(pGpu, HAL_IMPL_GA100),
                        NV_ERR_NOT_SUPPORTED);

    NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
        _s_kgspBootstrapBeforeBooter_TU102(
            pGpu, pKernelGsp, KGSP_BOOT_MODE_NORMAL));

    status = kgspExecuteBooterLoadStage_TU102(
        pGpu, pKernelGsp,
        _kgspGetBooterLoadArgs(pKernelGsp, KGSP_BOOT_MODE_NORMAL),
        1, NV_TRUE);

    if (status != KGSP_BOOTER_STATUS_STAGE1_COMPLETE)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE: stage 1 did not reach a clean halted hand-off: 0x%x\n",
                  status);
    }

    return status;
}

NV_STATUS
kgspBootstrapGa100Stage2_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    NV_STATUS status;

    NV_ASSERT_OR_RETURN(gpuIsImplementation(pGpu, HAL_IMPL_GA100),
                        NV_ERR_NOT_SUPPORTED);

    // Stage 2 reuses the same GSP image and WPR metadata address.  The caller
    // changes only GspFwWprMeta::sizeOfSignature before rerunning Scrubber,
    // FWSEC, GSP reset-into-RISCV, and Booter Load.
    NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
        _s_kgspBootstrapBeforeBooter_TU102(
            pGpu, pKernelGsp, KGSP_BOOT_MODE_NORMAL));

    status = kgspExecuteBooterLoadStage_TU102(
        pGpu, pKernelGsp,
        _kgspGetBooterLoadArgs(pKernelGsp, KGSP_BOOT_MODE_NORMAL),
        2, NV_FALSE);
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE: stage 2 Booter Load failed: 0x%x\n",
                  status);
        return status;
    }

    return _s_kgspBootstrapAfterBooter_TU102(
        pGpu, pKernelGsp, KGSP_BOOT_MODE_NORMAL);
}

/*!
 * Obtain sysmem addr or arguments to be consumed by Booter Unload.
 * Booter expects different arguments for normal unload, suspend, and GC6 enter.
 *
 * @param[in]  unloadMode  GSP unload mode
 */
static inline NvU64
_kgspGetBooterUnloadArgs
(
    KernelGsp *pKernelGsp,
    KernelGspUnloadMode unloadMode
)
{
    switch (unloadMode)
    {
        case KGSP_UNLOAD_MODE_NORMAL:
            return 0;
        case KGSP_UNLOAD_MODE_SR_SUSPEND:
            return memdescGetPhysAddr(pKernelGsp->pSRMetaDescriptor, AT_GPU, 0);
        case KGSP_UNLOAD_MODE_GC6_ENTER:
            return 0;
    }

    // unreachable
    NV_ASSERT_FAILED("unexpected GSP unload mode");
    return 0;
}

/*!
 * Teardown remaining GSP state after GSP-RM unloads.
 *
 * For pre-Hopper, this involves running FWSEC-SB to put back pre-OS apps and
 * Booter Unload to teardown WPR2.
 *
 * @param[in]   pGpu            GPU object pointer
 * @param[in]   pKernelGsp      GSP object pointer
 * @param[in]   unloadMode      GSP unload mode
 */
NV_STATUS
kgspTeardown_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    KernelGspUnloadMode unloadMode
)
{
    NV_STATUS status;

    //
    // Avoid cascading timeouts when attempting to invoke the below ucodes if
    // we are unloading due to a GSP-RM timeout.
    //
    threadStateResetTimeout(pGpu);

    if (unloadMode != KGSP_UNLOAD_MODE_GC6_ENTER)
    {
        KernelGspPreparedFwsecCmd preparedCmd;

        // Reset GSP so we can load FWSEC-SB
        status = kflcnReset_HAL(pGpu, staticCast(pKernelGsp, KernelFalcon));
        NV_ASSERT((status == NV_OK) || (status == NV_ERR_GPU_IN_FULLCHIP_RESET));

        // Invoke FWSEC-SB to put back PreOsApps during driver unload
        status = kgspPrepareForFwsecSb_HAL(pGpu, pKernelGsp, pKernelGsp->pFwsecUcode, &preparedCmd);
        if (status != NV_OK)
        {
            NV_PRINTF(LEVEL_ERROR, "failed to prepare for FWSEC-SB for PreOsApps during driver unload: 0x%x\n", status);
            NV_ASSERT_FAILED("FWSEC-SB prep failed");
        }
        else
        {
            status = kgspExecuteFwsec_HAL(pGpu, pKernelGsp, &preparedCmd);
            if ((status != NV_OK) && (status != NV_ERR_GPU_IN_FULLCHIP_RESET))
            {
                NV_PRINTF(LEVEL_ERROR, "failed to execute FWSEC-SB for PreOsApps during driver unload: 0x%x\n", status);
                NV_ASSERT_FAILED("FWSEC-SB failed");
            }
        }
    }

    // Execute Booter Unload
    status = kgspExecuteBooterUnloadIfNeeded_HAL(pGpu, pKernelGsp,
                                                 _kgspGetBooterUnloadArgs(pKernelGsp, unloadMode));
    return status;
}

void
kgspGetGspRmBootUcodeStorage_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    BINDATA_STORAGE **ppBinStorageImage,
    BINDATA_STORAGE **ppBinStorageDesc
)
{
    const BINDATA_ARCHIVE *pBinArchive = kgspGetBinArchiveGspRmBoot_HAL(pKernelGsp);

    *ppBinStorageImage = (BINDATA_STORAGE *) bindataArchiveGetStorage(pBinArchive, BINDATA_LABEL_UCODE_IMAGE);
    *ppBinStorageDesc  = (BINDATA_STORAGE *) bindataArchiveGetStorage(pBinArchive, BINDATA_LABEL_UCODE_DESC);
}

/*!
 * Populate WPR meta structure.
 *
 * Firmware scrubs the last 256mb of FB, no memory outside of this region
 * may be used until the FW RM has scrubbed the remainder of memory.
 *
 *   ---------------------------- <- fbSize (end of FB, 1M aligned)
 *   | VGA WORKSPACE            |
 *   ---------------------------- <- vbiosReservedOffset  (64K? aligned)
 *   | (potential align. gap)   |
 *   ---------------------------- <- gspFwWprEnd (128K aligned)
 *   | FRTS data                |    (frtsSize is 0 on GA100)
 *   | ------------------------ | <- frtsOffset
 *   | BOOT BIN (e.g. SK + BL)  |
 *   ---------------------------- <- bootBinOffset
 *   | GSP FW ELF               |
 *   ---------------------------- <- gspFwOffset
 *   | GSP FW (WPR) HEAP        |
 *   ---------------------------- <- gspFwHeapOffset**
 *   | Booter-placed metadata   |
 *   | (struct GspFwWprMeta)    |
 *   ---------------------------- <- gspFwWprStart (128K aligned)
 *   | GSP FW (non-WPR) HEAP    |
 *   ---------------------------- <- nonWprHeapOffset, gspFwRsvdStart
 *
 *  gspFwHeapOffset** contains the entire WPR heap region, which can be subdivided
 *  for various GSP FW components.
 *
 * @param       pGpu          GPU object pointer
 * @param       pKernelGsp    KernelGsp object pointer
 * @param       pGspFw        Pointer to GSP-RM fw image.
 */
NV_STATUS
kgspPopulateWprMeta_TU102
(
    OBJGPU         *pGpu,
    KernelGsp      *pKernelGsp,
    GSP_FIRMWARE   *pGspFw
)
{
    KernelMemorySystem  *pKernelMemorySystem  = GPU_GET_KERNEL_MEMORY_SYSTEM(pGpu);
    KernelDisplay       *pKernelDisplay = GPU_GET_KERNEL_DISPLAY(pGpu);
    MemoryManager       *pMemoryManager = GPU_GET_MEMORY_MANAGER(pGpu);
    GspFwWprMeta        *pWprMeta = pKernelGsp->pWprMeta;
    RM_RISCV_UCODE_DESC *pRiscvDesc = pKernelGsp->pGspRmBootUcodeDesc;
    NvU64                vbiosReservedOffset;
    NvU64                mmuLockLo, mmuLockHi;
    NvBool               bIsMmuLockValid;
    NvU32                data;

    ct_assert(sizeof(*pWprMeta) == 256);

    NV_ASSERT_OR_RETURN(IS_GSP_CLIENT(pGpu), NV_ERR_NOT_SUPPORTED);

    NV_ASSERT_OR_RETURN(pKernelGsp->pGspRmBootUcodeImage != NULL, NV_ERR_INVALID_STATE);
    NV_ASSERT_OR_RETURN(pKernelGsp->gspRmBootUcodeSize != 0, NV_ERR_INVALID_STATE);
    NV_ASSERT_OR_RETURN(pRiscvDesc != NULL, NV_ERR_INVALID_STATE);

    NV_ASSERT_OK_OR_RETURN(kmemsysGetUsableFbSize_HAL(pGpu, pKernelMemorySystem, &pWprMeta->fbSize));

    //
    // Start layout calculations at the top and work down.
    // Figure out where VGA workspace is located.  We do not have to adjust
    // it ourselves (see vgaRelocateWorkspaceBase_HAL()).
    //
    if (gpuFuseSupportsDisplay_HAL(pGpu) &&
        kdispGetVgaWorkspaceBase(pGpu, pKernelDisplay, &pWprMeta->vgaWorkspaceOffset))
    {
        if (pWprMeta->vgaWorkspaceOffset < (pWprMeta->fbSize - DRF_SIZE(NV_PRAMIN)))
        {
            const NvU32 VBIOS_WORKSPACE_SIZE = 0x20000;

            // Point NV_PDISP_VGA_WORKSPACE_BASE to end-of-FB
            pWprMeta->vgaWorkspaceOffset = (pWprMeta->fbSize - VBIOS_WORKSPACE_SIZE);
        }
    }
    else
    {
        pWprMeta->vgaWorkspaceOffset = (pWprMeta->fbSize - DRF_SIZE(NV_PRAMIN));
    }
    pWprMeta->vgaWorkspaceSize = pWprMeta->fbSize - pWprMeta->vgaWorkspaceOffset;

    // Check for MMU locked region (locked by VBIOS)
    NV_ASSERT_OK_OR_RETURN(
        memmgrReadMmuLock_HAL(pGpu, pMemoryManager, &bIsMmuLockValid, &mmuLockLo, &mmuLockHi));

    if (bIsMmuLockValid)
        vbiosReservedOffset = NV_MIN(mmuLockLo, pWprMeta->vgaWorkspaceOffset);
    else
        vbiosReservedOffset = pWprMeta->vgaWorkspaceOffset;

    // Set the size of the GSP FW ahead of kgspGetWprEndMargin()
    pWprMeta->sizeOfRadix3Elf = pGspFw->imageSize;

    // End of WPR region (128KB aligned), shifted for any WPR end margin
    pWprMeta->gspFwWprEnd = NV_ALIGN_DOWN64(vbiosReservedOffset - kgspGetWprEndMargin(pGpu, pKernelGsp), 0x20000);

    pWprMeta->frtsSize = kgspGetFrtsSize(pGpu, pKernelGsp);
    pWprMeta->frtsOffset = pWprMeta->gspFwWprEnd - pWprMeta->frtsSize;

    // Offset of boot binary image (4K aligned)
    pWprMeta->sizeOfBootloader = pKernelGsp->gspRmBootUcodeSize;
    pWprMeta->bootBinOffset = NV_ALIGN_DOWN64(pWprMeta->frtsOffset - pWprMeta->sizeOfBootloader, 0x1000);

    //
    // Compute the start of the ELF.  Align to 64K to avoid issues with
    // inherent alignment constraints.
    //
    pWprMeta->gspFwOffset = NV_ALIGN_DOWN64(pWprMeta->bootBinOffset - pWprMeta->sizeOfRadix3Elf, 0x10000);

    //
    // The maximum size of the GSP-FW heap depends on the statically-sized regions before and after
    // it in the pre-scrubbed region of FB.
    //
    const NvU64 MB = (1ULL << 20);
    const NvU64 nonWprHeapSize = NV_ALIGN_UP64(kgspGetNonWprHeapSize(pGpu, pKernelGsp), MB);
    const NvU64 wprMetaSize = NV_ALIGN_UP64(sizeof(*pWprMeta), MB);
    const NvU64 preWprHeapSize = wprMetaSize + nonWprHeapSize;
    const NvU64 postWprHeapSize = NV_ALIGN_UP64(pWprMeta->fbSize - pWprMeta->gspFwOffset, MB);
    const NvU64 wprHeapSize = kgspGetFwHeapSize(pGpu, pKernelGsp, preWprHeapSize, postWprHeapSize);

    // GSP-RM heap in WPR, align to 1MB
    pWprMeta->gspFwHeapOffset = NV_ALIGN_DOWN64(pWprMeta->gspFwOffset - wprHeapSize, MB);
    pWprMeta->gspFwHeapSize = NV_ALIGN_DOWN64(pWprMeta->gspFwOffset - pWprMeta->gspFwHeapOffset, MB);

    // Number of VF partitions allocating sub-heaps from the WPR heap
    pWprMeta->gspFwHeapVfPartitionCount = pGpu->bVgpuGspPluginOffloadEnabled ? MAX_PARTITIONS_WITH_GFID_32VM : 0;

    //
    // Start of WPR region (128K alignment requirement, but 1MB aligned so that
    // the extra padding sits in WPR instead of in between the end of the
    // non-WPR heap and the start of WPR).
    //
    pWprMeta->gspFwWprStart = pWprMeta->gspFwHeapOffset - wprMetaSize;

    // Non WPR heap (1MB aligned)
    pWprMeta->nonWprHeapSize = nonWprHeapSize;
    pWprMeta->nonWprHeapOffset = pWprMeta->gspFwWprStart - pWprMeta->nonWprHeapSize;

    pWprMeta->gspFwRsvdStart = pWprMeta->nonWprHeapOffset;

    // Physical address of GSP-RM firmware in system memory.
    pWprMeta->sysmemAddrOfRadix3Elf =
        memdescGetPhysAddr(pKernelGsp->pGspUCodeRadix3Descriptor, AT_GPU, 0);

    // Physical address of boot loader firmware in system memory.
    pWprMeta->sysmemAddrOfBootloader =
        memdescGetPhysAddr(pKernelGsp->pGspRmBootUcodeMemdesc, AT_GPU, 0);

    // Set necessary info from bootloader desc
    pWprMeta->bootloaderCodeOffset = pRiscvDesc->monitorCodeOffset;
    pWprMeta->bootloaderDataOffset = pRiscvDesc->monitorDataOffset;
    pWprMeta->bootloaderManifestOffset = pRiscvDesc->manifestOffset;

    if (pKernelGsp->pSignatureMemdesc != NULL)
    {
        pWprMeta->sysmemAddrOfSignature = memdescGetPhysAddr(pKernelGsp->pSignatureMemdesc, AT_GPU, 0);
        pWprMeta->sizeOfSignature = memdescGetSize(pKernelGsp->pSignatureMemdesc);
    }

    // CrashCat queue (if allocated in sysmem)
    KernelCrashCatEngine *pKernelCrashCatEng = staticCast(pKernelGsp, KernelCrashCatEngine);
    MEMORY_DESCRIPTOR *pCrashCatQueueMemDesc = kcrashcatEngineGetQueueMemDesc(pKernelCrashCatEng);
    if (pCrashCatQueueMemDesc != NULL)
    {
        NV_ASSERT_CHECKED(memdescGetAddressSpace(pCrashCatQueueMemDesc) == ADDR_SYSMEM);
        pWprMeta->sysmemAddrOfCrashReportQueue = memdescGetPhysAddr(pCrashCatQueueMemDesc, AT_GPU, 0);
        pWprMeta->sizeOfCrashReportQueue = (NvU32)memdescGetSize(pCrashCatQueueMemDesc);
    }

    if ((osReadRegistryDword(pGpu, NV_REG_STR_RM_BOOT_GSPRM_WITH_BOOST_CLOCKS, &data) == NV_OK) &&
        (data == NV_REG_STR_RM_BOOT_GSPRM_WITH_BOOST_CLOCKS_DISABLED))
    {
        pKernelGsp->bBootGspRmWithBoostClocks = NV_FALSE;
    }

    if ((pGpu->idInfo.PCIDeviceID == 0x20BB10DE) &&
        (pGpu->idInfo.PCISubDeviceID == 0x14A110DE))
    {
        pKernelGsp->bBootGspRmWithBoostClocks = NV_FALSE;
    }

    pWprMeta->bootCount = 0;
    pWprMeta->verified = 0;
    pWprMeta->revision = GSP_FW_WPR_META_REVISION;
    pWprMeta->magic = GSP_FW_WPR_META_MAGIC;

    if (pKernelGsp->bBootGspRmWithBoostClocks)
    {
        pWprMeta->flags |= GSP_FW_FLAGS_CLOCK_BOOST;
    }

#if 0
    NV_PRINTF(LEVEL_ERROR, "WPR meta data offset:     0x%016llx\n", pWprMeta->gspFwWprStart);
    NV_PRINTF(LEVEL_ERROR, "  magic:                  0x%016llx\n", pWprMeta->magic);
    NV_PRINTF(LEVEL_ERROR, "  revision:               0x%016llx\n", pWprMeta->revision);
    NV_PRINTF(LEVEL_ERROR, "  sysmemAddrOfRadix3Elf:  0x%016llx\n", pWprMeta->sysmemAddrOfRadix3Elf);
    NV_PRINTF(LEVEL_ERROR, "  sizeOfRadix3Elf:        0x%016llx\n", pWprMeta->sizeOfRadix3Elf);
    NV_PRINTF(LEVEL_ERROR, "  sysmemAddrOfBootloader: 0x%016llx\n", pWprMeta->sysmemAddrOfBootloader);
    NV_PRINTF(LEVEL_ERROR, "  sizeOfBootloader:       0x%016llx\n", pWprMeta->sizeOfBootloader);
    NV_PRINTF(LEVEL_ERROR, "  sysmemAddrOfSignature:  0x%016llx\n", pWprMeta->sysmemAddrOfSignature);
    NV_PRINTF(LEVEL_ERROR, "  sizeOfSignature:        0x%016llx\n", pWprMeta->sizeOfSignature);
    NV_PRINTF(LEVEL_ERROR, "  gspFwRsvdStart:         0x%016llx\n", pWprMeta->gspFwRsvdStart);
    NV_PRINTF(LEVEL_ERROR, "  nonWprHeap:             0x%016llx - 0x%016llx (0x%016llx)\n", pWprMeta->nonWprHeapOffset, pWprMeta->nonWprHeapOffset + pWprMeta->nonWprHeapSize - 1, pWprMeta->nonWprHeapSize);
    NV_PRINTF(LEVEL_ERROR, "  gspFwWprStart:          0x%016llx\n", pWprMeta->gspFwWprStart);
    NV_PRINTF(LEVEL_ERROR, "  gspFwHeap:              0x%016llx - 0x%016llx (0x%016llx)\n", pWprMeta->gspFwHeapOffset, pWprMeta->gspFwHeapOffset + pWprMeta->gspFwHeapSize - 1, pWprMeta->gspFwHeapSize);
    NV_PRINTF(LEVEL_ERROR, "  gspFwOffset:            0x%016llx - 0x%016llx (0x%016llx)\n", pWprMeta->gspFwOffset, pWprMeta->gspFwOffset + pWprMeta->sizeOfRadix3Elf - 1, pWprMeta->sizeOfRadix3Elf);
    NV_PRINTF(LEVEL_ERROR, "  bootBinOffset:          0x%016llx - 0x%016llx (0x%016llx)\n", pWprMeta->bootBinOffset, pWprMeta->bootBinOffset + pWprMeta->sizeOfBootloader - 1, pWprMeta->sizeOfBootloader);
    NV_PRINTF(LEVEL_ERROR, "  frtsOffset:             0x%016llx - 0x%016llx (0x%016llx)\n", pWprMeta->frtsOffset, pWprMeta->frtsOffset + pWprMeta->frtsSize - 1, pWprMeta->frtsSize);
    NV_PRINTF(LEVEL_ERROR, "  gspFwWprEnd:            0x%016llx\n", pWprMeta->gspFwWprEnd);
    NV_PRINTF(LEVEL_ERROR, "  fbSize:                 0x%016llx\n", pWprMeta->fbSize);
    NV_PRINTF(LEVEL_ERROR, "  vgaWorkspaceOffset:     0x%016llx - 0x%016llx (0x%016llx)\n", pWprMeta->vgaWorkspaceOffset, pWprMeta->vgaWorkspaceOffset + pWprMeta->vgaWorkspaceSize - 1, pWprMeta->vgaWorkspaceSize);
    NV_PRINTF(LEVEL_ERROR, "  bootCount:              0x%016llx\n", pWprMeta->bootCount);
    NV_PRINTF(LEVEL_ERROR, "  verified:               0x%016llx\n", pWprMeta->verified);
#endif

    NV_CHECK_OK_OR_RETURN(
        LEVEL_ERROR,
        _kgspPatchGa100Stage1WholeStackRejoin(
            pGpu, pKernelGsp, pWprMeta, pGspFw));

    return NV_OK;
}

/*!
 * Execute GSP sequencer operation
 *
 * @param[in]   pGpu            GPU object pointer
 * @param[in]   pKernelGsp      KernelGsp object pointer
 * @param[in]   opCode          Sequencer opcode
 * @param[in]   pPayload        Pointer to payload
 * @param[in]   payloadSize     Size of payload in bytes
 *
 * @return NV_OK if the sequencer operation was successful.
 *         Appropriate NV_ERR_xxx value otherwise.
 */
NV_STATUS
kgspExecuteSequencerCommand_TU102
(
    OBJGPU         *pGpu,
    KernelGsp      *pKernelGsp,
    NvU32           opCode,
    NvU32          *pPayload,
    NvU32           payloadSize
)
{
    NV_STATUS       status        = NV_OK;
    KernelFalcon   *pKernelFalcon = staticCast(pKernelGsp, KernelFalcon);
    NvU32           secMailbox0   = 0;

    switch (opCode)
    {
        case GSP_SEQ_BUF_OPCODE_CORE_RESUME:
        {
            KernelFalcon *pKernelSec2Falcon = staticCast(GPU_GET_KERNEL_SEC2(pGpu), KernelFalcon);

            NV_ASSERT_OK_OR_RETURN(kflcnResetIntoRiscv_HAL(pGpu, pKernelFalcon));
            kgspProgramLibosBootArgsAddr_HAL(pGpu, pKernelGsp);

            NV_PRINTF(LEVEL_INFO, "---------------Starting SEC2 to resume GSP-RM------------\n");
            // Start SEC2 in order to resume GSP-RM
            kflcnStartCpu_HAL(pGpu, pKernelSec2Falcon);

            // Wait for reload to be completed.
            status = gpuTimeoutCondWait(pGpu, _kgspIsReloadCompleted, NULL, NULL);

            // Check SEC mailbox.
            secMailbox0 = kflcnRegRead_HAL(pGpu, pKernelSec2Falcon, NV_PFALCON_FALCON_MAILBOX0);

            if ((status != NV_OK) || (secMailbox0 != NV_OK))
            {
                NV_PRINTF(LEVEL_ERROR, "Timeout waiting for SEC2-RTOS to resume GSP-RM. SEC2 Mailbox0 is : 0x%x\n", secMailbox0);
                DBG_BREAKPOINT();
                return NV_ERR_TIMEOUT;
            }

            // Ensure the CPU is started
            if (kflcnIsRiscvActive_HAL(pGpu, pKernelFalcon))
            {
                NV_PRINTF(LEVEL_INFO, "GSP ucode loaded and RISCV started.\n");
            }
            else
            {
                NV_ASSERT_FAILED("Failed to boot GSP");
                status = NV_ERR_NOT_READY;
            }
            break;
        }

        default:
        {
            status = NV_ERR_INVALID_ARGUMENT;
            break;
        }
    }

    return status;
}

/*!
 * Reset the GSP HW
 *
 * @return NV_OK if the GSP HW was properly reset
 */
NV_STATUS
kgspResetHw_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    GPU_FLD_WR_DRF_DEF(pGpu, _PGSP, _FALCON_ENGINE, _RESET, _TRUE);

    // Reg read cycles needed for signal propagation.
    for (NvU32 i = 0; i < FLCN_RESET_PROPAGATION_DELAY_COUNT; i++)
    {
        GPU_REG_RD32(pGpu, NV_PGSP_FALCON_ENGINE);
    }

    GPU_FLD_WR_DRF_DEF(pGpu, _PGSP, _FALCON_ENGINE, _RESET, _FALSE);

    // Reg read cycles needed for signal propagation.
    for (NvU32 i = 0; i < FLCN_RESET_PROPAGATION_DELAY_COUNT; i++)
    {
        GPU_REG_RD32(pGpu, NV_PGSP_FALCON_ENGINE);
    }

    return NV_OK;
}

static NvBool kgspCrashCatReportImpactsGspRm(CrashCatReport *pReport)
{
    NV_CRASHCAT_CONTAINMENT containment;

    containment = crashcatReportSourceContainment_HAL(pReport);
    switch (containment)
    {
       case NV_CRASHCAT_CONTAINMENT_RISCV_MODE_M:
       case NV_CRASHCAT_CONTAINMENT_RISCV_HART:
       case NV_CRASHCAT_CONTAINMENT_UNCONTAINED:
           return NV_TRUE;
       default:
           return NV_FALSE;
    }
}

NvBool
kgspHealthCheck_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp
)
{
    NvBool bHealthy = NV_TRUE;
    char buildIdString[64];
    LibosElfNoteHeader *pBuildIdNoteHeader = pKernelGsp->pBuildIdSection;

    // CrashCat is the primary reporting interface for GSP issues
    KernelCrashCatEngine *pKernelCrashCatEng = staticCast(pKernelGsp, KernelCrashCatEngine);
    if (kcrashcatEngineConfigured(pKernelCrashCatEng))
    {
        CrashCatEngine *pCrashCatEng = staticCast(pKernelCrashCatEng, CrashCatEngine);
        CrashCatReport *pReport;

        while ((pReport = crashcatEngineGetNextCrashReport(pCrashCatEng)) != NULL)
        {
            if (crashcatReportIsWatchdog_HAL(pReport))
            {
                NV_PRINTF(LEVEL_INFO, "Assign a CrashcatReport to pWatchdogReport\n");
                //
                // Keep the first report until the corresponding RPC is done
                // Before that, subsequent reports are ignored
                //
                if (pKernelGsp->pWatchdogReport != NULL)
                    objDelete(pReport);
                else
                    pKernelGsp->pWatchdogReport = pReport;

                continue;
            }

            if (kgspCrashCatReportImpactsGspRm(pReport))
                bHealthy = NV_FALSE;

            NV_PRINTF(LEVEL_ERROR,
                "****************************** GSP-CrashCat Report *******************************\n");
            crashcatReportLog(pReport);

            kgspInitNocatData(pGpu, pKernelGsp, GSP_NOCAT_CRASHCAT_REPORT);

            // Build id string can be used by offline decoder to decode crashcat data/addresses to symbols
            if (pKernelGsp->pBuildIdSection != NULL)
            {
              portStringBufferToHex(buildIdString, 64, pBuildIdNoteHeader->data + pBuildIdNoteHeader->namesz, pBuildIdNoteHeader->descsz);

              prbEncAddString(&pKernelGsp->nocatData.nocatBuffer,
                              GSP_XIDREPORT_BUILDID,
                              &buildIdString[0]);
            }

            // ErrorCode of nocat event is used for categorizing GSP crash data collected from the field via nocat
            // Since lowest bit of ra is always empty, we use bit 0 to store the sign bit, for
            // differentiating task crash vs libos crash
            // signbit of ra - 1 bit, 0
            //	ra           - (28 - 1) bits, 27:1
            //	scause       - 4 bits, 31:28
            //	stval        - 32 bits, 63:32
            pKernelGsp->nocatData.errorCode |= (crashcatReportRa_HAL(pReport) >> 63) & 1;
            pKernelGsp->nocatData.errorCode |= crashcatReportRa_HAL(pReport) & 0xFFFFFFE;
            pKernelGsp->nocatData.errorCode |= (crashcatReportXcause_HAL(pReport) & 0xF) << 28;
            pKernelGsp->nocatData.errorCode |= (crashcatReportXtval_HAL(pReport) & 0xFFFFFFFF) << 32;

            prbEncAddUInt32(&pKernelGsp->nocatData.nocatBuffer, GSP_XIDREPORT_XID, 120);
            prbEncAddUInt32(&pKernelGsp->nocatData.nocatBuffer, GSP_XIDREPORT_GPUINSTANCE, gpuGetInstance(pGpu));
            crashcatReportLogToProtobuf_HAL(pReport, &pKernelGsp->nocatData.nocatBuffer);
            kgspPostNocatData(pGpu, pKernelGsp, osGetTimestamp());

            objDelete(pReport);
        }
    }

    if (!bHealthy)
    {
        NvBool bFirstFatal = !pKernelGsp->bFatalError;

        pKernelGsp->bFatalError = NV_TRUE;

        if (pKernelGsp->pRpc)
        {
            // Ideally we could have crashcat report and RPC history in the same NOCAT event. But for each NOCAT event
            // there is a size limit of 1k per event, and crashcat data/ rpc history each takes up like 700 bytes, so we have to create 2 events.
            // Technically both event are associated with the xid 120 report
            // Since any non-terminating NOCAT event after the first terminating event will be dropped,
            // we need to set a earlier time here than the Crashcat Nocat event for RPC history to be preserved in NOCAT
            kgspInitNocatData(pGpu, pKernelGsp, GSP_NOCAT_GSP_RPC_HISTORY);
            prbEncAddUInt32(&pKernelGsp->nocatData.nocatBuffer, GSP_XIDREPORT_XID, 120);
            kgspLogRpcDebugInfoToProtobuf(pGpu, pKernelGsp->pRpc, pKernelGsp, &pKernelGsp->nocatData.nocatBuffer);
            kgspPostNocatData(pGpu, pKernelGsp, pKernelGsp->pRpc->rpcHistory[pKernelGsp->pRpc->rpcHistoryCurrent].ts_start);

            kgspLogRpcDebugInfo(pGpu, pKernelGsp->pRpc, GSP_ERROR, pKernelGsp->bPollingForRpcResponse);
        }

        if (bFirstFatal)
        {
            kgspRcAndNotifyAllChannels(pGpu, pKernelGsp, GSP_ERROR, NV_TRUE);
            gpuMarkDeviceForReset(pGpu);
        }

        gpuCheckEccCounts_HAL(pGpu);

        NV_PRINTF(LEVEL_ERROR,
                  "**********************************************************************************\n");

        if (pGpu->getProperty(pGpu, PDB_PROP_GPU_SUPPORTS_TDR_EVENT))
        {
            NV_ASSERT_FAILED("GSP timed out. Triggering TDR.");
            gpuNotifySubDeviceEvent(pGpu, NV2080_NOTIFIERS_UCODE_RESET, NULL, 0, 0, 0);
        }
    }
    return bHealthy;
}

/*!
 * GSP Interrupt Service Routine
 *
 * @return 32-bit interrupt status AFTER all known interrupt-sources were
 *         serviced.
 */
NvU32
kgspService_TU102
(
    OBJGPU     *pGpu,
    KernelGsp  *pKernelGsp
)
{
    NvU32         intrStatus;
    KernelFalcon *pKernelFalcon = staticCast(pKernelGsp, KernelFalcon);

    // Get the IRQ status for sources routed to host
    intrStatus = kflcnGetPendingHostInterrupts(pGpu, pKernelFalcon);

    // Exit immediately if there is nothing to do
    if (intrStatus == 0)
    {
        NV_ASSERT_FAILED("KGSP service called when no KGSP interrupt pending\n");
        return 0;
    }

    if (!API_GPU_ATTACHED_SANITY_CHECK(pGpu))
    {
        NV_PRINTF(LEVEL_ERROR, "GPU is detached, bailing!\n");
        return 0;
    }

    if (intrStatus & DRF_DEF(_PFALCON, _FALCON_IRQSTAT, _HALT, _TRUE))
    {
        //
        // The _HALT is triggered by ucode as part of the CrashCat protocol to
        // signal the host that some handling is required. Clear the interrupt
        // before handling, so that once the GSP code continues, we won't miss
        // a second _HALT interrupt for the next step.
        //
        kflcnRegWrite_HAL(pGpu, pKernelFalcon, NV_PFALCON_FALCON_IRQSCLR,
            DRF_DEF(_PFALCON, _FALCON_IRQSCLR, _HALT, _SET));

        kgspDumpGspLogs(pKernelGsp, NV_FALSE);
        (void)kgspHealthCheck_HAL(pGpu, pKernelGsp);
#if defined(DEBUG)
        NV_PRINTF(LEVEL_ERROR, "GSP-RM entered into ICD\n");
        DBG_BREAKPOINT();
#endif
    }
    if (intrStatus & DRF_DEF(_PFALCON, _FALCON_IRQSTAT, _SWGEN0, _TRUE))
    {
        //
        // Clear edge triggered interrupt BEFORE (and never after)
        // servicing it to avoid race conditions.
        //
        kflcnRegWrite_HAL(pGpu, pKernelFalcon, NV_PFALCON_FALCON_IRQSCLR,
            DRF_DEF(_PFALCON, _FALCON_IRQSCLR, _SWGEN0, _SET));

        kgspRpcRecvEvents(pGpu, pKernelGsp);

        //
        // If lockdown has been engaged (as notified by an RPC event),
        // we shouldn't access any more GSP registers.
        //
        NV_CHECK_OR_RETURN(LEVEL_SILENT, !pKernelGsp->bInLockdown, 0);
    }

    kgspServiceFatalHwError_HAL(pGpu, pKernelGsp, intrStatus);

    if (intrStatus & kflcnGetEccInterruptMask_HAL(pGpu, pKernelFalcon))
    {
        kgspEccServiceEvent_HAL(pGpu, pKernelGsp);
    }

    //
    // Don't retrigger for fatal errors since they can't be cleared without an
    // engine reset, which results in an interrupt storm until reset
    //
    if (!pKernelGsp->bFatalError)
    {
        kflcnIntrRetrigger_HAL(pGpu, pKernelFalcon);
    }

    return kflcnGetPendingHostInterrupts(pGpu, pKernelFalcon);
}

static NvBool
_kgspIsProcessorSuspended
(
    OBJGPU  *pGpu,
    void    *pVoid
)
{
    KernelGsp *pKernelGsp = reinterpretCast(pVoid, KernelGsp *);
    NvU32 mailbox;

    // Check for LIBOS_INTERRUPT_PROCESSOR_SUSPENDED in mailbox
    mailbox = kflcnRegRead_HAL(pGpu, staticCast(pKernelGsp, KernelFalcon),
                               NV_PFALCON_FALCON_MAILBOX0);
    return (mailbox == 0x80000000);
}

NV_STATUS
kgspWaitForProcessorSuspend_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    return gpuTimeoutCondWait(pGpu, _kgspIsProcessorSuspended, pKernelGsp, NULL);
}

NvBool
kgspIsWpr2Up_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    NvU32 data = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
    NvU32 wpr2HiVal = DRF_VAL(_PFB, _PRI_MMU_WPR2_ADDR_HI, _VAL, data);
    return (wpr2HiVal != 0);
}

NV_STATUS
kgspWaitForGfwBootOk_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    NV_STATUS status = NV_OK;

    status = gpuWaitForGfwBootComplete_HAL(pGpu);
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "failed to wait for GFW boot complete: 0x%x VBIOS version %s\n",
                  status, pKernelGsp->vbiosVersionStr);
        NV_PRINTF(LEVEL_ERROR, "(the GPU may be in a bad state and may need to be reset)\n");
    }

    return status;
}

void
kgspFreeSuspendResumeData_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    // release sr meta data resources
    if (pKernelGsp->pSRMetaDescriptor != NULL)
    {
        memdescFree(pKernelGsp->pSRMetaDescriptor);
        memdescDestroy(pKernelGsp->pSRMetaDescriptor);
        pKernelGsp->pSRMetaDescriptor = NULL;
    }

    // release sr meta data resources
    if (pKernelGsp->pSRRadix3Descriptor != NULL)
    {
        memdescFree(pKernelGsp->pSRRadix3Descriptor);
        memdescDestroy(pKernelGsp->pSRRadix3Descriptor);
        pKernelGsp->pSRRadix3Descriptor = NULL;
    }
}

NV_STATUS
kgspPrepareSuspendResumeData_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    GspFwSRMeta gspfwSRMeta;
    NvP64 pVa = NvP64_NULL;
    NvP64 pPriv = NvP64_NULL;
    NV_STATUS nvStatus = NV_OK;

    // Fill in GspFwSRMeta structure
    portMemSet(&gspfwSRMeta, 0, sizeof(gspfwSRMeta));
    gspfwSRMeta.magic                   = GSP_FW_SR_META_MAGIC;
    gspfwSRMeta.revision                = GSP_FW_SR_META_REVISION;
    // Region to be saved is from start of WPR2 till end of frts.
    gspfwSRMeta.sizeOfSuspendResumeData = (pKernelGsp->pWprMeta->frtsOffset + pKernelGsp->pWprMeta->frtsSize) -
                                          (pKernelGsp->pWprMeta->nonWprHeapOffset + pKernelGsp->pWprMeta->nonWprHeapSize);
    gspfwSRMeta.flags                   = pKernelGsp->pWprMeta->flags;

    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         kgspCreateRadix3(pGpu,
                                          pKernelGsp,
                                          &pKernelGsp->pSRRadix3Descriptor,
                                          NULL,
                                          NULL,
                                          gspfwSRMeta.sizeOfSuspendResumeData),
                         exit_fail_cleanup);

    gspfwSRMeta.sysmemAddrOfSuspendResumeData = memdescGetPhysAddr(pKernelGsp->pSRRadix3Descriptor, AT_GPU, 0);

    // Create SR Metadata Area
    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         memdescCreate(&pKernelGsp->pSRMetaDescriptor,
                                       pGpu,
                                       sizeof(GspFwSRMeta),
                                       256,
                                       NV_TRUE,
                                       ADDR_SYSMEM,
                                       NV_MEMORY_UNCACHED,
                                       MEMDESC_FLAGS_NONE),
                         exit_fail_cleanup);

    memdescTagAlloc(nvStatus, NV_FB_ALLOC_RM_INTERNAL_OWNER_SR_METADATA,
                    pKernelGsp->pSRMetaDescriptor);
    NV_ASSERT_OK_OR_GOTO(nvStatus, nvStatus,
                         exit_fail_cleanup);

    // Copy SR Metadata Structure
    NV_ASSERT_OK_OR_GOTO(nvStatus,
                         memdescMap(pKernelGsp->pSRMetaDescriptor,
                                    0,
                                    memdescGetSize(pKernelGsp->pSRMetaDescriptor),
                                    NV_TRUE,
                                    NV_PROTECT_WRITEABLE,
                                    &pVa,
                                    &pPriv),
                         exit_fail_cleanup);

    portMemCopy(pVa, sizeof(gspfwSRMeta), &gspfwSRMeta, sizeof(gspfwSRMeta));

    memdescUnmap(pKernelGsp->pSRMetaDescriptor,
                 NV_TRUE,
                 pVa, pPriv);

    return nvStatus;

exit_fail_cleanup:
    kgspFreeSuspendResumeData_HAL(pGpu, pKernelGsp);
    return nvStatus;
}

void
kgspDumpMailbox_TU102
(
    OBJGPU    *pGpu,
    KernelGsp *pKernelGsp
)
{
    NvU32 idx;
    NvU32 data;

    for (idx = 0; idx < NV_PGSP_MAILBOX__SIZE_1; idx++)
    {
        data = GPU_REG_RD32(pGpu, NV_PGSP_MAILBOX(idx));
        NV_PRINTF(LEVEL_ERROR, "GSP: MAILBOX(%d) = 0x%08X\n", idx, data);
    }
}

void
kgspReadEmem_TU102
(
    KernelGsp *pKernelGsp,
    NvU64      offset,
    NvU64      size,
    void      *pBuf
)
{
    NvU32 ememMask = DRF_SHIFTMASK(NV_PGSP_EMEMC_OFFS) | DRF_SHIFTMASK(NV_PGSP_EMEMC_BLK);
    OBJGPU *pGpu = ENG_GET_GPU(pKernelGsp);
    NvU32 limit = size - NVBIT(DRF_SHIFT(NV_PGSP_EMEMC_OFFS));
    NvU32 *pBuffer = pBuf;

    portMemSet(pBuf, 0, size);

#if defined(DEBUG) || defined(DEVELOP)
    NV_ASSERT_OR_RETURN_VOID((offset & ~ememMask) == 0);
    NV_ASSERT_OR_RETURN_VOID(limit <= ememMask);
    NV_ASSERT_OR_RETURN_VOID(offset + limit <= ememMask);
#else
    NV_CHECK_OR_RETURN_VOID(LEVEL_SILENT, (offset & ~ememMask) == 0);
    NV_CHECK_OR_RETURN_VOID(LEVEL_SILENT, limit <= ememMask);
    NV_CHECK_OR_RETURN_VOID(LEVEL_SILENT, offset + limit <= ememMask);
#endif

    GPU_REG_WR32(pGpu, NV_PGSP_EMEMC(pKernelGsp->ememPort),
                 offset | DRF_DEF(_PGSP, _EMEMC, _AINCR, _TRUE));

    for (NvU32 idx = 0; idx < size / sizeof(NvU32); idx++)
        pBuffer[idx] = GPU_REG_RD32(pGpu, NV_PGSP_EMEMD(pKernelGsp->ememPort));
}
