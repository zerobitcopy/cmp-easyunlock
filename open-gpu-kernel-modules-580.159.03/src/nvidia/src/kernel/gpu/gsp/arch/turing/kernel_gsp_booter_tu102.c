/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu/gsp/kernel_gsp.h"

#include "gpu/gpu.h"
#include "gpu/falcon/kernel_falcon.h"
#include "gpu/sec2/kernel_sec2.h"

#include <string.h>

#include "published/turing/tu102/dev_falcon_v4.h"
#include "published/turing/tu102/dev_fb.h"

#define GA100_SEC2_FALCON_EXCI                  0x008400d0
#define GA100_SEC2_FALCON_DEBUGINFO             0x00840094
#define GA100_SEC2_FALCON_IRQSTAT_ALIAS         0x008400fc
#define GA100_SEC2_FALCON_HWCFG1                0x0084012c
#define GA100_SEC2_FALCON_CG2                   0x00840134
#define GA100_SEC2_FALCON_TRACEIDX              0x00840148
#define GA100_SEC2_FALCON_TRACEPC               0x0084014c
#define GA100_SEC2_FALCON_EXTERRADDR            0x00840168
#define GA100_SEC2_FALCON_EXTERRSTAT            0x0084016c
#define GA100_SEC2_FALCON_SCTL                  0x00840240
#define GA100_SEC2_FALCON_SSTAT                 0x00840244
#define GA100_SEC2_FALCON_SCTL1                 0x00840250
#define GA100_SEC2_ENGCTL_PRIV_LEVEL_MASK       0x0084027c
#define GA100_SEC2_IMEM_PRIV_LEVEL_MASK         0x00840280
#define GA100_SEC2_DMEM_PRIV_LEVEL_MASK         0x00840284
#define GA100_SEC2_EXE_PRIV_LEVEL_MASK          0x0084028c
#define GA100_SEC2_IRQTMR_PRIV_LEVEL_MASK       0x00840290
#define GA100_SEC2_ENGINE                       0x008403c0
#define GA100_SEC2_RESET_PRIV_LEVEL_MASK        0x008403c4
#define GA100_SEC2_POST_REG_I_12000             0x00840480
#define GA100_SEC2_POST_REG_I_12100             0x00840484
#define GA100_SEC2_POST_REG_I_12400             0x00840490
#define GA100_SEC2_POST_REG_I_12600             0x00840498
#define GA100_SEC2_POST_REG_I_1C000             0x00840700
#define GA100_SEC2_POST_REG_I_1C300             0x0084070c
#define GA100_GSP_STATUS                        0x00110040
#define GA100_LMR                               0x00100ce0
#define GA100_MUTEX3_OWNER                      0x001e0e18
#define GA100_FEAT_OVR_ECC_PLM                  0x00823800
#define GA100_FEAT_OVR_PLM                      0x00823804
#define GA100_FEAT_READOUT_1                    0x00823818
#define GA100_FEAT_OVR_SM_SPD                   0x0082381c
#define GA100_FEAT_OVR_SM_SPD_1                 0x00823820
#define GA100_FUSE_SS_PLM                       0x008200fc
#define GA100_CFG1                              0x009a0204
#define GA100_WPR_LOCK_CFG_PLM                  0x001fa7c8
#define GA100_WPR_CFG_PLM                       0x001fa7cc
#define GA100_MAILBOX_PROBE_VALUE               0xa55a5aa5

static void
s_dumpGa100BooterLoadPost_TU102
(
    OBJGPU *pGpu,
    KernelFalcon *pKernelFlcn,
    NvU32 stage,
    NV_STATUS status,
    NvU32 resultMailbox0,
    NvU32 resultMailbox1
)
{
    NvU32 liveMailbox0 = kflcnRegRead_HAL(
        pGpu, pKernelFlcn, NV_PFALCON_FALCON_MAILBOX0);
    NvU32 liveMailbox1 = kflcnRegRead_HAL(
        pGpu, pKernelFlcn, NV_PFALCON_FALCON_MAILBOX1);

    // Keep this as one grep-friendly record immediately after Booter Load.
    // GA100_MUTEX3_OWNER is read-only owner state; do not read the token
    // allocation register at 0x001e0e04 from this diagnostic path.
    NV_PRINTF(LEVEL_ERROR,
              "GA100_BOOTER_LOAD_POST stage=%u status=0x%x "
              "result_mailbox0=0x%08x result_mailbox1=0x%08x "
              "live_mailbox0=0x%08x live_mailbox1=0x%08x "
              "mutex3_owner=0x%02x cfg1=0x%08x lmr=0x%08x "
              "feat_ecc_plm=0x%08x feat_plm=0x%08x readout1=0x%08x "
              "ss0=0x%08x ss1=0x%08x fuse_ss_plm=0x%08x "
              "wpr_lock_cfg_plm=0x%08x wpr_cfg_plm=0x%08x "
              "wpr2_lo=0x%08x wpr2_hi=0x%08x\n",
              stage, status,
              resultMailbox0, resultMailbox1,
              liveMailbox0, liveMailbox1,
              GPU_REG_RD32(pGpu, GA100_MUTEX3_OWNER) & 0xffU,
              GPU_REG_RD32(pGpu, GA100_CFG1),
              GPU_REG_RD32(pGpu, GA100_LMR),
              GPU_REG_RD32(pGpu, GA100_FEAT_OVR_ECC_PLM),
              GPU_REG_RD32(pGpu, GA100_FEAT_OVR_PLM),
              GPU_REG_RD32(pGpu, GA100_FEAT_READOUT_1),
              GPU_REG_RD32(pGpu, GA100_FEAT_OVR_SM_SPD),
              GPU_REG_RD32(pGpu, GA100_FEAT_OVR_SM_SPD_1),
              GPU_REG_RD32(pGpu, GA100_FUSE_SS_PLM),
              GPU_REG_RD32(pGpu, GA100_WPR_LOCK_CFG_PLM),
              GPU_REG_RD32(pGpu, GA100_WPR_CFG_PLM),
              GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO),
              GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI));
}

static void
s_dumpGa100BooterState_TU102
(
    OBJGPU *pGpu,
    KernelFalcon *pKernelFlcn,
    NvU32 stage,
    const char *point,
    NV_STATUS status,
    NvU32 requestedMailbox0,
    NvU32 requestedMailbox1
)
{
    if ((strcmp(point, "PRE_RESET") == 0) ||
        (strcmp(point, "POST_RESET") == 0))
    {
        return;
    }

    NvU32 mailbox0;
    NvU32 mailbox1;
    NvU32 cpuctl;
    NvU32 irqstat;
    NvU32 irqstatAlias;
    NvU32 irqmode;
    NvU32 irqmask;
    NvU32 irqdest;
    NvU32 bootvec;
    NvU32 os;
    NvU32 rm;
    NvU32 exci;
    NvU32 debuginfo;
    NvU32 hwcfg;
    NvU32 hwcfg1;
    NvU32 hwcfg2;
    NvU32 cg2;
    NvU32 traceidx;
    NvU32 tracepc;
    NvU32 exterraddr;
    NvU32 exterrstat;
    NvU32 sctl;
    NvU32 sstat;
    NvU32 sctl1;
    NvU32 engine;
    NvU32 engctlMask;
    NvU32 imemMask;
    NvU32 dmemMask;
    NvU32 exeMask;
    NvU32 irqtimerMask;
    NvU32 resetPrivMask;
    NvU32 dmactl;
    NvU32 dmatrfbase;
    NvU32 dmatrfbase1;
    NvU32 dmatrfmoffs;
    NvU32 dmatrfcmd;
    NvU32 dmatrffboffs;
    NvU32 imemc;
    NvU32 dmemc;
    NvU32 wpr2Lo;
    NvU32 wpr2Hi;
    NvU32 gspStatus;
    NvU32 lmr;
    NvU32 featPlm;
    NvU32 ss0;
    NvU32 ss1;
    NvU32 fuseSsPlm;
    NvU32 cfg1;

    mailbox0 = kflcnRegRead_HAL(pGpu, pKernelFlcn,
                                NV_PFALCON_FALCON_MAILBOX0);
    mailbox1 = kflcnRegRead_HAL(pGpu, pKernelFlcn,
                                NV_PFALCON_FALCON_MAILBOX1);
    cpuctl = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_CPUCTL);
    irqstat = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_IRQSTAT);
    irqstatAlias = GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_IRQSTAT_ALIAS);
    irqmode = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_IRQMODE);
    irqmask = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_IRQMASK);
    irqdest = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_IRQDEST);
    bootvec = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_BOOTVEC);
    os = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_OS);
    rm = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_RM);
    exci = GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_EXCI);
    debuginfo = GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_DEBUGINFO);
    hwcfg = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_HWCFG);
    hwcfg1 = GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_HWCFG1);
    hwcfg2 = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_HWCFG2);
    cg2 = GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_CG2);
    traceidx = GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_TRACEIDX);
    tracepc = GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_TRACEPC);
    exterraddr = GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_EXTERRADDR);
    exterrstat = GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_EXTERRSTAT);
    sctl = GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_SCTL);
    sstat = GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_SSTAT);
    sctl1 = GPU_REG_RD32(pGpu, GA100_SEC2_FALCON_SCTL1);
    engine = GPU_REG_RD32(pGpu, GA100_SEC2_ENGINE);
    engctlMask = GPU_REG_RD32(pGpu, GA100_SEC2_ENGCTL_PRIV_LEVEL_MASK);
    imemMask = GPU_REG_RD32(pGpu, GA100_SEC2_IMEM_PRIV_LEVEL_MASK);
    dmemMask = GPU_REG_RD32(pGpu, GA100_SEC2_DMEM_PRIV_LEVEL_MASK);
    exeMask = GPU_REG_RD32(pGpu, GA100_SEC2_EXE_PRIV_LEVEL_MASK);
    irqtimerMask = GPU_REG_RD32(pGpu, GA100_SEC2_IRQTMR_PRIV_LEVEL_MASK);
    resetPrivMask = GPU_REG_RD32(pGpu, GA100_SEC2_RESET_PRIV_LEVEL_MASK);
    dmactl = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_DMACTL);
    dmatrfbase = kflcnRegRead_HAL(pGpu, pKernelFlcn,
                                  NV_PFALCON_FALCON_DMATRFBASE);
    dmatrfbase1 = kflcnRegRead_HAL(pGpu, pKernelFlcn,
                                   NV_PFALCON_FALCON_DMATRFBASE1);
    dmatrfmoffs = kflcnRegRead_HAL(pGpu, pKernelFlcn,
                                   NV_PFALCON_FALCON_DMATRFMOFFS);
    dmatrfcmd = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_DMATRFCMD);
    dmatrffboffs = kflcnRegRead_HAL(pGpu, pKernelFlcn,
                                    NV_PFALCON_FALCON_DMATRFFBOFFS);
    imemc = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_IMEMC(0));
    dmemc = kflcnRegRead_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_DMEMC(0));

    wpr2Lo = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_LO);
    wpr2Hi = GPU_REG_RD32(pGpu, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
    gspStatus = GPU_REG_RD32(pGpu, GA100_GSP_STATUS);
    lmr = GPU_REG_RD32(pGpu, GA100_LMR);
    featPlm = GPU_REG_RD32(pGpu, GA100_FEAT_OVR_PLM);
    ss0 = GPU_REG_RD32(pGpu, GA100_FEAT_OVR_SM_SPD);
    ss1 = GPU_REG_RD32(pGpu, GA100_FEAT_OVR_SM_SPD_1);
    fuseSsPlm = GPU_REG_RD32(pGpu, GA100_FUSE_SS_PLM);
    cfg1 = GPU_REG_RD32(pGpu, GA100_CFG1);

    NV_PRINTF(LEVEL_ERROR,
              "GA100_BOOTER_STATE stage=%u point=%s status=0x%x\n"
              "  mailbox   requested0=0x%08x requested1=0x%08x current0=0x%08x current1=0x%08x\n"
              "  falcon    cpuctl=0x%08x bootvec=0x%08x os=0x%08x rm=0x%08x\n"
              "  irq       stat=0x%08x stat_alias=0x%08x mode=0x%08x mask=0x%08x dest=0x%08x\n"
              "  sec2      exci=0x%08x debug=0x%08x traceidx=0x%08x tracepc=0x%08x\n"
              "            exterraddr=0x%08x exterrstat=0x%08x\n",
              stage, point, status,
              requestedMailbox0, requestedMailbox1, mailbox0, mailbox1,
              cpuctl, bootvec, os, rm,
              irqstat, irqstatAlias, irqmode, irqmask, irqdest,
              exci, debuginfo, traceidx, tracepc,
              exterraddr, exterrstat);

    NV_PRINTF(LEVEL_ERROR,
              "GA100_BOOTER_REGS  stage=%u point=%s\n"
              "  config    hwcfg=0x%08x hwcfg1=0x%08x hwcfg2=0x%08x cg2=0x%08x\n"
              "  secure    sctl=0x%08x sstat=0x%08x sctl1=0x%08x engine=0x%08x\n"
              "  privmask  engctl=0x%08x imem=0x%08x dmem=0x%08x exe=0x%08x irqtimer=0x%08x reset=0x%08x\n"
              "  dma       ctl=0x%08x base=0x%08x base1=0x%08x moffs=0x%08x cmd=0x%08x fboffs=0x%08x\n"
              "  memport   imemc0=0x%08x dmemc0=0x%08x\n"
              "  memory    wpr2_lo=0x%08x wpr2_hi=0x%08x gsp_status=0x%08x lmr=0x%08x\n"
              "  feature   feat_plm=0x%08x ss0=0x%08x ss1=0x%08x fuse_ss_plm=0x%08x cfg1=0x%08x\n",
              stage, point,
              hwcfg, hwcfg1, hwcfg2, cg2,
              sctl, sstat, sctl1, engine,
              engctlMask, imemMask, dmemMask, exeMask, irqtimerMask,
              resetPrivMask,
              dmactl, dmatrfbase, dmatrfbase1, dmatrfmoffs, dmatrfcmd,
              dmatrffboffs,
              imemc, dmemc,
              wpr2Lo, wpr2Hi, gspStatus, lmr,
              featPlm, ss0, ss1, fuseSsPlm, cfg1);

    if (((stage == 1) || (stage == 2)) &&
        (strcmp(point, "POST_EXECUTE") == 0))
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_BOOTER_STAGE%u_POST_INTERNAL\n"
                  "  I[0x1c000] @0x00840700=0x%08x I[0x1c300] @0x0084070c=0x%08x\n"
                  "  I[0x12000] @0x00840480=0x%08x I[0x12100] @0x00840484=0x%08x\n"
                  "  I[0x12400] @0x00840490=0x%08x I[0x12600] @0x00840498=0x%08x\n",
                  stage,
                  GPU_REG_RD32(pGpu, GA100_SEC2_POST_REG_I_1C000),
                  GPU_REG_RD32(pGpu, GA100_SEC2_POST_REG_I_1C300),
                  GPU_REG_RD32(pGpu, GA100_SEC2_POST_REG_I_12000),
                  GPU_REG_RD32(pGpu, GA100_SEC2_POST_REG_I_12100),
                  GPU_REG_RD32(pGpu, GA100_SEC2_POST_REG_I_12400),
                  GPU_REG_RD32(pGpu, GA100_SEC2_POST_REG_I_12600));
    }
}

static NvBool
s_probeGa100MailboxWrite_TU102
(
    OBJGPU *pGpu,
    KernelFalcon *pKernelFlcn,
    NvU32 stage,
    const char *point
)
{
    NvU32 original;
    NvU32 readback;
    NvU32 restoreReadback;

    original = kflcnRegRead_HAL(pGpu, pKernelFlcn,
                                NV_PFALCON_FALCON_MAILBOX0);
    kflcnRegWrite_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_MAILBOX0,
                      GA100_MAILBOX_PROBE_VALUE);
    readback = kflcnRegRead_HAL(pGpu, pKernelFlcn,
                                NV_PFALCON_FALCON_MAILBOX0);
    kflcnRegWrite_HAL(pGpu, pKernelFlcn, NV_PFALCON_FALCON_MAILBOX0,
                      original);
    restoreReadback = kflcnRegRead_HAL(pGpu, pKernelFlcn,
                                       NV_PFALCON_FALCON_MAILBOX0);

    NV_PRINTF(LEVEL_ERROR,
              "GA100_MAILBOX_PROBE stage=%u point=%s requested=0x%08x readback=0x%08x restore=0x%08x restore_readback=0x%08x writable=%u\n",
              stage, point, GA100_MAILBOX_PROBE_VALUE, readback,
              original, restoreReadback,
              readback == GA100_MAILBOX_PROBE_VALUE);

    return readback == GA100_MAILBOX_PROBE_VALUE;
}

static NV_STATUS
s_executeBooterUcode_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    KernelGspFlcnUcode *pBooterUcode,
    KernelFalcon *pKernelFlcn,
    const NvU32 mailbox0Arg,
    const NvU32 mailbox1Arg,
    NvU32 *pMailbox0Result,
    NvU32 *pMailbox1Result
)
{
    NV_STATUS status;
    NvU32 mailbox0, mailbox1;

    NV_ASSERT_OR_RETURN(pBooterUcode != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pKernelFlcn != NULL, NV_ERR_INVALID_STATE);

    mailbox0 = kflcnRegRead_HAL(pGpu, pKernelFlcn,
                                NV_PFALCON_FALCON_MAILBOX0);
    mailbox1 = kflcnRegRead_HAL(pGpu, pKernelFlcn,
                                NV_PFALCON_FALCON_MAILBOX1);

    NV_PRINTF(LEVEL_INFO,
              "before Booter mailbox0 0x%08x, mailbox1 0x%08x\n",
              mailbox0, mailbox1);

    mailbox0 = mailbox0Arg;
    mailbox1 = mailbox1Arg;

    NV_PRINTF(LEVEL_INFO,
              "starting Booter with mailbox0 0x%08x, mailbox1 0x%08x\n",
              mailbox0, mailbox1);

    pKernelGsp->bLibosLogsPollingEnabled = NV_FALSE;

    status = kgspExecuteHsFalcon_HAL(pGpu, pKernelGsp,
                                     pBooterUcode, pKernelFlcn,
                                     &mailbox0, &mailbox1);

    pKernelGsp->bLibosLogsPollingEnabled = NV_TRUE;

    NV_PRINTF(LEVEL_INFO,
              "after Booter mailbox0 0x%08x, mailbox1 0x%08x\n",
              mailbox0, mailbox1);

    if (pMailbox0Result != NULL)
        *pMailbox0Result = mailbox0;
    if (pMailbox1Result != NULL)
        *pMailbox1Result = mailbox1;

    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "failed to execute Booter: status 0x%x, mailbox 0x%x\n",
                  status, mailbox0);
    }

    return status;
}

static NV_STATUS
s_validateBooterResult_TU102
(
    NV_STATUS status,
    NvU32 mailbox0
)
{
    if (status != NV_OK)
        return status;

    if (mailbox0 != 0)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "Booter failed with non-zero error code: 0x%x\n",
                  mailbox0);
        return NV_ERR_GENERIC;
    }

    return NV_OK;
}

NV_STATUS
kgspExecuteBooterLoadStage_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    const NvU64 sysmemAddrOfData,
    NvU32 stage,
    NvBool bExpectedStage1Completion
)
{
    NV_STATUS status;
    NvU32 mailbox0 = 0, mailbox1 = 0;
    NvU32 resultMailbox0 = 0, resultMailbox1 = 0;
    KernelSec2 *pKernelSec2 = GPU_GET_KERNEL_SEC2(pGpu);
    KernelFalcon *pKernelSec2Falcon =
        staticCast(pKernelSec2, KernelFalcon);
    NvBool bGa100 = gpuIsImplementation(pGpu, HAL_IMPL_GA100);

    NV_ASSERT_OR_RETURN(pKernelGsp->pBooterLoadUcode != NULL,
                        NV_ERR_INVALID_STATE);

    if (sysmemAddrOfData != 0)
    {
        mailbox0 = NvU64_LO32(sysmemAddrOfData);
        mailbox1 = NvU64_HI32(sysmemAddrOfData);
    }

    if (bGa100 && (stage != 0))
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE: Booter Load stage=%u metadata_pa=0x%016llx expected_stage1=%u\n",
                  stage, sysmemAddrOfData, bExpectedStage1Completion);
    }
    else
    {
        NV_PRINTF(LEVEL_INFO,
                  "executing Booter Load, sysmemAddrOfData 0x%llx\n",
                  sysmemAddrOfData);
    }

    NV_PRINTF(LEVEL_ERROR,
              "GA100_BOOTER_LOAD_BEGIN role=%s stage=%u metadata_pa=0x%016llx\n",
              (stage == 1) ? "EXPERIMENT_PRIMARY" :
              ((stage == 2) ? "EXPERIMENT_ALT" : "SINGLE_STAGE"),
              stage, sysmemAddrOfData);

    if (bGa100)
    {
        s_dumpGa100BooterState_TU102(pGpu, pKernelSec2Falcon, stage,
                                     "PRE_RESET", NV_OK,
                                     mailbox0, mailbox1);
    }

    status = kflcnReset_HAL(pGpu, pKernelSec2Falcon);
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE: SEC2 reset failed at stage %u: 0x%x\n",
                  stage, status);
        return status;
    }

    if (bGa100)
    {
        s_dumpGa100BooterState_TU102(pGpu, pKernelSec2Falcon, stage,
                                     "POST_RESET", status,
                                     mailbox0, mailbox1);

        if ((stage == 2) &&
            !s_probeGa100MailboxWrite_TU102(pGpu, pKernelSec2Falcon,
                                            stage, "POST_RESET"))
        {
            NV_PRINTF(LEVEL_ERROR,
                      "GA100_TWO_STAGE: refusing stage 2 because host mailbox write permission was not restored\n");
            return NV_ERR_INVALID_STATE;
        }

        s_dumpGa100BooterState_TU102(pGpu, pKernelSec2Falcon, stage,
                                     "PRE_EXECUTE", NV_OK,
                                     mailbox0, mailbox1);
    }

    status = s_executeBooterUcode_TU102(pGpu, pKernelGsp,
                                        pKernelGsp->pBooterLoadUcode,
                                        pKernelSec2Falcon,
                                        mailbox0, mailbox1,
                                        &resultMailbox0,
                                        &resultMailbox1);

    if (bGa100)
    {
        s_dumpGa100BooterLoadPost_TU102(
            pGpu, pKernelSec2Falcon, stage, status,
            resultMailbox0, resultMailbox1);

        s_dumpGa100BooterState_TU102(pGpu, pKernelSec2Falcon, stage,
                                     "POST_EXECUTE", status,
                                     mailbox0, mailbox1);

        if (stage == 1)
        {
            // Diagnostic only.  The stage-2 gate is the probe after SEC2 reset.
            s_probeGa100MailboxWrite_TU102(pGpu, pKernelSec2Falcon,
                                           stage, "POST_EXECUTE");
        }
    }

    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "failed to execute Booter Load stage %u: 0x%x\n",
                  stage, status);
        return status;
    }

    if (bExpectedStage1Completion)
    {
        if (resultMailbox0 != 0U)
        {
            NV_PRINTF(LEVEL_ERROR,
                      "GA100_TWO_STAGE: stage 1 halted with Booter error "
                      "mailbox0=0x%08x mailbox1=0x%08x; refusing teardown/stage 2\n",
                      resultMailbox0, resultMailbox1);
            return NV_ERR_INVALID_STATE;
        }

        NV_PRINTF(LEVEL_ERROR,
                  "GA100_TWO_STAGE: stage 1 halted as expected; mailbox0=0x%08x mailbox1=0x%08x, preparing same-image stage 2\n",
                  resultMailbox0, resultMailbox1);
        return KGSP_BOOTER_STATUS_STAGE1_COMPLETE;
    }

    return s_validateBooterResult_TU102(status, resultMailbox0);
}

NV_STATUS
kgspExecuteBooterLoad_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    const NvU64 sysmemAddrOfData
)
{
    return kgspExecuteBooterLoadStage_TU102(pGpu, pKernelGsp,
                                            sysmemAddrOfData, 0,
                                            NV_FALSE);
}

NV_STATUS
kgspExecuteBooterUnloadIfNeeded_TU102
(
    OBJGPU *pGpu,
    KernelGsp *pKernelGsp,
    const NvU64 sysmemAddrOfSuspendResumeData
)
{
    NV_STATUS status;
    KernelSec2 *pKernelSec2 = GPU_GET_KERNEL_SEC2(pGpu);
    NvU32 mailbox0 = 0xFF, mailbox1 = 0xFF;
    NvU32 resultMailbox0 = 0, resultMailbox1 = 0;

    if (IS_GPU_GC6_STATE_ENTERING(pGpu))
    {
        mailbox0 = mailbox1 = 0xdeaddead;
    }

    if (API_GPU_IN_RESET_SANITY_CHECK(pGpu))
        return NV_ERR_GPU_IN_FULLCHIP_RESET;

    if (!kgspIsWpr2Up_HAL(pGpu, pKernelGsp))
    {
        NV_PRINTF(LEVEL_INFO,
                  "skipping executing Booter Unload as WPR2 is not up\n");
        return NV_OK;
    }

    NV_PRINTF(LEVEL_INFO, "executing Booter Unload\n");
    NV_ASSERT_OR_RETURN(pKernelGsp->pBooterUnloadUcode != NULL,
                        NV_ERR_INVALID_STATE);

    NV_ASSERT_OK(kflcnReset_HAL(pGpu,
                               staticCast(pKernelSec2, KernelFalcon)));

    if (sysmemAddrOfSuspendResumeData != 0)
    {
        mailbox0 = NvU64_LO32(sysmemAddrOfSuspendResumeData);
        mailbox1 = NvU64_HI32(sysmemAddrOfSuspendResumeData);
    }

    status = s_executeBooterUcode_TU102(pGpu, pKernelGsp,
                                        pKernelGsp->pBooterUnloadUcode,
                                        staticCast(pKernelSec2, KernelFalcon),
                                        mailbox0, mailbox1,
                                        &resultMailbox0,
                                        &resultMailbox1);
    status = s_validateBooterResult_TU102(status, resultMailbox0);
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "failed to execute Booter Unload: 0x%x\n", status);
        return status;
    }

    if (IS_GPU_GC6_STATE_ENTERING(pGpu))
    {
        if (!kgspIsWpr2Up_HAL(pGpu, pKernelGsp))
        {
            NV_PRINTF(LEVEL_ERROR,
                      "failed to execute Booter Unload: WPR2 is cleared despite GC6\n");
            return NV_ERR_GENERIC;
        }
    }
    else
    {
        if (kgspIsWpr2Up_HAL(pGpu, pKernelGsp))
        {
            NV_PRINTF(LEVEL_ERROR,
                      "failed to execute Booter Unload: WPR2 is still up\n");
            return NV_ERR_GENERIC;
        }
    }

    return status;
}
