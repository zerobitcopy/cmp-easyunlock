#!/bin/bash
set -euo pipefail

# ===================== Config Section =====================
DRV_VERSION="580.159.03"
SRC_DIR="./open-gpu-kernel-modules-${DRV_VERSION}"
FW_DIR="./firmware"
FW_TYPE=""
KERN_VER=$(uname -r)
# Compiled kernel objects stay inside source folder, no copy to system module dir
KO_PATH="${SRC_DIR}/kernel-open"
# NVIDIA run installer download URL
NVIDIA_RUN_URL="https://download.nvidia.com/XFree86/Linux-x86_64/${DRV_VERSION}/NVIDIA-Linux-x86_64-${DRV_VERSION}.run"
NVIDIA_RUN_FILE="./NVIDIA-Linux-x86_64-${DRV_VERSION}.run"
# Browser User-Agent to bypass 403 restriction
WGET_UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"
# ==========================================================

# ===================== Parse Input Arguments =====================
show_usage() {
    echo -e "\033[31mUsage: $0 --vram=<mode>\033[0m"
    echo "Valid vram modes:"
    echo "  --vram=8g_to_32g   -> 8GB card → 32GB unlock"
    echo "  --vram=8g_to_64g   -> 8GB card → 64GB unlock"
    echo "  --vram=10g_to_40g  -> 10GB card → 40GB unlock"
    echo "  --vram=10g_to_80g  -> 10GB card → 80GB unlock"
    exit 1
}

if [[ $# -ne 1 ]]; then
    echo -e "\033[31mERROR: Missing required --vram parameter!\033[0m"
    show_usage
fi

VRAM_ARG="$1"
if [[ ! "${VRAM_ARG}" =~ ^--vram= ]]; then
    echo -e "\033[31mERROR: Invalid argument, must start with --vram=\033[0m"
    show_usage
fi

VRAM_MODE="${VRAM_ARG#--vram=}"

# get FW_TYPE
case "${VRAM_MODE}" in
    "8g_to_32g")
        FW_TYPE="gsp_tu10x.bin.32"
        ;;
    "8g_to_64g")
        FW_TYPE="gsp_tu10x.bin.64"
        ;;
    "10g_to_40g")
        FW_TYPE="gsp_tu10x.bin.40"
        ;;
    "10g_to_80g")
        FW_TYPE="gsp_tu10x.bin.80"
        ;;
    *)
        echo -e "\033[31mERROR: Unknown vram mode '${VRAM_MODE}'\033[0m"
        show_usage
        ;;
esac

echo -e "\033[32m===== NVIDIA OpenGPU Build & Temporary Insmod Loader Script =====\033[0m"
echo "Kernel Version: ${KERN_VER}"
echo "Driver Version: ${DRV_VERSION}"
echo "Source Directory: ${SRC_DIR}"
echo "KO Path (direct insmod): ${KO_PATH}"
echo "Selected VRAM Mode: ${VRAM_MODE}"
echo "Target Firmware File: ${FW_TYPE}"
echo -e "\033[33mNote: KO files will NOT be copied to system module directories, only temporary loaded. Original driver restores after reboot.\033[0m"

# Require root privilege
if [ "$(id -u)" -ne 0 ]; then
    echo -e "\033[31mERROR: Run this script with sudo -i or root account\033[0m"
    exit 1
fi

# Function: Check if target NVIDIA driver version is installed
check_nvidia_driver_installed() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "\033[31m❌ nvidia-smi not found, NVIDIA driver missing. Will download & install.\033[0m"
        return 1
    fi

    local INSTALLED_VER
    INSTALLED_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | xargs)
    # Empty version check
    if [[ -z "${INSTALLED_VER}" ]]; then
        echo -e "\033[31m❌ No valid driver version output from nvidia-smi, reinstall required.\033[0m"
        return 1
    fi

    if [[ "${INSTALLED_VER}" == "${DRV_VERSION}" ]]; then
        echo -e "\033[32m✅ Detected matching driver version ${INSTALLED_VER}\033[0m"
        return 0
    else
        echo -e "\033[31m❌ Current driver: ${INSTALLED_VER}, Target: ${DRV_VERSION}. Reinstall required.\033[0m"
        return 1
    fi
}

# Function: Fully unload NVIDIA kernel modules and terminate related processes
clean_nvidia_modules() {
    echo -e "\033[34m[Cleanup] Switch to multi-user target, stop services and kill NVIDIA processes\033[0m"
    # Switch to text console, close X/Wayland
    systemctl isolate multi-user.target
    # Stop related services
    systemctl stop nvidia-dcgm || true
    systemctl stop nvidia-persistenced || true
    # Force kill occupied processes
    pkill -9 nv-hostengine 2>/dev/null || true
    pkill -9 nvidia-persistenced 2>/dev/null || true
    pkill -9 nvidia-smi 2>/dev/null || true
    pkill -9 nvidia-settings 2>/dev/null || true

    # Unload kernel modules in reverse dependency order
    modprobe -r nvidia-drm 2>/dev/null || true
    modprobe -r nvidia-modeset 2>/dev/null || true
    modprobe -r nvidia-uvm 2>/dev/null || true
    modprobe -r nvidia 2>/dev/null || true

    # Double check residual modules
    if lsmod | grep -q nvidia; then
        echo -e "\033[31m❌ NVIDIA kernel modules still loaded, cannot proceed installation!\033[0m"
        echo "Residual module list:"
        lsmod | grep nvidia
        exit 1
    fi
    echo -e "\033[32m✅ All NVIDIA kernel modules unloaded completely\033[0m"
}

# Function: Download and install NVIDIA run driver
install_nvidia_run_driver() {
    # Mandatory full cleanup before driver installation
    clean_nvidia_modules

    echo -e "\033[34m[Driver Install] Start downloading NVIDIA ${DRV_VERSION} run installer\033[0m"
    # Install download & build dependencies
    apt update && apt install -y wget build-essential linux-headers-${KERN_VER} libelf-dev pkg-config

    # Download run package if missing
    if [ ! -f "${NVIDIA_RUN_FILE}" ]; then
        wget -c -L --user-agent="${WGET_UA}" "${NVIDIA_RUN_URL}" -O "${NVIDIA_RUN_FILE}" || {
            echo -e "\033[31mDownload failed, remove incomplete file and exit\033[0m"
            rm -f "${NVIDIA_RUN_FILE}"
            exit 1
        }
    fi

    chmod +x "${NVIDIA_RUN_FILE}"

    echo -e "\033[34m[Driver Install] Launch silent NVIDIA driver installation\033[0m"
    # Only keep valid parameters supported by 580.159.03
    ./"${NVIDIA_RUN_FILE}" \
        -no-x-check \
        -no-nouveau-check \
        -no-dkms \
        -kernel-source-path /usr/src/linux-headers-${KERN_VER} \
        -silent

    echo -e "\033[32m✅ NVIDIA ${DRV_VERSION} proprietary driver installed successfully\033[0m"
    # Re-verify driver version after install
    if ! check_nvidia_driver_installed; then
        echo -e "\033[31mDriver version verification failed after installation, please troubleshoot manually!\033[0m"
        exit 1
    fi
}

# ===================== Pre-check: Verify & install matched driver =====================
echo -e "\033[34m[Pre-check] Validate installed NVIDIA driver version\033[0m"
if ! check_nvidia_driver_installed; then
    install_nvidia_run_driver
fi

# Check source folder exists
if [ ! -d "${SRC_DIR}" ]; then
    echo -e "\033[31mERROR: Source directory ${SRC_DIR} missing, extract open-gpu source archive first!\033[0m"
    exit 1
fi

# Step 1: Fallback install build dependencies
echo -e "\033[34m[1/5] Fallback installation of build dependencies\033[0m"
apt update
apt install -y build-essential linux-headers-${KERN_VER} libelf-dev pkg-config

# Step 2: Clean & compile open-gpu kernel modules
echo -e "\033[34m[2/5] Compile open-gpu kernel modules\033[0m"
cd "${SRC_DIR}"
make clean
make -j$(nproc) KERNEL_UNAME=${KERN_VER}
cd ../

# Verify compiled object exists
if [ ! -f "${KO_PATH}/nvidia.ko" ]; then
    echo -e "\033[31mBuild failed: ${KO_PATH}/nvidia.ko not generated\033[0m"
    exit 1
fi

# ===================== Pre-unload preparation: Stop GUI & NVIDIA services =====================
echo -e "\033[34m[3/5] Switch to text target, stop GPU related services & processes\033[0m"
clean_nvidia_modules

# Step 3: Confirm no residual NVIDIA kernel modules
echo -e "\033[34m[4/5] Confirm no residual NVIDIA kernel modules\033[0m"

# Backup & replace firmware
mkdir -p "${FW_DIR}"
mkdir -p "/usr/lib/firmware/nvidia/${DRV_VERSION}"
if [ ! -f "${FW_DIR}/gsp_tu10x.bin" ]; then
    echo "Backup original GSP firmware to ${FW_DIR}"
    cp -f "/usr/lib/firmware/nvidia/${DRV_VERSION}/gsp_tu10x.bin" "${FW_DIR}/gsp_tu10x.bin"
fi
cp "${FW_DIR}/${FW_TYPE}" /usr/lib/firmware/nvidia/${DRV_VERSION}/gsp_tu10x.bin -f
echo -e "\033[32m✅ Firmware replaced with ${FW_TYPE}\033[0m"

# Step 4: Direct insmod load compiled KO from source directory
echo -e "\033[34m[5/5] Direct insmod load compiled KO files inside source folder\033[0m"
insmod "${KO_PATH}/nvidia.ko"
insmod "${KO_PATH}/nvidia-uvm.ko"
insmod "${KO_PATH}/nvidia-modeset.ko"
insmod "${KO_PATH}/nvidia-drm.ko" modeset=1

# Post-load verification
echo -e "\033[32m==================== Post-load Verification ====================\033[0m"
echo "Currently loaded nvidia kernel modules:"
lsmod | grep nvidia
echo -e "\nActual loaded path of nvidia.ko:"
modinfo nvidia | grep filename

echo -e "\033[33mNote: Temporary effect only. Original driver restores after reboot, no permanent system file modification.\033[0m"
echo "GPU validation command: nvidia-smi"
echo "Restore graphical desktop: systemctl isolate graphical.target"

sleep 3
nvidia-smi