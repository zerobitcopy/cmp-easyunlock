DRV_VERSION="580.159.03"
FW_DIR="./firmware"
FW_TYPE="gsp_tu10x.bin"


systemctl isolate multi-user.target

#systemctl stop nvidia-persistenced

systemctl stop nvidia-dcgm

pkill -9 nv-hostengine
pkill -9 nv-hosten

#lsof /dev/nvidia*

rmmod nvidia_uvm
rmmod nvidia_drm
rmmod nvidia_modeset
rmmod nvidia


cp ${FW_DIR}/${FW_TYPE} /usr/lib/firmware/nvidia/${DRV_VERSION}/gsp_tu10x.bin -f

modprobe nvidia
modprobe nvidia_modeset
modprobe nvidia_drm
modprobe nvidia_uvm

echo "wait 5s and reboot"
sleep 5
reboot

