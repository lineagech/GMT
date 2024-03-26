#!/bin/bash

sudo cp ./tmp/*.ko /tmp/
sudo insmod /tmp/libnvm.ko
sudo insmod /tmp/libnvm_vmalloc.ko

#sudo umount /export/home3
echo -n "0000:dc:00.0" | sudo tee /sys/bus/pci/drivers/nvme/unbind
sleep 2
echo -n "0000:dc:00.0" | sudo tee /sys/bus/pci/drivers/libnvm\ helper/bind
sleep 2
sudo chmod 777 /dev/libnvm0


echo -n "0000:64:00.0" | sudo tee /sys/bus/pci/drivers/nvme/unbind
sleep 2
echo -n "0000:64:00.0" | sudo tee /sys/bus/pci/drivers/libnvm_vmalloc/bind
#echo -n "0000:64:00.0" | sudo tee /sys/bus/pci/drivers/libnvm/bind
sleep 2
sudo chmod 777 /dev/libnvm_vmalloc0
#

echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo always  | sudo tee /sys/kernel/mm/transparent_hugepage/defrag


