
ifneq ($(KERNELRELEASE),)
	src := /home/csl/cuc1057/BaM/tmm-bam/module
	obj-m := libnvm.o
	libnvm-objs := pci.o list.o ctrl.o map.o
	ccflags-y += -D_CUDA -I/usr/src/nvidia-515.65.01/nvidia -I/home/csl/cuc1057/BaM/tmm-bam/src -I/home/csl/cuc1057/BaM/tmm-bam/src/linux
	KBUILD_EXTRA_SYMBOLS := /usr/src/nvidia-515.65.01/Module.symvers
else

.PHONY: default reload unload load clean install

default:
	$(MAKE) -C /lib/modules/5.4.0-146-generic/build M=/home/csl/cuc1057/BaM/tmm-bam/module modules

clean:
	$(MAKE) -C /lib/modules/5.4.0-146-generic/build M=/home/csl/cuc1057/BaM/tmm-bam/module clean

reload: unload load

unload:
	-rmmod libnvm.ko

load:
	insmod libnvm.ko max_num_ctrls=64

install: default
	$(MAKE) -C /lib/modules/5.4.0-146-generic/build M=/home/csl/cuc1057/BaM/tmm-bam/module modules_install
	#$(MAKE) INSTALL_MOD_DIR=libnvm -C /lib/modules/5.4.0-146-generic/build M=/home/csl/cuc1057/BaM/tmm-bam/module modules_install

endif

