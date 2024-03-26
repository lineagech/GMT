#ifndef __NVM_INTERNAL_LINUX_IOCTL_H__
#define __NVM_INTERNAL_LINUX_IOCTL_H__
#ifdef __linux__

#include <linux/types.h>
#include <asm/ioctl.h>

#define NVM_IOCTL_TYPE          0x80



/* Memory map request */
struct nvm_ioctl_map
{
    uint64_t    vaddr_start;
    size_t      n_pages;
    uint64_t*   ioaddrs;
};



/* Supported operations */
enum nvm_ioctl_type
{
    NVM_MAP_HOST_MEMORY         = _IOW(NVM_IOCTL_TYPE, 1, struct nvm_ioctl_map),
    //NVM_MAP_HOST_MEMORY         = 12345678,
#ifdef _CUDA
    NVM_MAP_DEVICE_MEMORY       = _IOW(NVM_IOCTL_TYPE, 2, struct nvm_ioctl_map),
    //NVM_MAP_DEVICE_MEMORY       = 12345679,
#endif
    NVM_UNMAP_MEMORY            = _IOW(NVM_IOCTL_TYPE, 3, uint64_t),
    //NVM_UNMAP_MEMORY            = 12345680,
    NVM_MAP_KERNEL_MEMORY       = _IOW(NVM_IOCTL_TYPE, 4, uint64_t)
    //NVM_MAP_KERNEL_MEMORY       = 12345681
};


#endif /* __linux__ */
#endif /* __NVM_INTERNAL_LINUX_IOCTL_H__ */
