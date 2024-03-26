#ifndef __NVM_PERF_H__
#define __NVM_PERF_H__

#include "nvm_types.h"
//#define __NVM_PERF__
//#define thread_id() (blockDim.x * blockIdx.x + threadIdx.x)
#define __thread_id() 0

// CHIA-HAO
typedef struct
{
    uint64_t kernel_start;
    uint64_t kernel_end;
    uint64_t seq_read_start;
    uint64_t seq_read_end;
    uint64_t coalesce_page_start;
    uint64_t coalesce_page_end;
    uint64_t acquire_page_start;
    uint64_t acquire_page_end;
    clock_t read_data_start;
    clock_t read_data_end;
    
    uint64_t first_stage_start;
    uint64_t first_stage_end;
    uint64_t second_stage_start;
    uint64_t second_stage_end;

} nvm_perf_t;

// CHIA-HAO
typedef struct __align__(64) {
    uint64_t start_time;
    uint64_t submit_time;
    uint64_t complete_time;
    uint64_t sq_enqueue_second_time;
    uint64_t kernel_start_time;
    uint64_t kernel_end_time;
} nvme_record_t;





__device__ nvm_perf_t *perf_time_record = NULL;

#endif
