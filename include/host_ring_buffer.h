#ifndef __HOST_QUEUE_H__
#define __HOST_QUEUE_H__

#include "nvm_types.h"
#include "buffer.h"
#include "host_cache.h"
#include <simt/atomic>

#define bytes_mask_32 (0xffffffff)
//#define SAMPLE_FREQ (256) // pr
//#define SAMPLE_FREQ (64) // orig
//#define SAMPLE_FREQ (32)
//#define SAMPLE_FREQ (16)
//#define SAMPLE_FREQ (4) // bfs, sssp

#define SAMPLE_FREQ (4) // bfs, sssp
#define MID_PREDICTOR_THRESHOLD (10240)
//#define MID_PREDICTOR_THRESHOLD (2048)

#define DYNAMIC_SAMPLE_FREQ 0
#if DYNAMIC_SAMPLE_FREQ
volatile uint32_t invalid_slope_num = 0;
__host__ volatile uint32_t* dynamic_sample_freq_h = NULL;
__device__ volatile uint32_t* dynamic_sample_freq = NULL;
#endif

#define CLOCK_FREQ_GPU (1410*1024*1024)
__device__ volatile long long int clock_begin;
__device__ volatile long long int clock_now;

using callback_t = void (*) (uint64_t);

typedef struct __align__(32) {
    simt::atomic<uint64_t, simt::thread_scope_device> val;
} __attribute__((aligned(32))) padded_struct_64;

typedef struct __align__(32) {
    volatile void*      buffer;
    volatile bool*      ready;
    uint32_t            q_size;
    uint32_t            q_size_log2;
    uint64_t            curr_round_id;
    volatile uint64_t*           head;
    padded_struct_64    tail;
    padded_struct_64    in_ticket;
    padded_struct_64    access_count;
    padded_struct_64    post_count;
    padded_struct_64*   tickets;
} host_ring_buffer_t;

BufferPtr ptr, ptr2;

__forceinline__ __device__ 
unsigned long long int get_elapsed_seconds() 
{
    return (clock64() - clock_begin) / CLOCK_FREQ_GPU;
}

__global__ 
void get_clock_begin() 
{
    clock_begin = clock64();
}

inline __host__
host_ring_buffer_t* init_ring_buf_for_dev(uint32_t q_size, host_ring_buffer_t* host_buf)
{
    host_ring_buffer_t* ret = NULL;

    host_buf->q_size = q_size;
    host_buf->q_size_log2 = std::log2(q_size);
    ptr = createBuffer(q_size*sizeof(padded_struct_64), 0);
    cuda_err_chk(cudaMemset((void*)ptr.get(), 0, q_size*sizeof(padded_struct_64)));
    host_buf->tickets = (padded_struct_64*)ptr.get();

    ptr2 = createBuffer(q_size*16);
    //host_buf->buffer = (volatile void*)ptr2.get();
    cudaMalloc((void**)(&(host_buf->buffer)), q_size*16);
    host_buf->curr_round_id = 0;
    
    cuda_err_chk(cudaHostAlloc((void**)&(host_buf->ready), sizeof(bool), cudaHostAllocDefault));
    //cuda_err_chk(cudaMemset(host_buf->ready, 0, sizeof(bool)));
    memset((void*)host_buf->ready, 0, sizeof(bool));
    //printf("buf ready flag here is %u (%p)\n", *(host_buf->ready), host_buf->ready);
    //cuda_err_chk(cudaMallocHost((void**)&(host_buf->head), sizeof(uint64_t)));
    //host_buf->head = 0;
    cuda_err_chk(cudaMalloc((void**)&(host_buf->head), sizeof(uint64_t)));
    cudaMemset((void*)host_buf->head, 0, sizeof(uint64_t));


    cuda_err_chk(cudaMalloc(&ret, sizeof(host_ring_buffer_t)));
    cuda_err_chk(cudaMemset(ret, 0, sizeof(host_ring_buffer_t)));
    cuda_err_chk(cudaMemcpy(ret, host_buf, sizeof(host_ring_buffer_t), cudaMemcpyHostToDevice));

    printf("created host ring buffer (size %u, tickets %p, dev ptr %p)\n", host_buf->q_size, host_buf->tickets, ret);
    
    get_clock_begin<<<1,1>>>();

    return ret;
}

__forceinline__ __device__
void enqueue_mem_samples(host_ring_buffer_t* buf, uint64_t key, uint64_t virt_timestamp_diff, callback_t callback_func, uint64_t* ring_count)
{
    uint32_t ns = 8;
    //ulonglong4 sample = make_ulonglong4(key&bytes_mask_32, (key>>32)&bytes_mask_32, virt_timestamp_diff&bytes_mask_32, (virt_timestamp_diff>>32)&bytes_mask_32);
    //ulonglong2 sample = make_ulonglong2(key, virt_timestamp_diff);
    //printf("virt time diff %lu\n", virt_timestamp_diff);
    ulonglong2 sample = make_ulonglong2(virt_timestamp_diff, key);
    uint64_t ticket = buf->in_ticket.val.fetch_add(1, simt::memory_order_relaxed);
    uint32_t pos = ticket & (buf->q_size-1);
    uint64_t round_id = ticket >> buf->q_size_log2;
    
    //for (int i = 0; i < 1024; i++) printf("%d: %lu\n", i, buf->tickets[i].val.load(simt::memory_order_relaxed));
    //printf("queue size %u (%p)\n", buf->q_size, buf->tickets);
    //printf("pos %u, id %lu, round_id %lu\n", pos, buf->tickets[pos].val.load(simt::memory_order_relaxed), round_id);
    while (buf->tickets[pos].val.load(simt::memory_order_relaxed) != round_id) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        #if defined(__CUDA_ARCH__)
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
        #endif
#endif
    }

    ns = 8;
    while (buf->tickets[pos].val.load(simt::memory_order_acquire) != round_id) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        #if defined(__CUDA_ARCH__)
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
        #endif
#endif
    }

    // TODO: check if the loc which is going to be accessed is free now
    //while ( (ticket - *(buf->head)) >= buf->q_size );
    
    ulonglong2* buf_mem_loc = (ulonglong2*)(buf->buffer);
    buf_mem_loc[pos] = sample;
    //uint64_t* buf_mem_loc = (uint64_t*)(buf->buffer);
    //buf_mem_loc[2*pos] = virt_timestamp_diff;
    //buf_mem_loc[2*pos+1] = key;

    __threadfence();

    // TODO: notify host of fetching newest samples
    //if (ticket && ticket % SAMPLE_FREQ == 0) {
    //    (*callback_func)(ticket);
    //}

    uint64_t _count = buf->access_count.val.fetch_add(1, simt::memory_order_relaxed);
    if ((_count & (buf->q_size-1)) == buf->q_size-1) {
        while (buf->post_count.val.load() <= 0 || buf->post_count.val.load() < _count);
        (*buf->ready) = true;
        __threadfence_system();
        (*ring_count)++;
        // 072323
        __threadfence();
    }
    buf->post_count.val.fetch_add(1, simt::memory_order_relaxed);
    //printf("%u\n", count);

    buf->tickets[pos].val.fetch_add(1, simt::memory_order_acq_rel);
}

__forceinline__ __device__
void enqueue_mem_samples_blocked(host_ring_buffer_t* buf, uint64_t key, uint64_t virt_timestamp_diff, callback_t callback_func, uint64_t* ring_count)
{
    uint32_t ns = 8;
    ulonglong2 sample = make_ulonglong2(virt_timestamp_diff, key);
    uint64_t ticket = buf->in_ticket.val.fetch_add(1, simt::memory_order_relaxed);
    uint32_t pos = ticket & (buf->q_size-1);
    uint64_t round_id = ticket >> buf->q_size_log2;
    
    while (buf->tickets[pos].val.load(simt::memory_order_relaxed) != round_id) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        #if defined(__CUDA_ARCH__)
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
        #endif
#endif
    }

    ns = 8;
    while (buf->tickets[pos].val.load(simt::memory_order_acquire) != round_id) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        #if defined(__CUDA_ARCH__)
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
        #endif
#endif
    }

    // TODO: check if the loc which is going to be accessed is free now
    while ( (ticket - *(buf->head)) >= buf->q_size );
    
    ulonglong2* buf_mem_loc = (ulonglong2*)(buf->buffer);
    buf_mem_loc[pos] = sample;
    __threadfence();

    uint64_t _count = buf->access_count.val.fetch_add(1, simt::memory_order_relaxed);
    if ((_count & (buf->q_size-1)) == buf->q_size-1) {
        (*buf->ready) = true;
        __threadfence_system();
        (*ring_count)++;
        while (*(buf->ready)); // wait until ready flag becomes false
        atomicAdd((unsigned long long*)buf->head, (unsigned long long)buf->q_size);
        __threadfence();
    }

    buf->tickets[pos].val.fetch_add(1, simt::memory_order_acq_rel);
}


inline __host__
void dequeue_mem_samples()
{

}



#endif
