#ifndef __HOST_CACHE_H__
#define __HOST_CACHE_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
//#include "remote_ipc.h"
//#include "constants.h"
//#include "gds.h"

#include "buffer.h"
#include "queue.h"
#include "host_ring_buffer.h"
#include "linear_regression.h"
#include <thread>
#include <vector>
#include <map>
#include <fstream>
#include <unordered_map>
#include <simt/atomic>
#include <mutex>
#include <fcntl.h>
#include <sstream>
#include <chrono>
#include <deque>
#include <time.h>
#include <unistd.h>
#include <curand_kernel.h>

/* Utility */
#define DIV_ROUND_UP(n,d) (((n) + (d) - 1) / (d))
#define UNUSED(x) (void)(x)
#define CUDA_SAFE_CALL(x) \
    if ((x) != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR %s: %d %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError())); exit(-1); \
    }
#define GET_ARR_INDEX(val, size) (val/size)
#define WAIT_ON_MEM(mem, val)  while(read_without_cache(&mem) != val);
#define WAIT_ON_MEM_NE(mem, val)  while(read_without_cache(&mem) == val);
#define WAIT_COND(x, val, ns) \
    while(x != val) {\
        __nanosleep(ns);\
        if (ns < 256) ns *= 2;\
    }\

#define TRY_LOCK(lock) if (atomicExch((int*)&lock, 1) == 0)
#define MUTEX_LOCK(lock) while (atomicExch((int*)(&lock), 1))
#define MUTEX_UNLOCK(lock) atomicExch((int*)(&lock), 0)
#define MUTEX_IS_LOCKED(lock)

#define BEGIN_SINGLE_TRHEAD __syncthreads(); if ((threadIdx.x + threadIdx.y + threadIdx.z) == 0) {
#define END_SINGLE_THREAD } __syncthreads();

//#define TID (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)
#define LANE_ID (TID & 0x1f)
#define WARP_ID (TID >> 5)
#define NUM_WARPS ((blockDim.x * blockDim.y * blockDim.z) >> 5)
#define BLOCK_ID (blockIdx.x + gridDim.x * blockIdx.y + blockIdx.z * gridDim.x * gridDim.y)
#define GET_SMID(SMID) asm volatile ("mov.u32 %0, %%smid;" : "=r"(SMID) :);
#define LOW_MASK(b) ((1<<b)-1)
#define HIGH_MASK(b) ~LOW_MASK(b)
#define GET_TIME(timer) \
    asm volatile ("mov.u64 %0, %%globaltimer;" : "=l"(timer) : )


#define PROFILE 0
#define APPLY_PROFILE 0
#define REPLICATION 0
#define APPLY_BURST_PROFILE 0
#define NAIVE_EVICT_TO_HC 0
#define USE_PINNED_MEM 0
#define LOGGING_RT_INFO 0
// Enable this when running vectorAdd
#define CUSTOMIZED_FOR_VECTORADD 0
#define CUSTOMIZED_FOR_BFS 0
#define CUSTOMIZED_FOR_PR_COALESCE_PC 0
#define PRINT_TRAINING_PAIRS 0
#define BYPASS_INVALID_SAMPLES 1
#define BYPASS_UNREASONABLE_SAMPLES 1
#define SMART_DOWN_SCALING 1
#define ENABLE_COMPENSATION 1
//#define COMP_FACTOR 20
#define COMP_FACTOR 100

// PERF PROFILING FOR DATA "FAULT HANDLING"
#define TIME_BREAK_DOWN_FOR_PERF_TEST 0
#define NUM_CPU_CORES 16
#define NUM_FLUSH_THREADS 16
#define PRE_LAUNCH_THREADS 1
#define GET_RW_QUEUE_ENTRY_ALTER (PRE_LAUNCH_THREADS)
#define BASIC_TEST 1

//
#define MULTI_DATA_STREAM 0
#define NUM_DATA_STREAMS 32
#define USER_SPACE_ZERO_COPY 0
#define GET_GOLDEN_REUSE_DISTANCE 0
#define USE_RAND_POLICY 0

// DEBUG
//#define DEBUG 1
#if 0
#include <assert.h>
#define GPU_ASSERT(x) assert(x);
#define HOST_ASSERT(x) assert(x);
#define GPU_PRINT(...) printf(__VA_ARGS__);
#define CPU_PRINT(...) printf(__VA_ARGS__);

#define START(start) \
do{ \
    if (TID == 0) GET_TIME(start); \
} while(0)

#define END(t, start, end) \
do {\
    if (TID == 0) { \
        GET_TIME(end);\
        atomicAdd(&t, end-start);\
    }\
} while(0)

#else
#define GPU_ASSERT(x)
#define HOST_ASSERT(x)
#define GPU_PRINT(...)
//#define GPU_PRINT(...) printf(__VA_ARGS__);
#define CPU_PRINT(...) 
#define START(start)
#define END(t, start, end)
#endif

/* Data Structures Constants */
#define GPU_OPEN_FILE_TABLE_SIZE 128
#define GPU_RW_SIZE 8192
#define FILENAME_SIZE 128

/* Memory Pool Constants */
#define MEM_POOL_SIZE ((uint64_t)1024*(uint64_t)1024*(uint64_t)1024*32)
#define PAGE_MAP_SIZE (MEM_POOL_SIZE >> 20)

/* GDS PTR related */
#define PAGE_ARR_SIZE 1024
#define TO_DEVICE 0x1
#define FROM_DEVICE 0x2

#define PAGE_SIZE_4K ((size_t)1024*(size_t)4)
#define PAGE_SIZE_64K ((size_t)1024*(size_t)64)
#define PAGE_SIZE_2M ((size_t)1024*(size_t)1024*(size_t)2)
#define PAGE_SIZE PAGE_SIZE_64K
//#define PAGE_SIZE PAGE_SIZE_4K

#define GPU_MEM_SIZE_PAGE_4K ((size_t)1024*(size_t)1024*(size_t)1024*(size_t)2)
#define GPU_MEM_SIZE_PAGE_2M ((size_t)1024*(size_t)1024*(size_t)1024*(size_t)2)
#define GPU_MEM_SIZE_TEST ((size_t)1024*(size_t)1024*(size_t)1024*(size_t)2)

//#define HOST_MEM_SIZE (65536*8192) // for test
#ifdef GRAPH_WORKLOADS
#warning "Graph Workloads...!"
#define HOST_MEM_SIZE ((size_t)1024UL*(size_t)1024*(size_t)1024*(size_t)16)
#else
#define HOST_MEM_SIZE ((size_t)1024UL*(size_t)1024*(size_t)1024*(size_t)1)
#endif
#define HOST_MEM_NUM_PAGES (HOST_MEM_SIZE / PAGE_SIZE)

#define SECOND_TIER_SIZE (HOST_MEM_NUM_PAGES)
#define FIRST_TIER_SIZE (HOST_MEM_NUM_PAGES/4)
#define SHOULD_CACHED_BY_HOST(reuse_dist) (reuse_dist >= FIRST_TIER_SIZE && reuse_dist < SECOND_TIER_SIZE)

#define MEM_SAMPLES_RING_BUFFER_SIZE (1024*8)
/* Cuda Stream related */
//#define DATA_STREAM_NUM GPU_RW_SIZE

/* Threshold */
#define GDS_SIZE_THRESHOLD (512*1024)

#define INIT_SHARED_MEM_PTR(T, host_ptr, device_symbol) \
{\
    void* mapped_dev_ptr;\
    CUDA_SAFE_CALL(cudaHostAlloc((void**)(&host_ptr), sizeof(T), cudaHostAllocMapped));\
    CUDA_SAFE_CALL(cudaHostGetDevicePointer((void**)(&mapped_dev_ptr), (void*)host_ptr, 0));\
    assert(mapped_dev_ptr != NULL);\
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(device_symbol, &mapped_dev_ptr, sizeof(void*)));\
}\

#define INIT_NESTED_SHARED_MEM_PTR(T, host_ptr, E, e, num, device_symbol) \
{\
    void* mapped_dev_ptr;\
    CUDA_SAFE_CALL(cudaHostAlloc((void**)(&host_ptr), sizeof(T), cudaHostAllocMapped));\
    for (int i = 0; i < num; i++) { \
        CUDA_SAFE_CALL(cudaHostAlloc((void**)(&host_ptr->e[i]), sizeof(E), cudaHostAllocDefault));\
        /*CUDA_SAFE_CALL(cudaMallocManaged((void**)(&host_ptr->e[i]), sizeof(E)));*/\
        /*CUDA_SAFE_CALL(cudaMemAdvise((const void*)host_ptr->e[i], sizeof(E), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));*/\
    }\
    CUDA_SAFE_CALL(cudaHostGetDevicePointer((void**)(&mapped_dev_ptr), (void*)host_ptr, 0));\
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(device_symbol, &mapped_dev_ptr, sizeof(void*)));\
}\

#define INIT_DEVICE_MEM(T, dev_ptr, device_symbol)\
{\
    CUDA_SAFE_CALL(cudaMalloc((void**)(&dev_ptr), sizeof(T)));\
    CUDA_SAFE_CALL(cudaMemset((void*)dev_ptr), 0, sizeof(T));\
    CUDA_SAFE_CALL(cudaMemcpy(cudaMemcpyToSymbol(device_symbol, &device_ptr, sizeof(void*))));\
}\

#define INIT_DEVICE_MEM_ARRARY(T, device_symbol, host_ptr, len)\
{\
    void* dev_ptr = NULL;\
    CUDA_SAFE_CALL(cudaMalloc((void**)(&dev_ptr), sizeof(T)*len));\
    assert(dev_ptr != NULL);\
    CUDA_SAFE_CALL(cudaMemset((void*)dev_ptr, 0, sizeof(T)*len));\
    CUDA_SAFE_CALL(cudaMemcpy((void*)dev_ptr, (void*)host_ptr, sizeof(T)*len, cudaMemcpyHostToDevice));\
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(device_symbol, &dev_ptr, sizeof(void*)));\
}\



#define FREE_SHARED_MEM(host_ptr) CUDA_SAFE_CALL(cudaFreeHost((void*)host_ptr))

#define handle_error_en(en, msg) \
    do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0);

enum {
    ENTRY_AVAILABLE = 0,
    ENTRY_OCCUPIED
};

static const char* enum_entry_status[] = {"ENTRY AVAILABLE", "ENTRY OCCUPIED"};

enum {
    GPU_RW_READ = 0,
    GPU_RW_WRITE = 1,
    GPU_RW_EVICT_TO_HOST = 2,
    GPU_RW_EVICT_TO_HOST_PERF_TEST = 3,
    GPU_RW_EVICT_TO_HOST_ASYNC = 4,
    GPU_RW_FETCH_FROM_HOST = 5,
    GPU_RW_FETCH_TO_HOST_BY_HOST = 6,
    GPU_RW_FETCH_FROM_HOST_PERF_TEST = 7,
    GPU_RW_PIN_PAGE = 8,
    GPU_RW_UNPIN_PAGE = 9,
    GPU_RW_LOGGING = 10,
    GPU_RW_ARBITRARY_LOGGING = 11,
    GPU_RW_GET_REQS_NUM = 12,
    GPU_RW_SAMPLE_MEM = 13,
    GPU_RW_SAMPLE_MEM_EVICTION,
    GPU_RW_SAMPLE_MEM_RE_REF,
    GPU_RW_ALLOCATE_PAGE,
    GPU_RW_TYPE_SIZE,
};

enum {
    PAGE_ACTION_FETCH = 0,
    PAGE_ACTION_EVICT = 1,
    PAGE_ACTION_ACCESS = 2,
    PAGE_ACTION_SIZE
};

__device__ static const char* enum_rw_req[GPU_RW_TYPE_SIZE] = {
    "Read\0", 
    "Write\0", 
    "Evict to Host\0", 
    "Evict to Host Perf Test", 
    "Evict to Host Async\0", 
    "Fetch from Host\0", 
    "Fetch to Host By Host\0", 
    "Fetch from Host Perf Test", 
    "Pin Page\0", 
    "Unpin Page\0", 
    "Logging\0", 
    "Arbitrary Logging",
    "Get Reqs Num",
    "Sample Mem Traces",
    "Sample Mem Eviction Event",
    "Sample Mem Re-reference Event",
    "Allocate Empty Page",
};
__device__ static const char* enum_page_action[GPU_RW_TYPE_SIZE] = {"Fetch\0", "Evict\0", "Access\0", "Logging"};
static const char* enum_rw_req_h[GPU_RW_TYPE_SIZE] = {
    "Read\0", 
    "Write\0", 
    "Evict to Host\0", 
    "Evict to Host Perf Test", 
    "Evict to Host Async\0", 
    "Fetch from Host\0", 
    "Fetch to Host By Host\0", 
    "Fetch from Host Perf Test\0", 
    "Pin Page\0", 
    "Unpin Page\0", 
    "Logging\0", 
    "Arbitrary Logging\0",
    "Get Reqs Num",
    "Sample Mem Traces",
    "Sample Mem Eviction Event",
    "Sample Mem Re-reference Event",
    "Allocate Empty Page",
};
static const char* enum_page_action_h[GPU_RW_TYPE_SIZE] = {"Fetch\0", "Evict\0", "Access\0"};

enum {
    GPU_RW_EMPTY = 0,
    GPU_RW_PENDING,
    GPU_RW_PROCESSING,
    GPU_RW_WAIT_FLUSH,
    GPU_RW_ASYNC_PROCESSING,
    GPU_RW_READY
};

static const char* enum_rw_status[] = {"RW EMPTY", "RW PENDING", "RW PROCESSING", "RW WAIT FLUSH", "RW ASYNC PROCESSING", "RW READY"};

enum {
    FIRST_EVICTION_CLEAN = 0,
    FIRST_EVICTION_DIRTY,
    LAST_ITERATION, 
    NOT_LAST_AND_DIRTY,
    NO_HISTORY_USE_LAST_PRED,
    WRONG_PREDICTION_SO_TO_TIER2,
    SECOND_EVICTION_FOLLOW_LAST_PRED,
    TRANSITION_PRED,
    TIER_PRED_REASONS_SIZE,
};
__device__ static const char* tier_prediction_reasons_str[TIER_PRED_REASONS_SIZE] = {
    "first eviction (clean)", 
    "first eviction (dirty)", 
    "last iteration", 
    "not last iteration and is dirty", 
    "no history to use so use last time prediction", 
    "wrong prediction so evict to tier-2",
    "second eviction so follow last time prediction",
    "normal transition prediction"
};

enum {
    GPU_NO_LOCKED = 0,
    GPU_LOCKED
};

enum {
    HOST_CACHE_ENTRY_INVALID = 0,
    HOST_CACHE_ENTRY_VALID = 1,
};

enum {
    HOST_CACHE_ENTRY_UNLOCKED = 0,
    HOST_CACHE_ENTRY_LOCKED,
};

enum {
    HOST_CACHE_SUCCEEDED = 0,
    HOST_CACHE_FETCH_FAILED,
    HOST_CACHE_EVICT_SUCCEEDED,
    HOST_CACHE_EVICT_FAILED,
    HOST_CACHE_PIN_FAILED,
    HOST_CACHE_ALLOCATE_FAILED,
    HOST_CACHE_RET_CODE_NUM
};

enum {
    ZERO_COPY = 1,
    CUDA_MEMCPY_ASYNC = 2,
    READ_FROM_SSD = 3,
};

static const char* enum_lock_status[] = {"NOT LOCKED", "LOCKED"};
__device__ static const char* enum_hc_ret_code_str[] = {"Succ", "Fetch Failed", "Evict Succ", "Evict Failed", "Pin Failed"};


class ReuseDistCalculator;
class MemSampleCollector;

class GpuOpenFileTableEntry {
public:
    volatile int status;
    volatile char filename[FILENAME_SIZE];
    volatile int cpu_fd;
    volatile size_t size;
    volatile bool is_dirty;
        
    /* Read/Write Handler */
    //GPU_RW_HANDLER rw_handler;
    
    /* Device-side */
    __device__ int open(const char* filename) volatile;
    __device__ int close() volatile;
    
    /* Host-side */
    __host__ void init() volatile;  
};

//class GpuOpenFileTable {
class GpuOpenFileTable {
public:
    volatile GpuOpenFileTableEntry entries[GPU_OPEN_FILE_TABLE_SIZE];
    __host__ void init() volatile;
};

class __align__(64) LocalEntry {
public:
    volatile uint32_t type;
    volatile uint32_t status;
    #if USE_PINNED_MEM
    volatile bool pin_on_host;
    #endif
    volatile bool dirty;
    volatile uint64_t key;
    volatile void* volatile gpu_addr;
    volatile uint64_t bid;
};

class __align__(64) GpuRwQueueEntry {
public:
    /* Reqeusts from GPU, filling the following 5 fields */
    volatile uint32_t type;
    volatile uint32_t status;
    #if USE_PINNED_MEM
    volatile bool pin_on_host;
    #endif
    volatile bool dirty;
    volatile uint64_t key;
    volatile void* volatile gpu_addr;
    /**/
    volatile uint64_t bid;
    union __align__(64) {
        struct CacheReq {
            volatile int num_reqs;
            volatile int return_code;
            volatile bool is_cached_page_dirty;
        } CacheReq;
        struct RwReq {
            volatile size_t size;
            volatile size_t buffer_offset;
            volatile size_t file_offset;
            /* Complete status written by CPU */
            volatile ssize_t return_size;
        } RwReq;
        struct LoggingReq {
            volatile uint64_t arrayIdx;
            volatile uint64_t pageIdx;
            volatile uint32_t actionType;
        } LoggingReq;
        struct ArbitraryLoggingReq {
            volatile uint64_t threadCount;
        } ArbitraryLoggingReq;
        struct ReplicationReq {
            volatile uint64_t tag;
        } ReplicationReq;
        struct SampleMemReq {
            volatile uint64_t virt_timestamp_dist;
        } SampleMemReq;
        struct SampleMemResponse {
            volatile uint64_t actual_reuse_dist;
        } SampleMemResponse;
        struct FetchResponse {
        } FetchResponse;
    } u;

    /* Called by GPU */
    __device__ ssize_t remote_read_write(size_t _size, size_t _buffer_offset, size_t _file_offset, int req_type, void* dev_data_ptr, int& bid, bool dirty, bool* is_cached_page_dirty);
    /* Called by Host */
    __host__ void init() volatile;
};

class GpuRwQueue {
public:
    GpuRwQueueEntry entries[GPU_RW_SIZE];
    void* valid_in_pc[GPU_RW_SIZE];
    __device__ GpuRwQueueEntry* getEntryByIndex(int entry_index);
    __host__ void init() volatile;
};

class GpuFileRwManager {
public:
    volatile int entry_lock[GPU_RW_SIZE];
    volatile int lock;
    volatile void* buffer_base;

    simt::atomic<uint32_t, simt::thread_scope_device> ticket;
    uint8_t pad6[28];

    /* Requests queue */
    GpuRwQueue* gpu_rw_queue;

    __device__ void* getBufferBase() volatile;
    __device__ int getRWQueueEntry() volatile;
    __device__ int getRWQueueEntry2() volatile;
    __device__ void putRWQueueEntry(int entry_index) volatile;
    
    /* Initializer called by host */
    __host__ void init(GpuRwQueue* _gpu_rw_queue) volatile;
};

/* Called by GPU threads, and return a pointer pointing where valid data resides in GPU memory*/
__device__ void* readDataFromHost(int _fd, size_t _size, off_t _offset);
__device__ bool read_data_from_host(int _fd, size_t _size, off_t _offset, void* gpu_addr);

/* Called by GPU threads, giving a pointer pointing where valid data resides in GPU memory */
__forceinline__ __device__ int writeDataToHost(int _fd, size_t _size, off_t _offset, void* data);
__forceinline__ __device__ int accessHostCache(void* data, int key, int req_type, int& bid, bool dirty, bool* is_cached_page_dirty = NULL);
__forceinline__ __device__ int sendReqToHostCache(uint64_t key, int bid, uint32_t req_type, bool* is_cached_page_dirty);
__forceinline__ __device__ int getReqsNumOnHostCache();
__forceinline__ __device__ int accessHostCacheAsync(void* data, int req_type, int& bid);
__forceinline__ __device__ void* get_host_cache_addr(int bid);


class StreamManager {
public:
    cudaStream_t kernel_stream;
    #if MULTI_DATA_STREAM
    cudaStream_t data_stream_to_device[NUM_DATA_STREAMS];
    cudaStream_t data_stream_from_device[NUM_DATA_STREAMS];
    #else
    cudaStream_t data_stream_to_device;
    cudaStream_t data_stream_from_device;
    #endif
    //cudaStream_t data_stream[DATA_STREAM_NUM];
    StreamManager() {
        CUDA_SAFE_CALL(cudaStreamCreate(&kernel_stream));
        #if MULTI_DATA_STREAM
        for (uint32_t i = 0; i < NUM_DATA_STREAMS; i++) {
            CUDA_SAFE_CALL(cudaStreamCreate(&(this->data_stream_to_device[i])));
            CUDA_SAFE_CALL(cudaStreamCreate(&(this->data_stream_from_device[i])));
        }
        #else
        CUDA_SAFE_CALL(cudaStreamCreate(&(this->data_stream_to_device)));
        CUDA_SAFE_CALL(cudaStreamCreate(&(this->data_stream_from_device)));
        #endif
        //for (int i = 0; i < DATA_STREAM_NUM; i++) {
        //    CUDA_SAFE_CALL(cudaStreamCreate(&(data_stream[i])));
        //}
        fprintf(stderr, "%s\n", __func__);
    }

    ~StreamManager() {
        CUDA_SAFE_CALL(cudaStreamDestroy(kernel_stream));
        #if MULTI_DATA_STREAM
        for (uint32_t i = 0; i < NUM_DATA_STREAMS; i++) {
            CUDA_SAFE_CALL(cudaStreamDestroy(data_stream_to_device[i]));
            CUDA_SAFE_CALL(cudaStreamDestroy(data_stream_from_device[i]));
        }
        #else
        CUDA_SAFE_CALL(cudaStreamDestroy(data_stream_to_device));
        CUDA_SAFE_CALL(cudaStreamDestroy(data_stream_from_device));
        #endif
        //for (int i = 0; i < DATA_STREAM_NUM; i++) {
        //    CUDA_SAFE_CALL(cudaStreamDestroy(data_stream[i]));
        //}
        printf("%s\n", __func__);
    }
};

#if USE_PINNED_MEM
#define IS_PINNED(s) (s.is_pinned)
#else
#define IS_PINNED(s) (s.is_pinned.load(simt::std::memory_order_acquire))
//#define IS_PINNED(s) (s.is_pinned.load(simt::std::memory_order_acquire) != 0)
#endif
class HostCacheState {
public: 
    simt::atomic<uint32_t> lock;
    simt::atomic<uint32_t> state;
    simt::atomic<bool> is_pinned;
    //simt::atomic<uint32_t> is_pinned;
    #if USE_PINNED_MEM
    #endif
    //bool is_pinned;
    volatile bool is_dirty;
    void* gpu_addr;
    uint64_t tag;

    HostCacheState() {
        this->gpu_addr = NULL;
        this->tag = 0;
        this->is_dirty = false;
    }
    HostCacheState (const HostCacheState& _state) {
        this->gpu_addr = _state.gpu_addr;
        this->tag = _state.tag;
        this->is_dirty = _state.is_dirty;
    }
    ~HostCacheState() {
        std::cerr << "HostCacheState: Freeing " << this << std::endl;
    }
};

class alignas(32) PageInfo_d {
public:
    uint64_t range; // how many pages here
    uint64_t* fetch_count; // the number of fetch for a given page
};

class alignas(32) PageProfileInfo {
public:
    PageInfo_d page_info;
    PageInfo_d* page_info_d;
};

#define REUSE_HISTORY_LEN (8)

enum {
    TierZero = 0,
    Tier1_GPU = 1,
    Tier2_CPU,
    Tier3_SSD,
    TiersNum,
} MemTier;

__device__ const char* mem_tier_str[TiersNum] = {"Tier Zero", "Tier-1 GPU", "Tier-2 CPU", "Tier-3 SSD"};

#define print_reuse_history(prt, curr_tier) \
    printf("curr-tier %u: tier-1 %u, tier-2 %u, tier-3 %u\n", curr_tier, \
        prt.reuse_history[curr_tier][1], prt.reuse_history[curr_tier][2], prt.reuse_history[curr_tier][3])

class alignas(32) PageReuseTable {
public:
    uint8_t reuse_history[4][REUSE_HISTORY_LEN];
    uint8_t reuse_history_index;
    uint8_t last_predicted_tier;
    uint8_t last_actual_tier;
    uint8_t thrashing_count;
    uint8_t evicted_before;
    uint8_t evict_attempt_num;
    uint8_t under_mem_pressure;
    //uint8_t tmp_count;
    uint64_t curr_virt_timestamp;
    uint64_t last_virt_timestamp_sampled;
    uint64_t last_eviction_virt_timestamp;
    uint64_t last_reference_virt_timestamp;
    uint64_t estimated_remaining_reuse_dist;
#if GET_GOLDEN_REUSE_DISTANCE
    uint8_t  curr_predicted_tier;
    uint8_t  tier_prediction_reason;
    uint64_t dist_from_ref_to_eviction;
    uint64_t actual_reuse_dist_upon_re_ref;
    uint64_t remaining_virt_timestamp_dist;
#endif
};

class alignas(32) LinearRegInfo {
public:
    float slope = 1.0;
    float offset = 0.0;
};

class alignas(32) TierBins {
public:
    //uint64_t bins[4] = { 0 };
    simt::atomic<uint64_t, simt::thread_scope_device> bins[4];
    uint64_t threshold = 0;
};

class PerfTestTimeProfile {
public:
    std::chrono::high_resolution_clock::time_point thread_launch;
    std::chrono::high_resolution_clock::time_point handle_begin;
    std::chrono::high_resolution_clock::time_point transfer_begin;
    std::chrono::high_resolution_clock::time_point transfer_end;
    std::chrono::high_resolution_clock::time_point handle_end;
};

class HostCacheRuntimeState {
public:
    uint32_t num_pending_reqs;
};

class HostCache {
public:
    HostCache(Controller* ctrl, uint64_t gpu_mem_size);
    HostCache();
    ~HostCache();
    //void setGDSHandler(GDS_HANDLER* _gds_handler);
    void mainLoop();
    void launchThreadLoop();
    void threadLoop(int);
    void threadLoop2(int);
    void handleRequest(int);
    void preLoadData(int fd, size_t page_size);
    void loggingOnHost(uint64_t arrayIdx, uint64_t pageIdx, uint32_t actionType);
    void flushLog();
    void flushReuseDistTable();
    void readProfiledLog();
    void buildProfileInfo();
    void releaseDevMem();
    void debugPrint();
    //void hostFlushPage(uint64_t page);
    void registerRangesLBA(uint64_t);

    volatile GpuOpenFileTable* host_open_table;
    volatile GpuFileRwManager* host_rw_manager;
    volatile GpuFileRwManager* host_rw_manager_mem;
    volatile HostCacheRuntimeState* hc_runtime_state;
    GpuRwQueue* host_rw_queue;
    GpuRwQueue* host_rw_queue_mem;
  
    //GDS_HANDLER* gds_handler;  
    //std::vector<std::thread*> gds_handler_threads;
    //std::unordered_map<int, GDS_HANDLER*> gds_handlers;
    void* host_buffer[GPU_RW_SIZE];
    void* host_mem;
    uint64_t* prp1;
    uint64_t* prp2;
    DmaPtr host_dma;
    DmaPtr prp2_dma;
    std::vector<HostCacheState> host_cache_state;
    simt::std::atomic<uint64_t> bid;
    std::atomic<uint64_t> fetch_from_host_count;
    std::atomic<uint64_t> valid_entry_count;

    cudaStream_t cu_stream;
    Controller* ctrl;

    #define NUM_NVME_QUEUES 1
    QueuePair* qp[NUM_NVME_QUEUES];
    // TODO: initialize following 
    nvm_queue_host_t sq_host[NUM_NVME_QUEUES];
    nvm_queue_host_t cq_host[NUM_NVME_QUEUES];

    simt::atomic<uint64_t, simt::thread_scope_device>* q_head;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_tail;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_lock;
    simt::atomic<uint64_t, simt::thread_scope_device>* extra_reads;
    
    uint64_t gpu_mem_size;
    uint32_t* linear_reg_info_idx_h;
    LinearRegInfo *linear_reg_info_h;
    TierBins* tier_bins_h;
    uint8_t* tier_eviction_queue_h;

    MemSampleCollector *mem_sample_collector;
    #if GET_GOLDEN_REUSE_DISTANCE 
    MemSampleCollector *golden_mem_collector;
    #endif

    // ranges info
    std::vector<uint64_t> ranges_starting_lba;

    // 
    class PageActInfo {
    public:
        uint64_t timestamp;
        uint64_t actionType;
        PageActInfo(uint64_t _t, uint64_t _a) : timestamp(_t), actionType(_a) {}
    };
    std::map<uint64_t, std::map<uint64_t, uint64_t>> accessCountMap;
    std::map<uint64_t, std::map<uint64_t, std::vector<PageActInfo>>> fetchEvictMap; // arrayIdx, pageIdx, seq of pageActInfo
    std::string appName;
 
    std::vector<PageProfileInfo> profile_info_vec;
    #if TIME_BREAK_DOWN_FOR_PERF_TEST
    std::vector<PerfTestTimeProfile> perf_test_time_profile_vec;
    #endif

    #if PRE_LAUNCH_THREADS
    std::atomic<uint32_t> global_qid;
    #endif
    std::atomic<int64_t> pending_reqs_num;

    uint64_t num_pages;
    std::atomic<uint64_t> total_accesses;
    std::atomic<uint64_t> total_fetches;
    std::atomic<uint64_t> total_clean_fetches;
    std::atomic<uint64_t> total_dirty_fetches;
    std::atomic<uint64_t> total_cu_fetches;
    std::atomic<uint64_t> total_zc_fetches;
    std::atomic<uint64_t> total_tier1_evicts;
    std::atomic<uint64_t> total_tier1_dirty_evicts;
    std::atomic<uint64_t> total_tier1_clean_evicts;
    std::atomic<uint64_t> total_hits;

private:
    uint64_t num_ctrl_pages_in_one_line;
};

class Shct{
public:
    enum {
        NEAR = 0,
        INTERMEDIATE,
        DISTANT,
    };
    uint32_t n_pages;
    uint32_t *shct[4];
    uint32_t *outcome[4];
    Shct(uint32_t _n_pages):n_pages(_n_pages) {
        for (uint32_t i = 0; i < 4; i++) {
            CUDA_SAFE_CALL(cudaMalloc(&(this->shct[i]), _n_pages*PAGE_SIZE));
            CUDA_SAFE_CALL(cudaMalloc(&(this->outcome[i]), _n_pages*PAGE_SIZE));
        }
    }

    void update(bool hit, uint32_t array, uint32_t page) {
        if (hit) {
            this->outcome[array][page] = 1;
            if (this->shct[array][page] < 8) this->shct[array][page] += 1;
        }
        else {
            if (this->outcome[array][page] != 1) {
                this->shct[array][page] -= 1;
            }
            this->outcome[array][page] = 0;
        }
    }

    uint32_t query(uint32_t array, uint32_t page) {
        if (this->outcome[array][page] == 1) {
            return NEAR;
        }
        else if (this->shct[array][page] == 0) {
            return DISTANT;
        }
        else {
            return INTERMEDIATE;
        }
    }
};


#define FIRST_REF (ULONG_MAX)
class ReuseDistCalculator {
public:
    ReuseDistCalculator() {
        this->time = 0;
        this->dist = 0;
        last_access.clear();
    }

    uint64_t update(uint64_t key) 
    {
        dist = ULONG_MAX;
        //assert(last_access_before_evict_time.find(key) == last_access_before_evict_time.end());
        if (last_access_pos.find(key) != last_access_pos.end()) {
            auto pos = last_access_pos[key];
            dist = std::distance(pos, last_access.end());
            //reuse_dist[dist] += 1;
            last_access.erase(pos);
        }
        //fprintf(stderr, "time %lu - key %lu\n", time, key);
        last_access[this->time] = key;
        last_access_pos[key] = prev(last_access.end());

        this->time++;
        return dist;
    }

    uint64_t get_dist_from_key_to_end(uint64_t key) 
    {
        //fprintf(stderr, "evict for %lu\n", key);
        if (last_access_pos.find(key) == last_access_pos.end()) {
            fprintf(stderr, "error: not found key %lu\n", key);
        }
        assert(last_access_pos.find(key) != last_access_pos.end());
        // last access position in lru stack just before eviction
        //last_access_before_evict_pos[key] = last_access_pos[key];
        last_access_before_evict_time[key] = last_access_pos[key]->first;
        
        auto pos = last_access_pos[key];
        dist = std::distance(pos, last_access.end());
        
        return dist;
    }

    uint64_t get_dist_upon_re_ref(uint64_t key) 
    {
        //assert(last_access_before_evict_pos.find(key) != last_access_before_evict_pos.end());
        //fprintf(stderr, "re-ref for %lu\n", key);
        if (last_access_before_evict_time.find(key) == last_access_before_evict_time.end()) {
            fprintf(stderr, "error-2: not found key %lu\n", key);
            //return (uint64_t)-1;
        }
        assert(last_access_before_evict_time.find(key) != last_access_before_evict_time.end());
        //fprintf(stderr, "last_access_before_evict %lu - latest time %lu\n", last_access_before_evict_time[key], this->time);
        auto pos = last_access.lower_bound(last_access_before_evict_time[key]);
        dist = std::distance(pos, last_access.end());

        last_access_before_evict_time.erase(key);
        return dist;
    }

    void test(int k);

private:
    std::map<uint64_t,uint64_t> last_access; // last access time, key
    std::unordered_map<uint64_t,std::map<uint64_t,uint64_t>::iterator> last_access_pos;
    std::unordered_map<uint64_t,std::map<uint64_t,uint64_t>::iterator> last_access_before_evict_pos;
    std::unordered_map<uint64_t,uint64_t> last_access_before_evict_time;
    
    uint64_t time;
    uint64_t dist;
};

__device__ volatile uint32_t* linear_reg_info_idx = 0;
__device__ volatile LinearRegInfo* linear_reg_info = NULL;

// 091223
__device__ volatile TierBins* tier_bins_d = NULL;
#define TIER_EVICTION_QUEUE_LENGTH (10000)
class TierEvictionQueue {
public:
    __device__ simt::atomic<uint64_t, simt::thread_scope_device> total_evictions;
    __device__ simt::atomic<uint64_t, simt::thread_scope_device> tier_eviction_queue_head;
    __device__ simt::atomic<uint64_t, simt::thread_scope_device> tier_eviction_queue_tail;
};
__device__ volatile uint8_t* tier_eviction_queue = NULL;
TierEvictionQueue* teq_h = NULL;
__device__ TierEvictionQueue* teq = NULL;;


#define GET_PAGE_FAULT_RATE 0
__device__ volatile int32_t page_fault_count = 0;
__device__ volatile uint64_t page_fault_clock = 0;
__device__ volatile int32_t concurrent_evict_count = 0;
__device__ volatile uint64_t evict_clock = 0;


#define get_linear_model_index() ((*linear_reg_info_idx) & 0x1)
#define estimate_reuse_distance(virt_timestamp_diff) ({\
    uint64_t ret;\
    ret = (uint64_t)(((double)virt_timestamp_diff - (double)linear_reg_info[get_linear_model_index()].offset) / (double)linear_reg_info[get_linear_model_index()].slope);\
    ret;\
})

#define curr_reg_info() linear_reg_info[get_linear_model_index()].slope, linear_reg_info[get_linear_model_index()].offset
#define print_linear_model() printf("index %u - slope %f, offset %f\n", get_linear_model_index(), linear_reg_info[get_linear_model_index()].slope, linear_reg_info[get_linear_model_index()].offset);
    
#define PRINT_MEM_SAMPLE_STATUS 1
class MemSampleCollector {
public:
    MemSampleCollector(LinearRegInfo* _info, uint32_t* _info_idx, host_ring_buffer_t* _ring_buf_h, host_ring_buffer_t* _ring_buf_d) : 
        linear_reg_info_host(_info), linear_reg_info_idx_host(_info_idx), ring_buf_h(_ring_buf_h), ring_buf_d(_ring_buf_d) 
    {
        this->size = 16384;
        this->samples_count = 0;
        this->alpha = 0.1;
        this->linear_reg_constructor = new LinearRegression();
        this->leaving = false;
        
        // Sample buffer
        posix_memalign((void**)&this->sample_buffer, 4096, ring_buf_h->q_size*16);
        posix_memalign((void**)&this->retain_buffer, 4096, ring_buf_h->q_size*16);
        CUDA_SAFE_CALL(cudaHostRegister(this->sample_buffer, ring_buf_h->q_size*16, cudaHostRegisterPortable));
        
        this->retain_idx = 0;
        // Copy stream
        CUDA_SAFE_CALL(cudaStreamCreate(&(this->sample_stream)));

        //
        reuse_dist_calculator = new ReuseDistCalculator();
    }
    MemSampleCollector(GpuRwQueue* _rw_queue) : rw_queue(_rw_queue)
    {
        printf("mem-sample-collector: registering rw queue %p\n", _rw_queue);
        this->leaving = false;
        this->working_on_rw_queue = true;
        reuse_dist_calculator = new ReuseDistCalculator();
    }


    ~MemSampleCollector() 
    {
        this->leaving = true;
        delete reuse_dist_calculator;
        free(this->sample_buffer);
        free(this->retain_buffer);
        //CUDA_SAFE_CALL(cudaFree(linear_reg_info_host));
        CUDA_SAFE_CALL(cudaFree(ring_buf_h));
        CUDA_SAFE_CALL(cudaFree(ring_buf_d));
        CUDA_SAFE_CALL(cudaStreamDestroy(this->sample_stream));
    }

    void run_sgd(float& slope, float& offset) 
    {
        /*
        // run sgd using dist_pair_vec
        float diff = 0, diff_mul_x = 0;
        float virt_timestamp_dist_hat;
        float virt_diff;

        // p.first: virt_timestamp_dist, p.second: reuse_dist
        for (auto &p : dist_pair_vec) {
            virt_timestamp_dist_hat = p.second * slope + offset;
            virt_diff = virt_timestamp_dist_hat - p.first;
            diff += virt_diff;
            diff_mul_x += virt_diff * p.second;
        }

        diff = diff / 64 * this->alpha;
        diff_mul_x = diff_mul_x / 64 * this->alpha;

        offset -= diff;
        slope -= diff_mul_x;
        */
    }

    void run_ols(float& slope, float& offset)
    {
        this->linear_reg_constructor->get_ls_solution(dist_pair_vec.data(), dist_pair_vec.size() >> 1, offset, slope); 
    }

    void push_samples(uint64_t key, uint64_t virt_timestamp_dist) 
    {
        uint64_t estimated_reuse_dist;

        // Get the actual reuse distance
        estimated_reuse_dist = reuse_dist_calculator->update(key);

        // Push to the queue
        //queue.push_back({virt_timestamp_dist, estimated_reuse_dist});
        //while (queue.size() > size) {
        //    queue.pop_left();
        //}
        
        // TODO: thread safe implementation...
        //this->samples_count++;
        /*
        dist_pair_vec.push_back({virt_timestamp_dist, estimated_reuse_dist});
        if (this->samples_count.fetch_add(1) == 64) {
            // Run SGD? OLS?
            if (this->lock.exchange(1) == 0) {
                run_sgd(this->linear_reg_info_host->slope, this->linear_reg_info_host->offset);
                this->samples_count = 0;
                dist_pair_vec.clear();
            }
        }*/
    }

    void start_collector_thread(void)
    {
        std::thread t(&MemSampleCollector::collector_thread, std::ref(*this));
        t.detach();
    }

    void start_rw_queue_thread(void)
    {
        assert(this->working_on_rw_queue);
        std::thread t(&MemSampleCollector::scan_rw_queue, std::ref(*this));
        t.detach();
    }


    uint32_t calculate_reuse_dist_for_samples(uint64_t* samples)
    {
        uint32_t count = 0;
        auto start = std::chrono::high_resolution_clock::now();
        uint64_t estimated_reuse_dist;
        large_rd = 0;
        for (uint32_t i = 0; i < ring_buf_h->q_size; i++) {
            //std::cout << __func__ << " " << i << std::endl;
            //printf("%lu-%lu\n", samples[2*i], samples[2*i+1]);
            uint64_t key  = samples[2*i+1];
            estimated_reuse_dist = reuse_dist_calculator->update(samples[2*i+1]);
            //printf("key %lu - dist %lu (virt dist %lu)\n", samples[2*i+1], estimated_reuse_dist, samples[2*i]);
            samples[2*i+1] = estimated_reuse_dist;

            
            //if (estimated_reuse_dist != NO_REUSE) {
            if (estimated_reuse_dist != NO_REUSE) {
                // TODO: wiered..
                //if (samples[2*i] > ((uint64_t)1<<32)) continue;
            if (estimated_reuse_dist >= 50000) {
                large_rd++;
            }
            if (this->virt_timestamp_map.find(key) == this->virt_timestamp_map.end() || samples[2*i] < this->virt_timestamp_map[key]) {
                this->virt_timestamp_map[key] = samples[2*i];
                continue;
            }
            uint64_t time_diff = samples[2*i] - this->virt_timestamp_map[key];
            this->virt_timestamp_map[key] = samples[2*i];
            samples[2*i] = time_diff;
            
            #if BYPASS_UNREASONABLE_SAMPLES
                if (samples[2*i]/samples[2*i+1] > 100000) continue;
            #endif

            #if BYPASS_INVALID_SAMPLES
                if (samples[2*i+1] > samples[2*i]) continue;

            #if PRINT_TRAINING_PAIRS
                printf("%lu,%lu\n", samples[2*i], samples[2*i+1]);
            #endif 
                if ((samples[2*i] & 0x80000000) != 0) continue;
            //if (estimated_reuse_dist <= 1)  continue;
            #endif
                ((uint64_t*)this->retain_buffer)[2*this->retain_idx] = samples[2*i];
                ((uint64_t*)this->retain_buffer)[2*this->retain_idx+1] = samples[2*i+1];
                this->retain_idx++;
                count++;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
        #if PRINT_MEM_SAMPLE_STATUS
        std::cout << __func__ << " takes " << std::dec << elapsed.count() << " us. (count " << count << ")" << std::endl;
        #endif
        return count;
    }
    
    #define ENABLE_FILTER 0
    void collector_thread (void)
    {
        //printf("ring buffer ready flag %u (%p)\n", *(ring_buf_h->ready), ring_buf_h->ready);
        while (!leaving) {
            while (!*(ring_buf_h->ready));
            void* gpu_addr = (void*)ring_buf_h->buffer;
            cudaMemcpyAsync((void*)this->sample_buffer, (const void*)gpu_addr, ring_buf_h->q_size*16, 
                            cudaMemcpyDeviceToHost, this->sample_stream);
            cudaStreamSynchronize(this->sample_stream);
            __atomic_store_n(ring_buf_h->ready, false, __ATOMIC_SEQ_CST);
            
            //if (*(uint64_t*)this->sample_buffer == 0) continue;
            //for (int i = 0; i < 32/*ring_buf_h->q_size*/; i++) {
            //    printf("%d: %lu,%lu\n", i, ((uint64_t*)this->sample_buffer)[2*i], ((uint64_t*)this->sample_buffer)[2*i+1]);               
            //}
            large_rd = 0;
            if (calculate_reuse_dist_for_samples((uint64_t*)this->sample_buffer) >= (ring_buf_h->q_size>>1)) {
                uint32_t next_idx = ((*linear_reg_info_idx_host)+1) & 0x1;
                linear_reg_constructor->get_ls_solution((uint64_t*)this->sample_buffer, ring_buf_h->q_size, linear_reg_info_host[next_idx].offset, linear_reg_info_host[next_idx].slope); 

                #if DYNAMIC_SAMPLE_FREQ
                if (linear_reg_info_host[next_idx].slope < 0) {
                    invalid_slope_num++;
                    if (invalid_slope_num >= 2 && invalid_slope_num < 32 && (*dynamic_sample_freq_h) < 256) {
                        (*dynamic_sample_freq_h) *= 8;
                    }
                }
                #endif

                // @0625: add a filter
                #if ENABLE_FILTER
                double n = ((double)(*linear_reg_info_idx_host));
                linear_reg_info_host[(next_idx)].slope = (linear_reg_info_host[(next_idx+1)&0x1].slope * n + linear_reg_info_host[next_idx].slope)/(n+1.0);
                linear_reg_info_host[(next_idx)].offset = (linear_reg_info_host[(next_idx+1)&0x1].offset * n + linear_reg_info_host[next_idx].offset)/(n+1.0);
                #endif
                if (linear_reg_info_host[next_idx].slope == 1.0) {
                    continue;
                }
                #if ENABLE_COMPENSATION
                else if (linear_reg_info_host[next_idx].slope < 100) {
                    linear_reg_info_host[next_idx].slope *= COMP_FACTOR;
                }
                else if (linear_reg_info_host[next_idx].slope < 1000) {
                #if BYPASS_UNREASONABLE_SAMPLES
                    linear_reg_info_host[next_idx].slope *= 1;
                #else
                    linear_reg_info_host[next_idx].slope *= 10;
                #endif

                }
                #if SMART_DOWN_SCALING // avoid clogged samples bias
                else if (linear_reg_info_host[next_idx].slope > 100000 && large_rd == 0) {
                    linear_reg_info_host[next_idx].slope /= 100.0;
                }
                //else if (linear_reg_info_host[next_idx].slope > 10000 && large_rd == 0) {
                //    linear_reg_info_host[next_idx].slope /= 10.0;
                //}

                #endif
                #endif

                __atomic_add_fetch(linear_reg_info_idx_host, 1, __ATOMIC_SEQ_CST);
                #if PRINT_MEM_SAMPLE_STATUS
                printf("%s: linear_reg_info_idx %u (slope %f, offset %f)\n", __func__, (*linear_reg_info_idx_host) % 2, linear_reg_info_host[next_idx].slope, linear_reg_info_host[next_idx].offset);
                #endif
                //usleep(1);
                //__atomic_store_n(ring_buf_h->ready, false, __ATOMIC_SEQ_CST);
                this->retain_idx = 0;
            }
            else {
                if (this->retain_idx >= (ring_buf_h->q_size>>1)) {
                    uint32_t next_idx = ((*linear_reg_info_idx_host)+1) & 0x1;
                    linear_reg_constructor->get_ls_solution((uint64_t*)this->retain_buffer, this->retain_idx, linear_reg_info_host[next_idx].offset, linear_reg_info_host[next_idx].slope); 
                    //if (large_rd == 0) {
                    if (linear_reg_info_host[next_idx].slope == 1.0) {
                        continue;
                    }
                    #if ENABLE_COMPENSATION
                    if (linear_reg_info_host[next_idx].slope < 100) {
                        linear_reg_info_host[next_idx].slope *= COMP_FACTOR;
                    }
                    else if (linear_reg_info_host[next_idx].slope < 1000) {
                        linear_reg_info_host[next_idx].slope *= 10;
                    }
                    #if SMART_DOWN_SCALING // avoid clogged samples bias
                    else if (linear_reg_info_host[next_idx].slope > 100000 && large_rd == 0) {
                        linear_reg_info_host[next_idx].slope /= 100.0;
                    }
                    #endif
                    #endif


                    __atomic_add_fetch(linear_reg_info_idx_host, 1, __ATOMIC_SEQ_CST);
                    #if PRINT_MEM_SAMPLE_STATUS
                    printf("%s: (use retain) linear_reg_info_idx %u (slope %f, offset %f)\n", __func__, (*linear_reg_info_idx_host) % 2, linear_reg_info_host[next_idx].slope, linear_reg_info_host[next_idx].offset);
                    #endif
                    this->retain_idx = 0;
                }
            }
        }
    }

    void scan_rw_queue (void)
    {
        int queue_index = 0;
        int req_count = 0;
        printf("%s starting... (rw-queue %p)\n", __func__, this->rw_queue);
        do {
            //fprintf(stderr, "%s: checking queue_index %d\n", __func__, queue_index);
            while (this->rw_queue->entries[queue_index].status != GPU_RW_PENDING);
            if (this->rw_queue->entries[queue_index].status == GPU_RW_PENDING) {
                //this->rw_queue->entries[queue_index].status = GPU_RW_READY;
                //continue;
                //fprintf(stderr, "%s: checked queue_index %d\n", __func__, queue_index);
                this->rw_queue->entries[queue_index].status = GPU_RW_PROCESSING;
                // spawn another to address the request
                //std::thread handler_thread(&HostCache::handleRequest, std::ref(*this), queue_index);
                //handler_thread.detach();
                this->handle_rw_req(queue_index);
            }
            queue_index = (queue_index + 1) & (GPU_RW_SIZE-1);
        } while (!leaving);
    }

    void handle_rw_req(int queue_index)
    {
        uint32_t type             = this->rw_queue->entries[queue_index].type;
        uint64_t key              = this->rw_queue->entries[queue_index].key;
        
        if (type == GPU_RW_SAMPLE_MEM || type == GPU_RW_SAMPLE_MEM_EVICTION || type == GPU_RW_SAMPLE_MEM_RE_REF) {
            uint64_t actual_reuse_dist;
            // Get the actual reuse distance
            //fprintf(stderr, "%s: for queue index %d\n", __func__, queue_index);
            if (type == GPU_RW_SAMPLE_MEM) {
                actual_reuse_dist = reuse_dist_calculator->update(key);
            }
            else if (type == GPU_RW_SAMPLE_MEM_EVICTION) {
                actual_reuse_dist = reuse_dist_calculator->get_dist_from_key_to_end(key);
            }
            else if (type == GPU_RW_SAMPLE_MEM_RE_REF) {
                actual_reuse_dist = reuse_dist_calculator->get_dist_upon_re_ref(key);
                //printf("Re-ref::: %u... %lu\n", type, actual_reuse_dist);
            }
            this->rw_queue->entries[queue_index].u.SampleMemResponse.actual_reuse_dist = actual_reuse_dist;
            __sync_synchronize();
            //printf("getting actual reuse dist of page %lu : %lu\n", key, actual_reuse_dist);
        }

        //*/
        this->rw_queue->entries[queue_index].status = GPU_RW_READY;
        __sync_synchronize();
    }

    bool                                        leaving;
    bool                                        working_on_rw_queue;
    void*                                       sample_buffer;
    void*                                       retain_buffer;
    float                                       alpha; // For SGD decay rate
    uint64_t                                    size;
    //uint64_t                                    samples_count;
    cudaStream_t                                sample_stream;
    host_ring_buffer_t*                         ring_buf_h;
    host_ring_buffer_t*                         ring_buf_d;
    uint32_t                                    retain_idx;
    uint32_t*                                   linear_reg_info_idx_host;
    LinearRegInfo*                              linear_reg_info_host;
    LinearRegression*                           linear_reg_constructor;
    ReuseDistCalculator*                        reuse_dist_calculator;
    volatile GpuRwQueue*                        rw_queue;
    std::atomic<uint64_t>                       samples_count;
    std::deque<std::pair<uint64_t,uint64_t>>    queue;
    std::vector<uint64_t> dist_pair_vec; // virt_timestamp_dist, estimated_reuse_dist

    uint32_t sample_factor = 1;
    std::unordered_map<uint64_t,uint64_t> virt_timestamp_map;
    uint32_t large_rd = 0;
};

HostCache* createHostCache(Controller* ctrl);
void revokeHostRuntime();
void preLoadData(int fd, size_t page_size);
//void registerGPUMem(void*, size_t);

size_t pc_mem_size;

//extern __device__ volatile GpuOpenFileTable* gpu_open_table;
__device__ volatile GpuFileRwManager* gpu_rw_manager;
__device__ volatile GpuFileRwManager* gpu_rw_manager_mem;
__device__ volatile GpuRwQueue* gpu_rw_queue;
__device__ volatile GpuRwQueue* gpu_rw_queue_mem;
__device__ volatile HostCacheRuntimeState* gpu_hc_runtime_state;
__device__ PageProfileInfo* page_info_dev;
__device__ volatile Shct* gpu_shct;
__device__ volatile uint32_t kernel_iter = 0;
__device__ void* host_cache_base = NULL;
__device__ uint64_t first_tier_mem_pages = 0;
__device__ uint64_t second_tier_mem_pages = 0;
//__device__ volatile PageReuseTable** page_reuse_pattern;
__device__ host_ring_buffer_t *mem_samples_ring_buf = NULL;

__device__ volatile bool last_iteration = false;

__device__ volatile uint64_t* tmp_buf = NULL;
volatile uint64_t* tmp_buf_h = NULL;

// Chia-Hao: making signed integer on 070823
__device__ volatile int64_t* num_idle_slots;
volatile int64_t* num_idle_slots_h = NULL;

__device__ curandState* curand_state = NULL;
curandState* curand_state_h = NULL;
__global__ void init_rand_generator() { curand_init(9527, 7, 13, curand_state); }

#define idle_slots_too_few(threshold) ((*num_idle_slots) < threshold)

/* Host global */
HostCache * volatile host_cache = NULL;
StreamManager* stream_mngr = NULL;


// Analyze reuse distance
std::mutex logging_mtx;
uint64_t reuse_stack_size = 0;
std::unordered_map<uint64_t, uint64_t> reuse_stack; // (array_idx << 32 | page_idx)
std::unordered_map<uint64_t, uint64_t> reuse_stack_map; // key, index of reuse stack
std::unordered_map<uint64_t, uint64_t> reuseDistTable; // distance, frequency


/* Static Variables */
static bool revoke_runtime = false;
static bool revoke_runtime_done = false;
static bool preload = false;
static pthread_t host_runtime_thread;
static pthread_t flush_thread[NUM_FLUSH_THREADS];
static uint64_t flush_thread_args[NUM_FLUSH_THREADS];
static pthread_attr_t host_runtime_thread_attr;
static void* start_host_runtime_thread(void* arg);
static void* host_flush_page(void* arg);
static inline uint64_t get_lba(HostCache* hc, uint64_t key);

/* SPIN driver*/
//int spin_fd = -1;
//typedef unsigned long long u64;
//extern int register_file(char*, int);
//extern int register_gpu_mem(void*, size_t);
//extern ssize_t p2p_pread64(int, void*, size_t, off_t); 
//extern ssize_t pread64(int, void*, size_t, off_t); 
//extern ssize_t pread64_with_pc(int, void*, size_t, off_t, void*, void*&); 
//extern ssize_t p2p_pwrite64(int, void*, size_t, off_t); 
//extern void myinit(int spin_fd);
//extern void myend();

typedef unsigned long long u64;

inline void print_content(void* host_mem, uint64_t page)
{
    //unsigned char* ptr = (unsigned char*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, page);
    uint64_t* ptr = (uint64_t*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, page);
    printf("page %lu: ", page);
    for (int i = 0; i < 32; i++) printf("%lx", ptr[i]);
    printf("\n");
}

inline bool find_reuse_page(uint64_t key, int64_t& pos) 
{
    for (pos = reuse_stack.size()-1; pos >= 0; pos--) {
        if (key == reuse_stack[pos]) {
            return true;
        }
    }
    return false;
}

inline void read_data_to_hc(HostCache* hc, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long hc_entry, uint32_t queue_id = 0) 
{
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(hc->qp[queue_id]->sq), &(hc->sq_host[queue_id]));

    nvm_cmd_header(&cmd, cid, NVM_IO_READ, hc->qp[queue_id]->nvmNamespace);
    uint64_t prp1 = hc->prp1[hc_entry];
    uint64_t prp2 = hc->prp2[hc_entry];

    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&hc->qp[queue_id]->sq, &(hc->sq_host[queue_id]), &cmd);
    uint32_t head, head_;
    __attribute__ ((unused)) uint64_t pc_pos;
    __attribute__ ((unused)) uint64_t pc_prev_head;

    uint32_t cq_pos = cq_poll(&hc->qp[queue_id]->cq, cid, &head, &head_);

    hc->qp[queue_id]->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    pc_prev_head = hc->q_head->load(simt::memory_order_relaxed);
    pc_pos = hc->q_tail->fetch_add(1, simt::memory_order_acq_rel);

    //(void)pc_pos, (void)pc_prev_head;
    cq_dequeue(&hc->qp[queue_id]->cq, &(hc->cq_host[queue_id]), cq_pos, &hc->qp[queue_id]->sq, head, head_);
    //sq_dequeue(&qp->sq, sq_pos);

    put_cid(&hc->qp[queue_id]->sq, &(hc->sq_host[queue_id]), cid);
}


inline void write_data_from_hc(HostCache* hc, const uint64_t starting_lba, const uint64_t n_blocks, const uint64_t virt_page, uint32_t queue_id = 0) 
{
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(hc->qp[queue_id]->sq), &(hc->sq_host[queue_id]));
    //fprintf(stderr, "%s: cid: %u (start lba %lu, n_blocks %lu, virt page %lu)\n", __func__, (unsigned int) cid, starting_lba, n_blocks, virt_page);

    nvm_cmd_header(&cmd, cid, NVM_IO_WRITE, hc->qp[queue_id]->nvmNamespace);
    uint64_t prp1 = hc->prp1[virt_page];
    uint64_t prp2 = hc->prp2[virt_page];

    //fprintf(stderr, "prp1 %lx\n", prp1);
    //uint64_t* pp = (uint64_t*)prp2;
    //for (uint32_t i = 0; i < 127; i++) fprintf(stderr, "prp2[%u] %lu\n", pp[i]);

    ////printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&hc->qp[queue_id]->sq, &(hc->sq_host[queue_id]), &cmd);
    uint32_t head, head_;
    __attribute__ ((unused)) uint64_t pc_pos;
    __attribute__ ((unused)) uint64_t pc_prev_head;

    //printf("submit to host sq (pos %u)\n", sq_pos);
    uint32_t cq_pos = cq_poll(&hc->qp[queue_id]->cq, cid, &head, &head_);
    //printf("after cq_poll -> cq_pos %u\n", cq_pos);
    hc->qp[queue_id]->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    pc_prev_head = hc->q_head->load(simt::memory_order_relaxed);
    pc_pos = hc->q_tail->fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&hc->qp[queue_id]->cq, &(hc->cq_host[queue_id]), cq_pos, &hc->qp[queue_id]->sq, head, head_);
    //sq_dequeue(&qp->sq, sq_pos);

    //printf("cq_dequeue -> cq_pos %u\n", cq_pos);
    put_cid(&hc->qp[queue_id]->sq, &(hc->sq_host[queue_id]), cid);
}

inline uint16_t submit_io_async(HostCache* hc, const uint64_t starting_lba, const uint64_t n_blocks, const uint64_t virt_page, uint32_t queue_id = 0) 
{
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(hc->qp[queue_id]->sq), &(hc->sq_host[queue_id]));
    //fprintf(stderr, "%s: cid: %u (start lba %lu, n_blocks %lu, virt page %lu)\n", __func__, (unsigned int) cid, starting_lba, n_blocks, virt_page);

    nvm_cmd_header(&cmd, cid, NVM_IO_WRITE, hc->qp[queue_id]->nvmNamespace);
    uint64_t prp1 = hc->prp1[virt_page];
    uint64_t prp2 = hc->prp2[virt_page];

    //fprintf(stderr, "prp1 %lx\n", prp1);
    //uint64_t* pp = (uint64_t*)prp2;
    //for (uint32_t i = 0; i < 127; i++) fprintf(stderr, "prp2[%u] %lu\n", pp[i]);

    ////printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&hc->qp[queue_id]->sq, &(hc->sq_host[queue_id]), &cmd);

    return cid;
}

inline void poll_io(HostCache* hc, const uint16_t cid, uint32_t queue_id = 0) 
{
    uint32_t head, head_;
    __attribute__ ((unused)) uint64_t pc_pos;
    __attribute__ ((unused)) uint64_t pc_prev_head;

    uint32_t cq_pos = cq_poll(&hc->qp[queue_id]->cq, cid, &head, &head_);
    //printf("after cq_poll -> cq_pos %u\n", cq_pos);
    hc->qp[queue_id]->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    pc_prev_head = hc->q_head->load(simt::memory_order_relaxed);
    pc_pos = hc->q_tail->fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&hc->qp[queue_id]->cq, &(hc->cq_host[queue_id]), cq_pos, &hc->qp[queue_id]->sq, head, head_);

    //printf("cq_dequeue -> cq_pos %u\n", cq_pos);
    put_cid(&hc->qp[queue_id]->sq, &(hc->sq_host[queue_id]), cid);
}

__host__ void GpuOpenFileTableEntry::init() volatile
{
    this->cpu_fd = -1;
    this->status = ENTRY_AVAILABLE;
}

__host__ void GpuOpenFileTable::init() volatile
{
    for (int i = 0; i < GPU_OPEN_FILE_TABLE_SIZE; i++) {
        entries[i].init();
    }
}

//__device__ __forceinline__ ssize_t remote_read_write(size_t _size, off_t _buffer_offset, size_t _file_offset, int req_type, void* dev_data_ptr, int rw_queue_index, int* bid)
__device__ __forceinline__ int remote_read_write(uint32_t req_type, uint64_t key, void* dev_data_ptr, int rw_queue_index, int* bid, bool dirty, bool* is_cached_page_dirty = NULL)
{   
    uint8_t ns = 8;
    LocalEntry local;
    local.type = req_type;
    //local.status = GPU_RW_PENDING;
    #if USE_PINNED_MEM
    //local.pin_on_host = true;
    #endif
    local.dirty = dirty;
    local.key = key;
    local.gpu_addr = dev_data_ptr;
    local.bid = *bid;

    //gpu_rw_queue->entries[rw_queue_index].type = req_type;
    //gpu_rw_queue->entries[rw_queue_index].gpu_addr = dev_data_ptr;
    //gpu_rw_queue->entries[rw_queue_index].status = GPU_RW_PENDING;
    //if (req_type == GPU_RW_FETCH_FROM_HOST)
    //    gpu_rw_queue->entries[rw_queue_index].u.CacheReq.bid = *bid;

    //GPU_PRINT("%s: size %lu, buffer offset %lu, file offset %lu, req %d, gpu buffer addr %p\n", __func__, _size, _buffer_offset, _file_offset, req_type, dev_data_ptr);
    //GPU_ASSERT(gpu_rw_queue->entries[rw_queue_index].status == GPU_RW_EMPTY);
    memcpy((void*)&(gpu_rw_queue->entries[rw_queue_index]), (void*)&local, sizeof(LocalEntry));

    __threadfence_system();
    
    gpu_rw_queue->entries[rw_queue_index].status = GPU_RW_PENDING;

    __threadfence_system();

    WAIT_COND(gpu_rw_queue->entries[rw_queue_index].status, GPU_RW_READY, ns);
    //dev_data_ptr = (void*)((uint64_t)(gpu_rw_manager->getBufferBase()) + (uint64_t)buffer_offset);
    if (req_type == GPU_RW_EVICT_TO_HOST)
        *bid = gpu_rw_queue->entries[rw_queue_index].bid;
    
    if (req_type == GPU_RW_FETCH_FROM_HOST && is_cached_page_dirty != NULL) {
        *is_cached_page_dirty = gpu_rw_queue->entries[rw_queue_index].u.CacheReq.is_cached_page_dirty;
    }

    //GPU_PRINT("request completion: addr %llx", (unsigned long long)dev_data_ptr);

    //return gpu_rw_queue->entries[rw_queue_index].u.RwReq.return_size;
    return gpu_rw_queue->entries[rw_queue_index].u.CacheReq.return_code;
}

__host__ void GpuRwQueueEntry::init() volatile
{
    status = GPU_RW_EMPTY;
}

__device__ GpuRwQueueEntry* GpuRwQueue::getEntryByIndex(int entry_index)
{
    return (&(entries[entry_index])); 
}

__host__ void GpuRwQueue::init() volatile
{
    for (int i = 0; i < GPU_RW_SIZE; i++) {
        //entries[i]->init();
        entries[i].init();
        valid_in_pc[i] = malloc(DIV_ROUND_UP(64*1024*1024, 4096));
    }
}

__device__ void* GpuFileRwManager::getBufferBase() volatile
{
    return (void*)buffer_base;
}
    
__forceinline__ __device__ int GpuFileRwManager::getRWQueueEntry() volatile
{
    uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) & (GPU_RW_SIZE - 1);
    uint32_t ns = 2;
    do {
        if (atomicExch((int*)&entry_lock[i], GPU_LOCKED) == GPU_NO_LOCKED) {
            return i;
        }
        else {
            i = (i + 1) & (GPU_RW_SIZE - 1);
        }
        __nanosleep(ns);
        if (ns < 32) ns *= 2;
    } while (1);
}
 
__forceinline__ __device__ int GpuFileRwManager::getRWQueueEntry2() volatile
{
    uint32_t ret = this->ticket.fetch_add(1, simt::memory_order_acq_rel);
    uint32_t ret_q = ret & (GPU_RW_SIZE-1);
    do {
        if (atomicExch((int*)&entry_lock[ret_q], GPU_LOCKED) == GPU_NO_LOCKED) {
            return ret_q;
        }
    } while (1);
    //assert(entry_lock[(ret & (GPU_RW_SIZE-1))] == GPU_NO_LOCKED);
    return ret_q;
}

__device__ void GpuFileRwManager::putRWQueueEntry(int entry_index) volatile
{
    entry_lock[entry_index] = GPU_NO_LOCKED;
    __threadfence();
}

__host__ void GpuFileRwManager::init(GpuRwQueue* _gpu_rw_queue) volatile
{
    gpu_rw_queue = _gpu_rw_queue;
    for (int i = 0; i < GPU_RW_SIZE; i++) {
        entry_lock[i] = GPU_NO_LOCKED;
    }
}

//__forceinline__ __device__
__device__
uint32_t getTierByReuseDistance(uint64_t reuse_dist)
{
    uint32_t ret = 0;
    //printf("reuse_dist %lu (first-tier %lu _ second-tier %lu\n", reuse_dist, first_tier_mem_pages, second_tier_mem_pages);
    if (reuse_dist < first_tier_mem_pages) {
        ret = 1;
    }
    //else if (reuse_dist >= first_tier_mem_pages && reuse_dist < (second_tier_mem_pages+first_tier_mem_pages)) {
    else if (reuse_dist >= first_tier_mem_pages && reuse_dist < (second_tier_mem_pages)) {
        ret = 2;
    }
    else {
        ret = 3;
    }

    return ret;
}

__device__ void* readDataFromHost(int _fd, size_t _size, off_t _offset)
{
    //int req_type = GPU_RW_READ;
    //int rw_queue_index = -1;
    void* buffer_ptr = NULL;
    //void* data_ptr = NULL;
    //ssize_t return_size;
    //GpuRwQueueEntry* rw_queue_entry = NULL;
    
    GPU_PRINT("%s for %d, %lu, %lu\n", __func__, _fd, _size, _offset);
    // get a valid queue entry to submit request
    //rw_queue_index = gpu_rw_manager->getRWQueueEntry();
    //rw_queue_entry = (GpuRwQueueEntry*)(gpu_rw_queue->getEntryByIndex(rw_queue_index));
    //rw_queue_entry = (GpuRwQueueEntry*)&gpu_rw_queue->entries[rw_queue_index];

    // FIXME
    // Temporarily use buffer base as the buffer address
    // Afterwards we should develop buffer cache mechanism
    buffer_ptr = gpu_rw_manager->getBufferBase();
    
    // Call remote read API
    //uint32_t bid;
    //return_size = remote_read_write(_size, 0/*buffer offset*/, 0 /*file offset*/, req_type, buffer_ptr, rw_queue_index, bid);
    
    //if (return_size == -1) return NULL;
    
    return buffer_ptr;
}

__device__ bool read_data_from_host(int _fd, size_t _size, off_t _offset, void* gpu_addr)
{
    //int req_type = GPU_RW_READ;
    int rw_queue_index = -1;
    //void* data_ptr = NULL;
    //GpuRwQueueEntry* rw_queue_entry = NULL;
    //ssize_t return_size;

    // get a valid queue entry to submit request
#if GET_queue_indexRW_QUEUE_ENTRY_ALTER 
    rw_queue_index = gpu_rw_manager->getRWQueueEntry2();
#else
    rw_queue_index = gpu_rw_manager->getRWQueueEntry();
#endif
    //rw_queue_entry = (GpuRwQueueEntry*)&gpu_rw_queue->entries[rw_queue_index];

    // Call remote read API
    //return_size = remote_read_write(_size, 0/*buffer offset*/, _offset /*file offset*/, req_type, gpu_addr, rw_queue_index);
    
    // release the lock the entry
    gpu_rw_manager->putRWQueueEntry(rw_queue_index);

    //if (return_size == -1) return false;
    
    return true;
}

    
__forceinline__ __device__ 
int writeDataToHost(int _fd, size_t _size, off_t _offset, void* gpu_addr)
{
    //int req_type                        = GPU_RW_WRITE;
    int rw_queue_index                  = -1;
    //void* buffer_ptr                    = NULL;
    //void* data_ptr                      = NULL;
    //GpuRwQueueEntry* rw_queue_entry   = NULL;
    
    //ssize_t return_size;

    // get a valid queue entry to submit request
#if GET_RW_QUEUE_ENTRY_ALTER 
    rw_queue_index = gpu_rw_manager->getRWQueueEntry2();
#else
    rw_queue_index = gpu_rw_manager->getRWQueueEntry();
#endif
    //rw_queue_entry = (GpuRwQueueEntry*)&gpu_rw_queue->entries[rw_queue_index];

    // Call remote read API
    //return_size = remote_read_write(_size, 0/*buffer offset*/, _offset /*file offset*/, req_type, gpu_addr, rw_queue_index);

    // release the lock the entry
    gpu_rw_manager->putRWQueueEntry(rw_queue_index);

    //if (return_size == -1) return -1;
   
    return 0;
}

__forceinline__ __device__ 
int accessHostCache(void* gpu_addr, uint64_t key, uint32_t req_type, int* bid, bool dirty, bool* is_cached_page_dirty = NULL)
{
    //int req_type                        = __req_type;
    int rw_queue_index                  = -1;
    //GpuRwQueueEntry* rw_queue_entry   = NULL;
    
    int return_code;

    GPU_PRINT("%s - req type: %s - gpu addr %lx\n", __func__, enum_rw_req[req_type], (uint64_t)gpu_addr);
    
    // Get a valid queue entry to submit request
    //printf("Try to get queue index\n");
#if GET_RW_QUEUE_ENTRY_ALTER 
    rw_queue_index = gpu_rw_manager->getRWQueueEntry2();
#else
    rw_queue_index = gpu_rw_manager->getRWQueueEntry();
#endif
    //printf("Got index %d\n", rw_queue_index);
    //rw_queue_entry = (GpuRwQueueEntry*)&gpu_rw_queue->entries[rw_queue_index];

    // Call remote read API
    //return_size = remote_read_write(PAGE_SIZE, 0/*buffer offset*/, 0 /*file offset*/, req_type, gpu_addr, rw_queue_index, bid);
    //return_code = remote_read_write(req_type, key, gpu_addr, rw_queue_index, bid, dirty);
    // TODO: Chia-Hao: should change later, 100123
    return_code = remote_read_write(req_type, key, gpu_addr, rw_queue_index, bid, dirty, is_cached_page_dirty);

    // Release the lock the entry
    gpu_rw_manager->putRWQueueEntry(rw_queue_index);
    
    GPU_PRINT("%s - req type: %s - gpu addr %lx access done.\n", __func__, enum_rw_req[req_type], (uint64_t)gpu_addr);
    //printf("%s - req type: %s - gpu addr %lx\n access done.", __func__, enum_rw_req[__req_type], (uint64_t)gpu_addr);
    //if (return_size == -1) return -1;
   
    return return_code;
}

__forceinline__ __device__
int sendReqToHostCache(uint64_t key, int bid, uint32_t req_type, bool* is_cached_page_dirty = NULL)
{
    uint8_t ns = 8;
    LocalEntry local;
    local.type = req_type;
    //local.status = GPU_RW_PENDING;
    local.key = key;
    local.bid = bid;

    //printf("%s: %s - key %lu, bid %u\n", __func__, enum_rw_req[req_type], key, bid);
#if GET_RW_QUEUE_ENTRY_ALTER 
    int rw_queue_index = gpu_rw_manager->getRWQueueEntry2();
#else
    int rw_queue_index = gpu_rw_manager->getRWQueueEntry();
#endif
    memcpy((void*)&(gpu_rw_queue->entries[rw_queue_index]), (void*)&local, sizeof(LocalEntry));
    __threadfence_system();

    gpu_rw_queue->entries[rw_queue_index].status = GPU_RW_PENDING;
    __threadfence_system();
    
    WAIT_COND(gpu_rw_queue->entries[rw_queue_index].status, GPU_RW_READY, ns);

    // Chia-Hao: 070423
    if (is_cached_page_dirty) {
        *is_cached_page_dirty = gpu_rw_queue->entries[rw_queue_index].u.CacheReq.is_cached_page_dirty;
    }

    gpu_rw_manager->putRWQueueEntry(rw_queue_index);

    return gpu_rw_queue->entries[rw_queue_index].u.CacheReq.return_code;
}

__forceinline__ __device__
int pushMemSampleToHost(uint64_t key, int req = GPU_RW_SAMPLE_MEM)
{
    uint8_t ns = 8;
    LocalEntry local;
    local.type = req;
    local.key = key;

    //printf("%s: trying to get queue index for %lu (block-id %u, thread-id %u)\n", __func__, key, (uint32_t)blockIdx.x, (uint32_t)threadIdx.x);
//#if GET_RW_QUEUE_ENTRY_ALTER 
//    int rw_queue_index = gpu_rw_manager_mem->getRWQueueEntry2();
//#else
//    int rw_queue_index = gpu_rw_manager_mem->getRWQueueEntry();
//#endif
    int rw_queue_index = gpu_rw_manager_mem->getRWQueueEntry2();

    //printf("%s: queue index %d (%lu) (thread id %lu)\n", __func__, rw_queue_index, key, (uint64_t)blockIdx.x*blockDim.x+threadIdx.x);
    memcpy((void*)&(gpu_rw_queue_mem->entries[rw_queue_index]), (void*)&local, sizeof(LocalEntry));
    __threadfence_system();

    //printf("%s: queue index %d (%lu)\n", __func__, rw_queue_index, key);
    gpu_rw_queue_mem->entries[rw_queue_index].status = GPU_RW_PENDING;
    __threadfence_system();
    

    //printf("%s: start... %lu\n", __func__, key);
    WAIT_COND(gpu_rw_queue_mem->entries[rw_queue_index].status, GPU_RW_READY, ns);
    gpu_rw_manager_mem->putRWQueueEntry(rw_queue_index);
    //printf("%s: finished... %lu\n", __func__, key);

    return gpu_rw_queue_mem->entries[rw_queue_index].u.SampleMemResponse.actual_reuse_dist;
}

__forceinline__ __device__
bool getPredictionResult(uint64_t actual_reuse_dist, uint32_t predicted_tier) 
{
    // if prediction is correct, return true.
    // otherwise, return false

    uint32_t actual_tier = getTierByReuseDistance(actual_reuse_dist);
    return (actual_tier == predicted_tier);
}

__forceinline__ __device__
int getReqsNumOnHostCache()
{
    uint8_t ns = 8;
    LocalEntry local;
    local.type = GPU_RW_GET_REQS_NUM;
    local.status = GPU_RW_PENDING;

#if GET_RW_QUEUE_ENTRY_ALTER 
    int rw_queue_index = gpu_rw_manager->getRWQueueEntry2();
#else
    int rw_queue_index = gpu_rw_manager->getRWQueueEntry();
#endif
    memcpy((void*)&(gpu_rw_queue->entries[rw_queue_index]), (void*)&local, sizeof(LocalEntry));
    __threadfence_system();
    
    WAIT_COND(gpu_rw_queue->entries[rw_queue_index].status, GPU_RW_READY, ns);

    return gpu_rw_queue->entries[rw_queue_index].u.CacheReq.num_reqs;
}

__forceinline__ __device__ 
int accessHostCacheAsync(uint64_t tag, void* gpu_addr, int __req_type)
{
    //int req_type                        = __req_type;
    int rw_queue_index                  = -1;
    //void* buffer_ptr                    = NULL;
    //void* data_ptr                      = NULL;
    uint8_t ns                          = 8;
    //GpuRwQueueEntry* rw_queue_entry     = NULL;
    //ssize_t return_size;

    //GPU_PRINT("%s - req type: %s - gpu addr %lx\n", __func__, enum_rw_req[__req_type], (uint64_t)gpu_addr);
    
    // Get a valid queue entry to submit request
#if GET_RW_QUEUE_ENTRY_ALTER 
    rw_queue_index = gpu_rw_manager->getRWQueueEntry2();
#else
    rw_queue_index = gpu_rw_manager->getRWQueueEntry();
#endif
    //rw_queue_entry = (GpuRwQueueEntry*)&gpu_rw_queue->entries[rw_queue_index];

    // Call remote read API
    //return_size = remote_read_write(PAGE_SIZE, 0/*buffer offset*/, 0 /*file offset*/, req_type, gpu_addr, rw_queue_index, bid);
    gpu_rw_queue->entries[rw_queue_index].type = GPU_RW_EVICT_TO_HOST_ASYNC;
    gpu_rw_queue->entries[rw_queue_index].gpu_addr = gpu_addr;
    gpu_rw_queue->entries[rw_queue_index].u.ReplicationReq.tag = tag;
    gpu_rw_queue->entries[rw_queue_index].status = GPU_RW_PENDING;

    //GPU_PRINT("%s: size %lu, buffer offset %lu, file offset %lu, req %d, gpu buffer addr %p\n", __func__, _size, _buffer_offset, _file_offset, req_type, dev_data_ptr);
    //GPU_ASSERT(gpu_rw_queue->entries[rw_queue_index].status == GPU_RW_EMPTY);
    __threadfence_system();
    
    WAIT_COND(gpu_rw_queue->entries[rw_queue_index].status, GPU_RW_ASYNC_PROCESSING, ns);
    //dev_data_ptr = (void*)((uint64_t)(gpu_rw_manager->getBufferBase()) + (uint64_t)buffer_offset);
    //if (req_type == GPU_RW_EVICT_TO_HOST)
    //    *bid = gpu_rw_queue->entries[rw_queue_index].u.CacheReq.bid;
    
    // Release the lock the entry
    gpu_rw_manager->putRWQueueEntry(rw_queue_index);
    
    //GPU_PRINT("%s - req type: %s - gpu addr %lx\n access done.", __func__, enum_rw_req[__req_type], (uint64_t)gpu_addr);
    //if (return_size == -1) return -1;
   
    return 0;
}

__forceinline__ __device__
void* get_host_cache_addr(int bid)
{
    return NVM_PTR_OFFSET(host_cache_base, PAGE_SIZE, bid);
}

__forceinline__ __device__
int copyFromHostCache(void* data_in, void* data_out, size_t bytes, uint32_t eq_mask)
{
    // One warp enters here
    //int lane = lane_id();
    uint64_t* data_in_int = (uint64_t*)data_in;
    uint64_t* data_out_int = (uint64_t*)data_out;
    //uint32_t num_threads = __popc(__activemask());
    uint32_t num_threads = __popc(eq_mask);
    uint32_t oid = get_oid();

    //printf("oid %u, num_threads %u\n", oid, num_threads);
    for (uint64_t i = oid; i < bytes/sizeof(uint64_t); i += num_threads) {
        data_out_int[i] = data_in_int[i];
        //printf("dataou[%lu]: %x, dataint[%lu]:%x\n", i, data_out_int[i], i, data_in_int[i]);
    }

    return bytes;
}

typedef ulonglong4 vectype;

__forceinline__ __device__
int copyFromHostCacheVec(void* data_in, void* data_out, size_t bytes, uint32_t eq_mask)
{
    vectype* data_in_int = reinterpret_cast<vectype*>(data_in);
    vectype* data_out_int = reinterpret_cast<vectype*>(data_out);
    uint32_t num_threads = __popc(__activemask());
    uint32_t oid = get_oid();

    for (uint64_t i = oid; i < bytes/sizeof(vectype); i += num_threads) {
        data_out_int[i] = data_in_int[i];
    }

    return bytes;
}

__forceinline__ __device__
void loggingPageAction(uint64_t arrayIdx, uint64_t pageIdx, uint32_t actionType)
{
    uint8_t ns                          = 8;   
    uint32_t req_type                   = GPU_RW_LOGGING;
    int32_t rw_queue_index              = 0xFFFFFFFF;
    //void* buffer_ptr                    = NULL;
    //void* data_ptr                      = NULL;
    //GpuRwQueueEntry* rw_queue_entry   = NULL;
    
    //ssize_t return_size;

    //GPU_PRINT("%s - req type: %s - gpu addr %lx\n", __func__, enum_rw_req[__req_type], (uint64_t)gpu_addr);
    
    // Get a valid queue entry to submit request
#if GET_RW_QUEUE_ENTRY_ALTER 
    rw_queue_index = gpu_rw_manager->getRWQueueEntry2();
#else
    rw_queue_index = gpu_rw_manager->getRWQueueEntry();
#endif
    assert(rw_queue_index != 0xFFFFFFFF);
    //rw_queue_entry = (GpuRwQueueEntry*)&gpu_rw_queue->entries[rw_queue_index];

    // Call remote read API
    // return_size = remote_read_write(PAGE_SIZE, 0/*buffer offset*/, 0 /*file offset*/, req_type, gpu_addr, rw_queue_index, bid);
    gpu_rw_queue->entries[rw_queue_index].type = req_type;
    gpu_rw_queue->entries[rw_queue_index].u.LoggingReq.arrayIdx = arrayIdx;
    gpu_rw_queue->entries[rw_queue_index].u.LoggingReq.pageIdx = pageIdx;
    gpu_rw_queue->entries[rw_queue_index].u.LoggingReq.actionType = actionType;
    gpu_rw_queue->entries[rw_queue_index].status = GPU_RW_PENDING;

    __threadfence_system();
    
    WAIT_COND(gpu_rw_queue->entries[rw_queue_index].status, GPU_RW_READY, ns);
    
    // Release the lock the entry
    gpu_rw_manager->putRWQueueEntry(rw_queue_index);
    
    //GPU_PRINT("%s - req type: %s - gpu addr %lx\n access done.", __func__, enum_rw_req[__req_type], (uint64_t)gpu_addr);
}

__forceinline__ __device__
void loggingRtInfo(uint64_t thread_count)
{
    uint8_t ns                          = 8;   
    uint32_t req_type                   = GPU_RW_ARBITRARY_LOGGING;
    int32_t rw_queue_index              = 0xFFFFFFFF;

#if GET_RW_QUEUE_ENTRY_ALTER 
    rw_queue_index = gpu_rw_manager->getRWQueueEntry2();
#else
    rw_queue_index = gpu_rw_manager->getRWQueueEntry();
#endif
    assert(rw_queue_index != 0xFFFFFFFF);

    // Call remote read API
    //printf("%s: %lu\n", __func__, thread_count);
    gpu_rw_queue->entries[rw_queue_index].type = req_type;
    gpu_rw_queue->entries[rw_queue_index].u.ArbitraryLoggingReq.threadCount = thread_count;

    __threadfence_system();
    
    gpu_rw_queue->entries[rw_queue_index].status = GPU_RW_PENDING;
    __threadfence_system();
    

    WAIT_COND(gpu_rw_queue->entries[rw_queue_index].status, GPU_RW_READY, ns);
    
    // Release the lock the entry
    gpu_rw_manager->putRWQueueEntry(rw_queue_index);
}

__forceinline__ __device__
void sampleMemAccess(uint64_t arrayIdx, uint64_t pageIdx, uint64_t virt_timestamp_dist)
{
    uint8_t ns                          = 8;   
    uint32_t req_type                   = GPU_RW_SAMPLE_MEM;
    int32_t rw_queue_index              = 0xFFFFFFFF;
    
    // Get a valid queue entry to submit request
#if GET_RW_QUEUE_ENTRY_ALTER 
    rw_queue_index = gpu_rw_manager->getRWQueueEntry2();
#else
    rw_queue_index = gpu_rw_manager->getRWQueueEntry();
#endif
    assert(rw_queue_index != 0xFFFFFFFF);
    //rw_queue_entry = (GpuRwQueueEntry*)&gpu_rw_queue->entries[rw_queue_index];

    // Call remote read API
    gpu_rw_queue->entries[rw_queue_index].type = req_type;
    gpu_rw_queue->entries[rw_queue_index].key = (arrayIdx << 32) | pageIdx;
    gpu_rw_queue->entries[rw_queue_index].u.SampleMemReq.virt_timestamp_dist = virt_timestamp_dist;
    __threadfence_system();
    
    gpu_rw_queue->entries[rw_queue_index].status = GPU_RW_PENDING;
    __threadfence_system();
    
    WAIT_COND(gpu_rw_queue->entries[rw_queue_index].status, GPU_RW_READY, ns);
    //WAIT_COND(gpu_rw_queue->entries[rw_queue_index].status, GPU_RW_PROCESSING, ns);
    
    // Release the lock the entry
    gpu_rw_manager->putRWQueueEntry(rw_queue_index);
}

__device__ 
void notifyHostForSamples(uint64_t tail)
{
    uint8_t ns                          = 8;   
    uint32_t req_type                   = GPU_RW_SAMPLE_MEM;
    int32_t rw_queue_index              = 0xFFFFFFFF;
    
    // Get a valid queue entry to submit request
#if GET_RW_QUEUE_ENTRY_ALTER 
    rw_queue_index = gpu_rw_manager->getRWQueueEntry2();
#else
    rw_queue_index = gpu_rw_manager->getRWQueueEntry();
#endif
    assert(rw_queue_index != 0xFFFFFFFF);
    //rw_queue_entry = (GpuRwQueueEntry*)&gpu_rw_queue->entries[rw_queue_index];

    // Call remote read API
    gpu_rw_queue->entries[rw_queue_index].type = req_type;
    //gpu_rw_queue->entries[rw_queue_index].key = (arrayIdx << 32) | pageIdx;
    gpu_rw_queue->entries[rw_queue_index].key = tail;
    //gpu_rw_queue->entries[rw_queue_index].u.SampleMemReq.virt_timestamp_dist = virt_timestamp_dist;
    __threadfence_system();
    
    gpu_rw_queue->entries[rw_queue_index].status = GPU_RW_PENDING;
    __threadfence_system();
    
    WAIT_COND(gpu_rw_queue->entries[rw_queue_index].status, GPU_RW_READY, ns);
    //WAIT_COND(gpu_rw_queue->entries[rw_queue_index].status, GPU_RW_PROCESSING, ns);
    
    // Release the lock the entry
    gpu_rw_manager->putRWQueueEntry(rw_queue_index);

}

__device__
void testCallback(uint64_t key)
{
    //enqueue_mem_samples(nullptr, 0, 0, &notifyHostForSamples);
}

//__forceinline__ __device__
//void send

/* This function is used to estimate reuse distance based on virtual timestamp distance and 
   the assumption that virt_timestamp_dist = a*reuse_distance + b */
//__forceinline__ __device__
__device__
uint64_t estimateReuseDistance(uint64_t virt_timestamp_dist)
{
    uint64_t ret;
    
    if (virt_timestamp_dist < MID_PREDICTOR_THRESHOLD) {
        ret = virt_timestamp_dist;
    }
    //else if (linear_reg_info[get_linear_model_index()].slope < 0) {
    //    ret = virt_timestamp_dist;
    //}
    else {
        //ret = (uint64_t)(((float)virt_timestamp_dist - linear_reg_info->offset) / linear_reg_info->slope);
        //print_linear_model();
        ret = estimate_reuse_distance(virt_timestamp_dist);
    }

    return ret;
}

//__forceinline__ __device__
__device__
uint32_t which_tier_to_evict(PageReuseTable* prt, uint64_t curr_virt_timestamp, bool& reuse_place_decision, bool dirty = false)
{
    /*uint32_t reuse_history[4][8];
    uint32_t reuse_history_index;
    uint32_t last_predicted_tier; // initial value is zero
    uint64_t last_eviction_virt_timestamp;*/
    uint32_t ret = Tier3_SSD;
    uint32_t max_one = 0;
    uint32_t tiers_count[32];

    //return Tier2_CPU;
    //if (dirty && !last_iteration) return Tier2_CPU;
    //else Tier3_SSD;

    #if USE_RAND_POLICY
    if (last_iteration) return Tier3_SSD;
    return (((uint32_t)curand(curand_state) % 2) + 2);
    #endif

    //prt->tmp_count++;
    //printf("reuse dist %lu\n", prt->estimated_remaining_reuse_dist);
    if (prt->last_predicted_tier == 0) {
        if (dirty) {
            #if GET_GOLDEN_REUSE_DISTANCE
            prt->tier_prediction_reason = FIRST_EVICTION_DIRTY;
            #endif
            //return Tier3_SSD;
            return Tier2_CPU;
        }
        else {
            #if GET_GOLDEN_REUSE_DISTANCE
            prt->tier_prediction_reason = FIRST_EVICTION_CLEAN;
            #endif
            //return Tier3_SSD;
            //return Tier2_CPU;
            return (((uint32_t)curand(curand_state) % 2) + 2);
        }
    }
    
    // TODO: temp..
    //if (dirty && prt->tmp_count < 4) {
    
    /*if (dirty && !last_iteration) {
        #if GET_GOLDEN_REUSE_DISTANCE
        prt->tier_prediction_reason = NOT_LAST_AND_DIRTY;
        #endif
        return Tier2_CPU;
    }
    */
    if (last_iteration) {
        #if GET_GOLDEN_REUSE_DISTANCE
        prt->tier_prediction_reason = LAST_ITERATION;
        #endif
        return Tier3_SSD;
    }

    // CHIA-HAO: try?
    ///*
    if (prt->evict_attempt_num >= 1 || prt->under_mem_pressure) {
        prt->evict_attempt_num = 0;   
        #if GET_GOLDEN_REUSE_DISTANCE
        prt->tier_prediction_reason = WRONG_PREDICTION_SO_TO_TIER2;
        #endif
        prt->under_mem_pressure++;
        return Tier2_CPU;
        //return Tier3_SSD;
    }
    //*/
    
    reuse_place_decision = true;
    if (prt->evicted_before == 1) {
        #if GET_GOLDEN_REUSE_DISTANCE
        prt->tier_prediction_reason = SECOND_EVICTION_FOLLOW_LAST_PRED;
        #endif
        return prt->last_predicted_tier;
    }

    /*
    #pragma unroll
    for (uint32_t i = 0; i < REUSE_HISTORY_LEN; i++) {
        if (prt->reuse_history[prt->last_predicted_tier][i] < 32) 
            tiers_count[prt->reuse_history[prt->last_predicted_tier][i]]++;         
    }
    */
    ///*
    #pragma unroll
    for (uint32_t t = 1; t <= TiersNum; t++) {
        tiers_count[t-1] = prt->reuse_history[prt->last_predicted_tier][t];         
    }
    //*/
    
    for (uint32_t i = 0; i < 3; i++) {
        if (max_one < tiers_count[i]) {
            max_one = tiers_count[i];
            ret = i+1;
        }
    }
    if (max_one == 0) {
        #if GET_GOLDEN_REUSE_DISTANCE
        prt->tier_prediction_reason = NO_HISTORY_USE_LAST_PRED;
        #endif
        return prt->last_predicted_tier; 
    }

    reuse_place_decision = true;

#if GET_GOLDEN_REUSE_DISTANCE
    prt->tier_prediction_reason = TRANSITION_PRED;
#endif

    //if (ret == Tier1_GPU) return Tier2_CPU; 
    return ret;
}

//__forceinline__ __device__
__device__
void update_timestamp_upon_eviction(PageReuseTable* prt, uint64_t curr_virt_timestamp)
{
    prt->last_eviction_virt_timestamp = curr_virt_timestamp;
    prt->evicted_before += 1;
}

//__forceinline__ __device__
__device__
void update_timestamp_upon_re_reference(PageReuseTable* prt, uint64_t curr_virt_timestamp)
{
    uint64_t virt_timestamp_dist = curr_virt_timestamp - prt->last_eviction_virt_timestamp;
    uint64_t remaining_reuse_dist = estimateReuseDistance(virt_timestamp_dist);
    uint32_t curr_predicted_tier = getTierByReuseDistance(remaining_reuse_dist);

    //printf("curr %lu - last %lu (virt dist %lu)\n", curr_virt_timestamp, prt->last_eviction_virt_timestamp, remaining_reuse_dist);
    prt->reuse_history[prt->last_predicted_tier][curr_predicted_tier]++;
    prt->last_predicted_tier = curr_predicted_tier;

    prt->estimated_remaining_reuse_dist = remaining_reuse_dist;
#if GET_GOLDEN_REUSE_DISTANCE
    prt->remaining_virt_timestamp_dist = virt_timestamp_dist;
#endif 

    // Reset this: 090423
    prt->evict_attempt_num = 0;   
    prt->under_mem_pressure = 0;
    prt->thrashing_count++;
}

__forceinline__ __device__
void update_tier_bins(uint32_t tier) 
{
    tier_bins_d->bins[tier].fetch_add(1, simt::memory_order_acq_rel);
    uint64_t pos_head = teq->tier_eviction_queue_head.fetch_add(1, simt::memory_order_acq_rel);
    pos_head %= TIER_EVICTION_QUEUE_LENGTH;
    tier_eviction_queue[pos_head] = (uint8_t)tier;
    uint64_t pos_tail;
    ///*
    if (pos_head >= TIER_EVICTION_QUEUE_LENGTH-2) {
        pos_tail = teq->tier_eviction_queue_tail.fetch_add(1, simt::memory_order_acq_rel);
        pos_tail %= TIER_EVICTION_QUEUE_LENGTH;
        tier_bins_d->bins[tier_eviction_queue[pos_tail]]--;
    }
    //*/
    //teq->total_evictions.fetch_add(1, simt::memory_order_acq_rel);
}

__device__
bool page_stealing()
{
    //if (teq->tier_eviction_queue_head.load() > second_tier_mem_pages/4 && tier_bins_d->bins[3]>tier_bins_d->bins[2]*4) {
    if (tier_bins_d->bins[3] > second_tier_mem_pages/4 && tier_bins_d->bins[3]>tier_bins_d->bins[2]*8) {
        return true;
    }
    return false;
}


void HostCache::loggingOnHost(uint64_t arrayIdx, uint64_t pageIdx, uint32_t actionType)
{
    //if (actionType == PAGE_ACTION_EVICT)
        //std::cout << "Array " << arrayIdx << " - " << "Page " << pageIdx << " : " << enum_page_action_h[actionType] << "\n";
    //unordered_map<uint64_t, map<uint64_t, vector<PageActInfo>>> fetchEvictMap; // arrayIdx, pageIdx, seq of pageActInfo

    //if (fetchEvictMap.find(arrayIdx) != fetchEvictMap.end() && 
    //    fetchEvictMap[arrayIdx].find(pageIdx) != fetchEvictMap[arrayIdx].end() &&
    //    fetchEvictMap[arrayIdx][pageIdx].size() > 0) {
    //    std::cout << "Array " << arrayIdx << " - " << "Page " << pageIdx << " : " << enum_page_action_h[actionType] << "\n";
    //}
    ///*
    uint64_t key = (arrayIdx << 32) | pageIdx;
    //int64_t pos;
    //uint64_t dist;
    std::stringstream ss;
    if (actionType == PAGE_ACTION_EVICT) ss << "Ev " << std::dec << key << std::endl;
    else ss << std::dec << key << std::endl;
    std::cout << ss.str();
    //std::cout << std::dec << key << std::endl;
    /*
    if (actionType == PAGE_ACTION_ACCESS || actionType == PAGE_ACTION_FETCH) {
        //if (find_reuse_page(key, pos)) {
        logging_mtx.lock();
        if (reuse_stack_map.find(key) != reuse_stack_map.end()) {
            if (reuse_stack_map[key] == reuse_stack_size-1) {
                reuseDistTable[0] += 1;
            }
            else {
                dist = reuse_stack_size-reuse_stack_map[key]-1;
                reuseDistTable[dist] += 1;
                reuse_stack_size++;
            }
            //std::cerr << dist << ":" << reuseDistTable[dist] << std::endl;
        }
        else {
            //reuse_stack[reuse_stack_size] = key;
            reuse_stack_map[key] = reuse_stack_size;
            reuse_stack_size++;
        }
        logging_mtx.unlock();
    }
    */
    //fetchEvictMap[arrayIdx][pageIdx].push_back({std::time(0), actionType});
    
}


void HostCache::flushLog()
{
    std::fstream fs(appName+".pagelog", std::fstream::out);
    for (auto& p : fetchEvictMap) {
        uint64_t arrayIdx = p.first;
        auto& pageMap = p.second;
        std::cerr << "flushing array " << p.first << "..."<< std::endl;
        fs << "array " << arrayIdx << "\n";
        for (auto& m : pageMap) {
            auto pageIdx = m.first;
            auto& page_access_info = m.second;
            fs << "page " << pageIdx << " " << page_access_info.size() << "\n";
            //std::cerr << "page " << pageIdx << " " << page_access_info.size() << "\n";
            for (auto& info : page_access_info) {
                fs << info.timestamp << " " << info.actionType << "\n";
            }
        }
    }
    fs.close();
}

void HostCache::flushReuseDistTable()
{
    std::cout << __func__ << std::endl;
    std::fstream fs(appName+".reuse_dist", std::fstream::out);
    for (auto& p : reuseDistTable) {
        fs << p.first << ":" << p.second << std::endl;
    }
    fs.close();
}

void HostCache::buildProfileInfo()
{
    //profile_info_vev
    //void* test = NULL;
    size_t alloc_size;
    uint64_t* fetch_count_h = NULL;
    PageProfileInfo profile_info;
    for (auto& p : fetchEvictMap) {
        profile_info.page_info.range = p.second.rbegin()->first + 1; // largest index
        alloc_size = profile_info.page_info.range*sizeof(uint64_t);
        std::cerr << "array idx " << p.first << " - range " << p.second.rbegin()->first + 1 << " - alloc size " << alloc_size << std::endl;
        //profile_info.page_info.fetch_count = (uint64_t*)createBuffer(alloc_size, 0).get(); // gpu pointer
        CUDA_SAFE_CALL(cudaMalloc((void**)&profile_info.page_info.fetch_count, alloc_size));
        fetch_count_h = (uint64_t*)malloc(alloc_size); // host pointer
        if (!fetch_count_h) {
            std::cerr << "malloc for fetch_count_h failed." << std::endl;
            exit(1);
        }
        for (auto& info : p.second) {
            auto page_index = info.first;
            auto fetch_count = info.second.size();
            fetch_count_h[page_index] = fetch_count;
        }
        CUDA_SAFE_CALL(cudaMemcpy((void*)profile_info.page_info.fetch_count, (void*)fetch_count_h, alloc_size, cudaMemcpyHostToDevice));
        //profile_info.page_info_d = (PageInfo_d*)createBuffer(sizeof(PageInfo_d), 0).get();
        CUDA_SAFE_CALL(cudaMalloc((void**)&(profile_info.page_info_d), sizeof(PageInfo_d)));
        //CUDA_SAFE_CALL(cudaMalloc((void**)&test, sizeof(PageInfo_d)));
        fprintf(stderr, "pgae_info_d %p, fetch_count %p\n", profile_info.page_info_d, profile_info.page_info.fetch_count);
        CUDA_SAFE_CALL(cudaMemcpy((void*)profile_info.page_info_d, (void*)&profile_info.page_info, sizeof(PageInfo_d), cudaMemcpyHostToDevice));
        profile_info_vec.push_back(profile_info);
        free(fetch_count_h);
    }
}

///*
void HostCache::releaseDevMem()
{
    ///*
    for (auto info : profile_info_vec) {
        cudaFree(info.page_info.fetch_count);
        cudaFree(info.page_info_d);
    }
    //*/
}
//*/

void HostCache::readProfiledLog()
{
    std::fstream fs(appName+".pagelog", std::fstream::in);
    string line;
    fetchEvictMap.clear();
    if (fs.is_open()) {
        while (!fs.eof()) {
            uint64_t arrayIdx;
            uint64_t pageIdx;
            uint64_t timestamp;
            uint64_t actionType;
            size_t blank_pos;
            getline(fs, line, '\n');
            if (line.empty() && fs.eof()) break;
            /* Array Index */
            if (line.find("array") != string::npos) {
                blank_pos = line.find(' ');
                assert(line.substr(0, blank_pos) == "array");
                arrayIdx = (unsigned long long)stol(line.substr(blank_pos+1));
                getline(fs, line, '\n');
            }
            
            //if (line.find("page") != string::npos) {
            //    std::cerr << "eof? " << fs.eof() << std::endl;
            //}

            /* Page Index */
            blank_pos = line.find(' ');
            auto blank_pos_2 = line.find(' ', blank_pos+1);
            assert(line.substr(0, blank_pos) == "page");
            pageIdx = stol(line.substr(blank_pos+1, blank_pos_2-blank_pos-1));
            auto size = stol(line.substr(blank_pos_2+1));
            
            //std::cerr << __func__ << ":" << pageIdx << ", size " << size << std::endl;
            /* Page Act Info */
            for (int64_t i = 0; i < size; i++) {
                getline(fs, line, '\n');
                blank_pos = line.find(' ');
                timestamp = stol(line.substr(0, blank_pos));
                actionType = stol(line.substr(blank_pos+1));
                // TODO: test
                //assert(fetchEvictMap[arrayIdx][pageIdx][i].timestamp == timestamp);
                //assert(fetchEvictMap[arrayIdx][pageIdx][i].actionType == actionType);
                fetchEvictMap[arrayIdx][pageIdx].push_back({timestamp, actionType});
            }
        }
        std::cerr << "Validation Done...\n";
        fs.close();
        buildProfileInfo();
    }
    else {
        std::cerr << "Cannot open " << appName+".pagelog" << std::endl;
    }
}

cudaError_t err;
CUresult res;
CUdevice dev;
CUcontext ctx;
CUdevice_attribute attrib = CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING;
char* cu_res_string;
#define PRINT_CU_ERROR(exec_name) \
do {\
    cuGetErrorString(res, (const char**)&cu_res_string);\
    fprintf(stderr, "%s error : %s\n", exec_name, cu_res_string);\
} while(0)

#define FlushGPUDirect()\
do {\
   res = cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,\
                                    /*CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES);*/\
                                    0);\
   PRINT_CU_ERROR("cuFlushGPUDirectRDMAWrites");\
} while(0)

void* dummy_data = NULL;

/*
void cu_context_init()
{
    res = cuDeviceGet(&dev, 0);
    PRINT_CU_ERROR("cuDeviceGet");

    res = cuCtxCreate(&ctx, CU_CTX_SCHED_BLOCKING_SYNC, dev);
    PRINT_CU_ERROR("cuCtxCreate");
    
    int pi;
    res = cuDeviceGetAttribute(&pi, attrib, dev);
    PRINT_CU_ERROR("cuDeviceGetAttribute");
    fprintf(stderr, "RDMA ordering : %d\n", pi);

    //attrib = cudaDevAttrCanFlushRemoteWrites;
    err = cudaDeviceGetAttribute(&pi, cudaDevAttrCanFlushRemoteWrites, 0);
    //PRINT_CU_ERROR("cuDeviceGetAttribute");
    if (err == cudaSuccess)
        fprintf(stderr, "Can Flush Remote Writes : %d\n", pi);
    

    err = cudaDeviceGetAttribute(&pi, cudaDevAttrGPUDirectRDMASupported, 0);
    if (err == cudaSuccess)
        fprintf(stderr, "GPUDirect RDMA Supported? : %d\n", pi);

    err = cudaDeviceGetAttribute(&pi, cudaDevAttrGPUDirectRDMAFlushWritesOptions, 0);
    if (err == cudaSuccess)
        fprintf(stderr, "GPUDirect RDMA Flush Writes Options : %d\n", pi);

    err = cudaDeviceGetAttribute(&pi, cudaDevAttrGPUDirectRDMAWritesOrdering, 0);
    if (err == cudaSuccess)
        fprintf(stderr, "GPUDirect RDMA Writes Ordering : %d\n", pi);




    //res = cuCtxGetDevice(&dev);
    //PRINT_CU_ERROR("cuCtxCreate");

    //res = cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX, 
    //        CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
    //CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES);

    //cuCtxDestroy(ctx);
}
*/

//void spin_init()
//{
//    spin_fd = open("/dev/spindrv", O_RDWR);
//    if (spin_fd < 0) {
//        fprintf(stderr, "spindrv open failed %s\n", strerror(errno));
//        exit(-1);
//    }
//    myinit(spin_fd);
//}

ssize_t cpu_read(int fd, void* gpu_addr, size_t size, off_t file_offset, cudaStream_t* s, void* buf) 
{
    //void* buf = memalign(4096, size);
    //void* buf;
    //cudaHostAlloc(&buf, size, cudaHostAllocDefault);
    if (buf == NULL) {
        fprintf(stderr, "%s: buf allocation failed\n", __func__);
        exit(-1);
    }

    ssize_t read_size = 0;
    if (!preload) {
        //lseek(fd, file_offset, SEEK_SET);
        //read_size = read(fd, buf, size);
        read_size = pread(fd, buf, size, file_offset);
        fprintf(stderr, "posix read file %ld (fd %d, data %llx, size %ld, offset %ld)\n", read_size, fd, (u64)gpu_addr, size, file_offset);
    }

    if (read_size < 0) {
        fprintf(stderr, "posix read file failed %d (fd %d, data %llx, size %ld)\n", errno, fd, (u64)gpu_addr, size);
        perror("Error printed by perror"); 
        exit(-1);
    }
    
    //fprintf(stderr, "copying (read) data %d (%llx, %lu)\n", fd, (u64)gpu_addr, size);
    //cudaError_t status;
    //cudaEvent_t event;
    //cudaEventCreate(&event);
    //cudaStreamCreate(s);
    cudaMemcpyAsync(gpu_addr, buf, size, cudaMemcpyHostToDevice, *s);
    //cudaEventRecord(event, *s);
    cudaStreamSynchronize(*s);
    //cudaStreamWaitEvent(*s, event);
    //status = cudaEventQuery(event);
    //fprintf(stderr, "%s: Status %s\n", __func__, cudaGetErrorString(status));
    //while (status != cudaSuccess) {
    //    cudaEventQuery(event);
    //}
    //for (int i = 0; i < size; i++) {
    //    fprintf(stderr, "%c", ((char*)buf)[i]);
    //}
    //fprintf(stderr, "\n");

    //cudaFreeHost(buf);
    CPU_PRINT("cpu read for %d (%llx, %lu)\n", fd, (u64)gpu_addr, size);
    
    return read_size;
}

ssize_t cpu_write(int fd, void* gpu_addr, size_t size, cudaStream_t* s, void* buf) 
{
    //void* buf;
    //cudaHostAlloc(&buf, size, cudaHostAllocDefault);
    if (buf == NULL) {
        fprintf(stderr, "%s: buf allocation failed\n", __func__);
        exit(-1);
    }

    cudaMemcpyAsync(buf, gpu_addr, size, cudaMemcpyDeviceToHost, *s);
    cudaStreamSynchronize(*s);

    fprintf(stderr, "copying (write) data %d (%llx, %lu)\n", fd, (u64)gpu_addr, size);

    ssize_t write_size = write(fd, buf, size);
    if (write_size <= 0) {
        perror("Error printed by perror"); 
        exit(-1);
    }
    
    //cudaFreeHost(buf);
    fprintf(stderr, "cpu write for %d (%llx, %lu)\n", fd, (u64)gpu_addr, size);
  
    return write_size;
}

__global__ void verify (void* ptr) 
{
    page_info_dev = (PageProfileInfo*)ptr;
    printf("page_profile_info %lx\n", (uint64_t)page_info_dev);
}

HostCache::HostCache(Controller* _ctrl, uint64_t _gpu_mem_size = 0) : ctrl(_ctrl), gpu_mem_size(_gpu_mem_size)
//HostCache::HostCache()
{
    //INIT_SHARED_MEM_PTR(GpuOpenFileTable, this->host_open_table, gpu_open_table);
    INIT_SHARED_MEM_PTR(GpuFileRwManager, this->host_rw_manager, gpu_rw_manager);
    INIT_SHARED_MEM_PTR(GpuFileRwManager, this->host_rw_manager_mem, gpu_rw_manager_mem);
    INIT_SHARED_MEM_PTR(GpuRwQueue, this->host_rw_queue, gpu_rw_queue);
    INIT_SHARED_MEM_PTR(GpuRwQueue, this->host_rw_queue_mem, gpu_rw_queue_mem);
    INIT_SHARED_MEM_PTR(HostCacheRuntimeState, this->hc_runtime_state, gpu_hc_runtime_state);
    //for (int i = 0; i < GPU_RW_SIZE; i++) {
    //    INIT_SHARED_MEM_PTR(GPU_RW_QUEUE_ENTRY, this->host_rw_queue->entries[i], gpu_rw_queue->entries[i]);
    //}

    //__device__ PageProfileInfo* page_info_dev;
    //INIT_DEVICE_MEM(PageProfileInfo, dev_ptr, device_symbol);
    
    // Open file table initialization
    //this->host_open_table->init();

    // File read/write manager initialization
    this->host_rw_manager->init(this->host_rw_queue);
    this->host_rw_manager_mem->init(this->host_rw_queue_mem);

    // Read/write queue initialization
    this->host_rw_queue->init();
    this->host_rw_queue_mem->init();
    
    fprintf(stderr, "create host buffer for host cache\n");
    // Host buffers
    for (uint32_t i = 0; i < GPU_RW_SIZE; i++) {
        // TODO: Fix size 2m now
        cudaHostAlloc(&(this->host_buffer[i]), PAGE_SIZE, cudaHostAllocDefault);
        if (this->host_buffer[i] == NULL) {
            fprintf(stderr, "Host Allocate failed %d.", i);
            exit(-1);
        }
    }
    this->num_pages = HOST_MEM_SIZE / PAGE_SIZE;
    this->num_ctrl_pages_in_one_line = PAGE_SIZE / _ctrl->ctrl->page_size;

#if USER_SPACE_ZERO_COPY == 0
    // CHIA-HAO: call dma functions to get physical address
    posix_memalign((void**)&host_mem, PAGE_SIZE, HOST_MEM_SIZE);
    
    int ret = mlock(host_mem, (size_t)HOST_MEM_SIZE);
    if (ret != 0) {
        fprintf(stderr, "mlock for host_mem failed...(%s)\n", strerror(errno));
        exit(1);
    }
    madvise(host_mem, PAGE_SIZE, MADV_HUGEPAGE);
    printf("host mem base addr %lx\n", host_mem);
    //CUDA_SAFE_CALL(cudaHostRegister(host_mem, HOST_MEM_SIZE, cudaHostRegisterPortable));
    CUDA_SAFE_CALL(cudaHostRegister(host_mem, HOST_MEM_SIZE, cudaHostRegisterDefault));
    

#else
    ///*
    useHugePage = true;

    fprintf(stderr, "create dma for host cache\n");
    this->host_dma = createDma((const nvm_ctrl_t*)_ctrl->ctrl, (size_t)HOST_MEM_SIZE);
    host_mem = host_dma->vaddr;

    int ret = mlock(host_mem, (size_t)HOST_MEM_SIZE);
    if (ret != 0) {
        fprintf(stderr, "mlock for host_mem failed...\n");
        exit(1);
    }
    memset(host_mem, 0, (size_t)HOST_MEM_SIZE);
    printf("host cache memory vaddr %llx (#pages %llu)\n", (uint64_t)host_mem, host_dma->n_ioaddrs);
    
    // Fill in prp1 and prp2
    //useHugePage = true;
    ///*
    this->prp1 = new uint64_t[this->num_pages];
    this->prp2 = new uint64_t[this->num_pages];
    this->prp2_dma = createDma((const nvm_ctrl_t*)_ctrl->ctrl, (size_t)this->num_pages * _ctrl->ctrl->page_size);

    ret = mlock(prp2_dma->vaddr, (size_t)this->num_pages * _ctrl->ctrl->page_size);
    if (ret != 0) {
        fprintf(stderr, "mlock for prp2 failed...\n");
        exit(1);
    }
    const uint32_t uints_per_page = _ctrl->ctrl->page_size / sizeof(uint64_t);
    for (uint32_t i = 0; i < this->num_pages; i++) {
        //printf("page[%u]: %llx\n", i, host_dma->ioaddrs[i*this->num_ctrl_pages_in_one_line]);
        this->prp1[i] = host_dma->ioaddrs[i*this->num_ctrl_pages_in_one_line];
        this->prp2[i] = prp2_dma->ioaddrs[i];
        for (uint32_t j = 1; j < this->num_ctrl_pages_in_one_line; j++) {
            //printf("page[%u][%u]: %llx (phys %llx)\n", i, j, host_dma->ioaddrs[i*this->num_ctrl_pages_in_one_line+j], );
            ((uint64_t*)prp2_dma->vaddr)[i*uints_per_page + j-1] = host_dma->ioaddrs[i*this->num_ctrl_pages_in_one_line + j]; 
            //printf("page[%u][%u]: %llx (phys %llx)\n", i, j, host_dma->ioaddrs[i*this->num_ctrl_pages_in_one_line+j], ((uint64_t*)prp2_dma->vaddr)[i*uints_per_page + j-1]);
        }
        //printf("---------------------------\n");
    }
    printf("prp2 memory vaddr %llx (#pages %llu)\n", (uint64_t)this->prp2, prp2_dma->n_ioaddrs);
    //*/
    CUDA_SAFE_CALL(cudaHostRegister(host_mem, HOST_MEM_SIZE, cudaHostRegisterPortable));
    //CUDA_SAFE_CALL(cudaHostRegister(host_mem, HOST_MEM_SIZE, cudaHostRegisterDefault));
    
    this->q_head = new simt::atomic<uint64_t, simt::thread_scope_device>();
    this->q_tail = new simt::atomic<uint64_t, simt::thread_scope_device>();
#endif
    
    /* Copy # pages info to GPU */
    uint64_t gpu_mem_size_pages = gpu_mem_size / PAGE_SIZE;
    uint64_t cpu_mem_size_pages = HOST_MEM_NUM_PAGES;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(first_tier_mem_pages, &gpu_mem_size_pages, sizeof(uint64_t)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(second_tier_mem_pages, &cpu_mem_size_pages, sizeof(uint64_t)));

    /**/
    printf("cpu mem size pages %u\n", cpu_mem_size_pages);
    CUDA_SAFE_CALL(cudaMallocHost((void**)(&num_idle_slots_h), sizeof(uint64_t)));
    //CUDA_SAFE_CALL(cudaMemset((void*)(num_idle_slots_h), cpu_mem_size_pages, sizeof(uint64_t)));
    memcpy((void*)(num_idle_slots_h), (void*)&cpu_mem_size_pages, sizeof(uint64_t));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(num_idle_slots, &num_idle_slots_h, sizeof(uint64_t*)));
  
    /* For random generator */
    //#if USE_RAND_POLICY
    CUDA_SAFE_CALL(cudaMalloc((void**)(&curand_state_h), sizeof(curandState)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(curand_state, &curand_state_h, sizeof(curandState*)));
    init_rand_generator<<<1,1>>>();
    //#endif

    /* Linear Regression Info */
    CUDA_SAFE_CALL(cudaMallocHost((void**)(&linear_reg_info_idx_h), sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset((void*)(linear_reg_info_idx_h), 0, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(linear_reg_info_idx, &linear_reg_info_idx_h, sizeof(uint32_t*)));

    #if DYNAMIC_SAMPLE_FREQ
    uint32_t default_freq = SAMPLE_FREQ;
    CUDA_SAFE_CALL(cudaMallocHost((void**)(&dynamic_sample_freq_h), sizeof(uint32_t)));
    //CUDA_SAFE_CALL(cudaMemset((void*)(dynamic_sample_freq_h), 4, sizeof(uint32_t)));
    memcpy((void*)(dynamic_sample_freq_h), (void*)&default_freq, sizeof(uint32_t));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dynamic_sample_freq, &dynamic_sample_freq_h, sizeof(uint32_t*)));
    #endif

    CUDA_SAFE_CALL(cudaMallocHost((void**)(&linear_reg_info_h), sizeof(LinearRegInfo)*2)); // double buffer
    linear_reg_info_h[0].slope = 1000.0;
    linear_reg_info_h[0].offset = 0.0;
    linear_reg_info_h[1].slope = 1000.0;
    linear_reg_info_h[1].offset = 0.0;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(linear_reg_info, &linear_reg_info_h, sizeof(LinearRegInfo*)));
    
    /* Tier Bins */
    TierBins tier_bins_local;
    tier_bins_local.threshold = gpu_mem_size_pages + cpu_mem_size_pages;
    CUDA_SAFE_CALL(cudaMalloc((void**)(&tier_bins_h), sizeof(TierBins)));
    CUDA_SAFE_CALL(cudaMemcpy((void*)tier_bins_h, (void*)&tier_bins_local, sizeof(TierBins), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(tier_bins_d, &tier_bins_h, sizeof(TierBins*)));

    /* Tier Eviction Queue */
    CUDA_SAFE_CALL(cudaMalloc((void**)(&tier_eviction_queue_h), sizeof(uint8_t)*TIER_EVICTION_QUEUE_LENGTH));
    CUDA_SAFE_CALL(cudaMemset((void*)tier_eviction_queue_h, 0, sizeof(uint8_t)*TIER_EVICTION_QUEUE_LENGTH));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(tier_eviction_queue, &tier_eviction_queue_h, sizeof(uint8_t*)));
    CUDA_SAFE_CALL(cudaMalloc((void**)(&teq_h), sizeof(TierEvictionQueue)));
    CUDA_SAFE_CALL(cudaMemset((void*)teq_h, 0, sizeof(TierEvictionQueue)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(teq, &teq_h, sizeof(TierEvictionQueue*)));



    /* Circular buffer */
    host_ring_buffer_t *ring_buf_h = new host_ring_buffer_t();
    host_ring_buffer_t *ring_buf_d = init_ring_buf_for_dev(MEM_SAMPLES_RING_BUFFER_SIZE, ring_buf_h);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(mem_samples_ring_buf, &ring_buf_d, sizeof(host_ring_buffer_t*)));

    /* Memory Samples Collector */
    mem_sample_collector = new MemSampleCollector(linear_reg_info_h, linear_reg_info_idx_h, ring_buf_h, ring_buf_d);
    mem_sample_collector->start_collector_thread();
    #if GET_GOLDEN_REUSE_DISTANCE 
    golden_mem_collector = new MemSampleCollector(this->host_rw_queue_mem);
    golden_mem_collector->start_rw_queue_thread();
    #endif

    host_cache_state.resize(HOST_MEM_NUM_PAGES);
    bid = 0;

    //#if USE_PINNED_MEM
    {
        void* tmp_ptr = NULL;
        CUDA_SAFE_CALL(cudaHostGetDevicePointer((void**)(&tmp_ptr), (void*)host_mem, 0));\
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(host_cache_base, &tmp_ptr, sizeof(void*)));\
    }
    //#endif
    
    /* Queue Pair for Host Cache */
    #define HOST_QUEUE_NUM_ENTRIES 4096
    //this->qp = new QueuePair(_ctrl->ctrl, _ctrl->ns, _ctrl->info, _ctrl->aq_ref, _ctrl->n_qps+1, HOST_QUEUE_NUM_ENTRIES);
    for (uint32_t i = 0; i < NUM_NVME_QUEUES; i++) {
        this->qp[i] = new QueuePair(_ctrl->ctrl, _ctrl->ns, _ctrl->info, _ctrl->aq_ref, _ctrl->n_qps+i+1, HOST_QUEUE_NUM_ENTRIES);
        //this->sq_host.cid_h = new padded_struct_h();
        this->sq_host[i].cid_h = (padded_struct_h*)malloc(sizeof(padded_struct_h)*65536);
        //this->cq_host.pos_locks_h = new padded_struct_h();
        this->cq_host[i].pos_locks_h = (padded_struct_h*)malloc(sizeof(padded_struct_h)*HOST_QUEUE_NUM_ENTRIES);
        memset(this->cq_host[i].pos_locks_h, 0, sizeof(padded_struct_h)*HOST_QUEUE_NUM_ENTRIES);
    }

    // 
    stream_mngr = new StreamManager();
    appName = "pagerank128M";
    fetch_from_host_count = 0;

    // TODO: Handle Profile... 
#if APPLY_PROFILE
    readProfiledLog();
    assert(fetchEvictMap.size() != 0);
    INIT_DEVICE_MEM_ARRARY(PageProfileInfo, page_info_dev, &(profile_info_vec[0]), fetchEvictMap.size());
    std::cerr << "Init page info dev finished...\n";
#endif


#if TIME_BREAK_DOWN_FOR_PERF_TEST
    perf_test_time_profile_vec.resize(GPU_RW_SIZE);
#endif
    
#if BASIC_TEST
    // Test
    void* gpu_buff = NULL;
    
    //delete mem_sample_collector;
    //delete stream_mngr;

#if USER_SPACE_ZERO_COPY == 0
    cudaMalloc(&gpu_buff, PAGE_SIZE);
    #if MULTI_DATA_STREAM
    CUDA_SAFE_CALL(cudaMemcpyAsync(host_mem, gpu_buff, PAGE_SIZE, cudaMemcpyDeviceToHost, stream_mngr->data_stream_from_device[0]));
    #else
    CUDA_SAFE_CALL(cudaMemcpyAsync(host_mem, gpu_buff, PAGE_SIZE, cudaMemcpyDeviceToHost, stream_mngr->data_stream_from_device));
    #endif
    cudaFree(gpu_buff);
    fprintf(stderr, "cudaMemcpyAsync test for host mem is good.\n");
    // Write data test
    /*
    memset((void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, 0), 'X', PAGE_SIZE);
    write_data_from_hc(this, 0, PAGE_SIZE/512, 0, 0); 
    memset((void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, 1), 'Y', PAGE_SIZE);
    write_data_from_hc(this, 128, PAGE_SIZE/512, 1, 1 % NUM_NVME_QUEUES); 
    memset((void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, 2), 'Z', PAGE_SIZE);
    write_data_from_hc(this, 256, PAGE_SIZE/512, 2, 2 % NUM_NVME_QUEUES); 
    memset((void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, 3), 'A', PAGE_SIZE);
    write_data_from_hc(this, 384, PAGE_SIZE/512, 3, 3 % NUM_NVME_QUEUES); 
    memset((void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, 4), 'B', PAGE_SIZE);
    write_data_from_hc(this, 512, PAGE_SIZE/512, 4, 4 % NUM_NVME_QUEUES); 
    memset((void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, 5), 'C', PAGE_SIZE);
    write_data_from_hc(this, 640, PAGE_SIZE/512, 5, 5 % NUM_NVME_QUEUES); 
    fprintf(stderr, "write_data_from_hc to SSD test for host mem is good.\n");


    read_data_to_hc(this, 0, PAGE_SIZE/512, 6);
    read_data_to_hc(this, 128, PAGE_SIZE/512, 7);
    read_data_to_hc(this, 256, PAGE_SIZE/512, 8);
    read_data_to_hc(this, 384, PAGE_SIZE/512, 9);
    read_data_to_hc(this, 512, PAGE_SIZE/512, 10);
    read_data_to_hc(this, 640, PAGE_SIZE/512, 11);
    for (int i = 0; i < PAGE_SIZE; i++) {
        if (*((unsigned char*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, 6) + i) != 'X' ||
            *((unsigned char*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, 7) + i) != 'Y' ||
            *((unsigned char*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, 8) + i) != 'Z' ||
            *((unsigned char*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, 9) + i) != 'A' ||
            *((unsigned char*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, 10) + i) != 'B' ||
            *((unsigned char*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, 11) + i) != 'C'
            ) {
            fprintf(stderr, "read_data_to_hc failed!\n");
            exit(1);
        }
    }
    printf("\n");
    fprintf(stderr, "read_data_to_hc to SSD test for host mem is good.\n");
    */
#endif

#endif

    
}

HostCache::~HostCache()
{
    std::cerr << "HostCache Destructor...\n";
    //FREE_SHARED_MEM(this->host_open_table);
    //FREE_SHARED_MEM(this->host_rw_manager);
    //FREE_SHARED_MEM(this->host_rw_manager_mem);
    //FREE_SHARED_MEM(this->host_rw_queue);
    //FREE_SHARED_MEM(this->host_rw_queue_mem);
    for (int i = 0; i < GPU_RW_SIZE; i++) {
        //cudaFreeHost(this->host_buffer[i]);
    }
    //cudaFree(linear_reg_info_h);
    //cudaFree(linear_reg_info_idx_h);
    std::cerr << "Valid Entry Count " << valid_entry_count.load() << std::endl;
    std::cout << "Hit Rate of Tier-2: " << (float)(total_hits.load())/(float)(total_fetches.load()) << std::endl;
    std::cout << "Total Fetches to Tier-1: " << total_fetches.load() << std::endl;
    std::cout << "\tTotal Clean Fetches to Tier-1: " << total_clean_fetches.load() << std::endl;
    std::cout << "\tTotal Dirty Fetches to Tier-1: " << total_dirty_fetches.load() << std::endl;
    std::cout << "\tTotal CudaMemcpy Fetches to Tier-1: " << total_cu_fetches.load() << std::endl;
    std::cout << "\tTotal ZeroCopy Fetches to Tier-1: " << total_zc_fetches.load() << std::endl;
    std::cout << "Total Evicts from Tier-1: " << total_tier1_evicts.load() << std::endl;
    std::cout << "\tTotal Clean Evicts from Tier-1: " << total_tier1_clean_evicts.load() << std::endl;
    std::cout << "\tTotal Dirty Evicts from Tier-1: " << total_tier1_dirty_evicts.load() << std::endl;
    std::cout << "Total Accesses: " << total_accesses.load() << std::endl;

    //cudaFreeHost(host_mem);
    cudaHostUnregister(host_mem);
#if USER_SPACE_ZERO_COPY == 0
    free(host_mem);
#endif
    std::cerr << "Fetch from host count " << fetch_from_host_count << "\n";
    std::cerr << "ready to flush log...\n";
    #if PROFILE
    flushLog();
    flushReuseDistTable();
    //readProfiledLog();
    #endif

    for (uint32_t i = 0; i < NUM_NVME_QUEUES; i++) {
        delete this->qp[i];
    }
    delete stream_mngr;

    std::cout << "Clearing Host Cache State...\n";
    host_cache_state.clear();
    //std::cout << "Hit Rate of Tier-2: " << (float)(total_hits.load())/(float)(total_accesses.load()) << std::endl;
    //std::cout << "Total Accesses " << total_accesses.load() << std::endl;
}

//void HostCache::setGDSHandler(GDS_HANDLER* _gds_handler)
//{
//    this->gds_handlers[_gds_handler->getCPUfd()] = _gds_handler;
//}

void HostCache::handleRequest(int queue_index)
{
    uint32_t type             = this->host_rw_queue->entries[queue_index].type;
    uint64_t key              = this->host_rw_queue->entries[queue_index].key;
    #if USE_PINNED_MEM
    bool pin_on_host          = this->host_rw_queue->entries[queue_index].pin_on_host;
    #endif
    //int cpu_fd              = this->host_rw_queue->entries[queue_index].cpu_fd;
    //off_t file_offset       = this->host_rw_queue->entries[queue_index].u.RwReq.file_offset;
    //off_t buffer_offset     = this->host_rw_queue->entries[queue_index].u.RwReq.buffer_offset;
    //size_t size             = this->host_rw_queue->entries[queue_index].u.RwReq.size;
    bool is_dirty             = this->host_rw_queue->entries[queue_index].dirty;
    volatile void* gpu_addr   = this->host_rw_queue->entries[queue_index].gpu_addr;
    //volatile void* buf = this->host_buffer[queue_index];
    
    //cu_context_init();
    CPU_PRINT("%s: type %d (%s) - queue index %d, gpu_addr %lx\n",
        __func__, type, enum_rw_req_h[type], queue_index, (uint64_t)gpu_addr);
    ///*
    #if TIME_BREAK_DOWN_FOR_PERF_TEST
    this->perf_test_time_profile_vec[queue_index].handle_begin = std::chrono::high_resolution_clock::now();
    #endif

    total_accesses.fetch_add(1, std::memory_order_seq_cst);
    //if (type == GPU_RW_ARBITRARY_LOGGING) printf("%lu\n", this->host_rw_queue->entries[queue_index].u.ArbitraryLoggingReq.threadCount);
    if (type != GPU_RW_UNPIN_PAGE && type != GPU_RW_PIN_PAGE) {
        pending_reqs_num.fetch_add((int64_t)1, std::memory_order_acq_rel);
    }

    if (type == GPU_RW_EVICT_TO_HOST) {
        uint64_t _bid_first = bid.load(simt::memory_order_relaxed) % HOST_MEM_NUM_PAGES;
        uint64_t _bid = _bid_first;
        
        //if (pending_reqs_num.load(std::memory_order_acquire) >= 64) {
        if (pending_reqs_num.load(std::memory_order_acquire) >= 256) {
            this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_EVICT_FAILED;
            //fprintf(stderr, "Too many reqs (#reqs %lu)!!!\n", pending_reqs_num.load());
            goto req_ready;
        }

        // TODO
        /* If this cache line is temporarily occupied by dirty data, skip first
           If all the lines are locked, then return failed code */
        do {
            if (((_bid+1) % HOST_MEM_NUM_PAGES) == _bid_first) {
                this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_EVICT_FAILED;
                this->host_rw_queue->entries[queue_index].status = GPU_RW_READY;
                if (type != GPU_RW_PIN_PAGE) {
                    pending_reqs_num.fetch_sub((int64_t)1, std::memory_order_acq_rel);
                }
                __sync_synchronize();
                return;
            }
            _bid = bid.fetch_add(1, simt::memory_order_relaxed) % HOST_MEM_NUM_PAGES;
            //fprintf(stderr, "get bid %lu (bid first %lu)\n", _bid, _bid_first);
        } while (host_cache_state[_bid].state == HOST_CACHE_ENTRY_VALID && (host_cache_state[_bid].is_dirty || IS_PINNED(host_cache_state[_bid])));
        // If dirty, flush to the SSD
        //if (this->host_rw_queue->entries[queue_index].data_dirty) {
            //this->host_rw_queue->entries[queue_index].data_dirty = false;
        //}
        /* Assign location of the cache */
        this->host_rw_queue->entries[queue_index].bid = _bid;
        //fprintf(stderr, "evict key %lu (dirty? %u) to bid %lu (old-key %lu, valid? %u, dirty? %u)\n", key, is_dirty, _bid, host_cache_state[_bid].tag, host_cache_state[_bid].state.load(), host_cache_state[_bid].is_dirty);
        
        // Update cache slot state
        uint32_t lock = host_cache_state[_bid].lock.load(simt::std::memory_order_relaxed);
        if (lock == HOST_CACHE_ENTRY_UNLOCKED) {
            lock = host_cache_state[_bid].lock.exchange(HOST_CACHE_ENTRY_LOCKED, simt::std::memory_order_acquire);
            if (lock == HOST_CACHE_ENTRY_UNLOCKED && !IS_PINNED(host_cache_state[_bid])) {
                /* Start to transfer data from GPU */
                //fprintf(stderr, "evict to host: start transfer for key %lu\n", key);
                #if MULTI_DATA_STREAM
                cudaMemcpyAsync((void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, _bid), (const void*)gpu_addr, PAGE_SIZE, cudaMemcpyDeviceToHost, stream_mngr->data_stream_from_device[queue_index % NUM_DATA_STREAMS]);
                cudaStreamSynchronize(stream_mngr->data_stream_from_device[queue_index % NUM_DATA_STREAMS]);
                #else
                //fprintf(stderr, "evict to host: start transfer from device %p to host %p for key %lu done (dirty? %u) (bid %lu)\n", (void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, _bid), (const void*)gpu_addr, key, is_dirty, _bid);
                //unsigned char* test_p = (unsigned char*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, _bid);
                //fprintf(stderr, "test_p %p (host_mem %p)\n", test_p, host_mem);
                //test_p[0] = 'c';
                
                //fprintf(stderr, "from dev %p begin (bid %lu)\n", gpu_addr, _bid);
                cudaMemcpyAsync((void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, _bid), (const void*)gpu_addr, PAGE_SIZE, cudaMemcpyDeviceToHost, stream_mngr->data_stream_from_device);
                cudaStreamSynchronize(stream_mngr->data_stream_from_device);
                //fprintf(stderr, "from dev %p end (bid %lu)\n", gpu_addr, _bid);
                
                #endif
                //fprintf(stderr, "evict to host: start transfer for key %lu done (dirty? %u)\n", key, is_dirty);
                /* Update cache state */
                host_cache_state[_bid].state.store(HOST_CACHE_ENTRY_VALID, simt::std::memory_order_relaxed);
                host_cache_state[_bid].gpu_addr = (void*) gpu_addr;
                host_cache_state[_bid].tag = key;
                host_cache_state[_bid].is_dirty = is_dirty;
                //printf("evict to host: is_dirty? %u\n", is_dirty);
                this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_SUCCEEDED;
                
                // flush to ssd before releasing lock
#if USER_SPACE_ZERO_COPY
                //fprintf(stderr, "write data to ssd... %lu begin\n", _bid);
                //write_data_from_hc(this, _bid*PAGE_SIZE/512, PAGE_SIZE/512, _bid); 
                //fprintf(stderr, "write data to ssd... %lu end\n", _bid);
#endif
                //this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_SUCCEEDED;
                host_cache_state[_bid].lock.exchange(HOST_CACHE_ENTRY_UNLOCKED, simt::std::memory_order_release);

                #if USE_PINNED_MEM
                //if (pin_on_host) {
                //    host_cache_state[_bid].is_pinned = true;
                //}
                #endif
                //fprintf(stderr, "Evict to host - 1: %lu succeded (bid %lu)\n", key, _bid);
                valid_entry_count.fetch_add((uint64_t)1, std::memory_order_acquire);
                __atomic_sub_fetch(num_idle_slots_h, 1, __ATOMIC_SEQ_CST);

                if (is_dirty) {
                    total_tier1_dirty_evicts.fetch_add(1, std::memory_order_seq_cst);
                }
                else {
                    total_tier1_clean_evicts.fetch_add(1, std::memory_order_seq_cst);
                }
            }
            else {
                this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_EVICT_FAILED;
                //fprintf(stderr, "Evict to host - 2: %lu failed\n", key);
            }
        }
        else {
            this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_EVICT_FAILED;
            //fprintf(stderr, "Evict to host: %lu failed\n", key);
        }
        total_tier1_evicts.fetch_add(1, std::memory_order_seq_cst);
    }
    else if (type == GPU_RW_EVICT_TO_HOST_ASYNC) {
        uint64_t tag = this->host_rw_queue->entries[queue_index].u.ReplicationReq.tag;
        uint64_t _bid = bid.fetch_add(1, simt::memory_order_relaxed);
        _bid %= HOST_MEM_NUM_PAGES;
        // TODO: handle dirty data
        //if (this->host_rw_queue->entries[queue_index].data_dirty) {   
        //    this->host_rw_queue->entries[queue_index].data_dirty = false;
        //}
        
        // Update status first
        this->host_rw_queue->entries[queue_index].status = GPU_RW_ASYNC_PROCESSING;
        __sync_synchronize();

        // Lastly, do data transfer 
        #if MULTI_DATA_STREAM
        cudaMemcpyAsync((void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, _bid), (const void*)gpu_addr, PAGE_SIZE, cudaMemcpyDeviceToHost, stream_mngr->data_stream_from_device[queue_index % NUM_DATA_STREAMS]);
        cudaStreamSynchronize(stream_mngr->data_stream_from_device[queue_index % NUM_DATA_STREAMS]);
        #else
        cudaMemcpyAsync((void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, _bid), (const void*)gpu_addr, PAGE_SIZE, cudaMemcpyDeviceToHost, stream_mngr->data_stream_from_device);
        cudaStreamSynchronize(stream_mngr->data_stream_from_device);
        #endif
    }
    else if (type == GPU_RW_PIN_PAGE) {
        bool fail = true;
        uint64_t _bid = this->host_rw_queue->entries[queue_index].bid % HOST_MEM_NUM_PAGES;

        total_fetches.fetch_add(1, std::memory_order_seq_cst);
        /* Check if the current entry is valid and the tag matches with the key */
        if (host_cache_state[_bid].state.load(simt::memory_order_acquire) != HOST_CACHE_ENTRY_VALID 
                || host_cache_state[_bid].tag != key) {
            this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_PIN_FAILED;
            //fprintf(stderr, "PIN page failed - 1 (_bid %lu: tag %lu - key %lu)(state %lu)(dirty? %u)\n", _bid, host_cache_state[_bid].tag, key, host_cache_state[_bid].state.load(simt::memory_order_acquire), host_cache_state[_bid].is_dirty);
        }
        else {
            uint32_t lock = host_cache_state[_bid].lock.load(simt::std::memory_order_relaxed);
            if (lock == HOST_CACHE_ENTRY_UNLOCKED) {
                //host_cache_state[_bid].is_pinned = (type == GPU_RW_PIN_PAGE) ? true : false;
                lock = host_cache_state[_bid].lock.exchange(HOST_CACHE_ENTRY_LOCKED, simt::std::memory_order_acquire);
                if (lock == HOST_CACHE_ENTRY_UNLOCKED && host_cache_state[_bid].tag == key) {
                    //pending_reqs_num.fetch_add((int64_t)1, std::memory_order_acq_rel);
                    host_cache_state[_bid].is_pinned.exchange(true, simt::std::memory_order_release);
                    //host_cache_state[_bid].is_pinned.fetch_add(1, simt::std::memory_order_acquire);
                    
                    if (host_cache_state[_bid].is_dirty) { 
                        this->host_rw_queue->entries[queue_index].u.CacheReq.is_cached_page_dirty = true;
                        total_dirty_fetches.fetch_add(1, std::memory_order_seq_cst);
                    }
                    else {
                        this->host_rw_queue->entries[queue_index].u.CacheReq.is_cached_page_dirty = false;
                        total_clean_fetches.fetch_add(1, std::memory_order_seq_cst);
                    }
                    total_zc_fetches.fetch_add(1, std::memory_order_seq_cst);
        
                    this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_SUCCEEDED;
                    //fprintf(stderr, "PIN page succ %lu (reqs %lu)\n", key, pending_reqs_num.load());
                    total_hits.fetch_add(1, std::memory_order_seq_cst);
                }
                else {
                    this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_PIN_FAILED;
                    host_cache_state[_bid].lock.exchange(HOST_CACHE_ENTRY_UNLOCKED, simt::std::memory_order_release);
                    //fprintf(stderr, "PIN page failed - 3\n");
                }

            }
            else {
                this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_PIN_FAILED;
                //fprintf(stderr, "PIN page failed - 2\n");
            }
        }
    }
    else if (type == GPU_RW_UNPIN_PAGE) {
        uint64_t _bid = this->host_rw_queue->entries[queue_index].bid % HOST_MEM_NUM_PAGES;
        // should take the lock previously when doing lock page
        bool is_pinned = host_cache_state[_bid].is_pinned.exchange(false, simt::std::memory_order_release);
        assert(is_pinned == true);
        
        // Chia-Hao: 070423
        if (host_cache_state[_bid].is_dirty) {
            host_cache_state[_bid].is_dirty = false;
            host_cache_state[_bid].state.exchange(HOST_CACHE_ENTRY_INVALID, simt::std::memory_order_release);
            
            // 072223
            //__atomic_add_fetch(num_idle_slots_h, 1, __ATOMIC_SEQ_CST);
        }                 

        uint32_t locked = host_cache_state[_bid].lock.exchange(HOST_CACHE_ENTRY_UNLOCKED, simt::std::memory_order_release);
        assert(locked == HOST_CACHE_ENTRY_LOCKED);
        this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_SUCCEEDED;

        // 072223
        __atomic_add_fetch(num_idle_slots_h, 1, __ATOMIC_SEQ_CST);
    }
    else if (type == GPU_RW_FETCH_FROM_HOST) {
        //TODO
        bool fail = true;
        // TODO: bug here, should not mod HOST_MEM_NUM_PAGES
        uint64_t _bid = this->host_rw_queue->entries[queue_index].bid % HOST_MEM_NUM_PAGES;
        
        // check if address is still valid
        do {
            if (host_cache_state[_bid].state.load(simt::memory_order_relaxed) != HOST_CACHE_ENTRY_VALID || host_cache_state[_bid].tag != key) {
                //fprintf(stderr, "[%lu][%lu] tag %lu, state %u\n", key, _bid, host_cache_state[_bid].tag, host_cache_state[_bid].state.load());
                break;
            }
            uint32_t lock = host_cache_state[_bid].lock.load(simt::std::memory_order_relaxed);
            if (lock == HOST_CACHE_ENTRY_UNLOCKED) {
                uint32_t state = host_cache_state[_bid].state.load(simt::std::memory_order_acquire);
                lock = host_cache_state[_bid].lock.exchange(HOST_CACHE_ENTRY_LOCKED, simt::std::memory_order_acquire);
                /* Get the lock of this entry */
                if (lock == HOST_CACHE_ENTRY_UNLOCKED) {
                    /* Check if the tag matches */
                    if (state == HOST_CACHE_ENTRY_VALID && host_cache_state[_bid].tag == key) {
                        /* Start to transfer from host to GPU */
                        //fprintf(stderr, "fetch from host: start transfer for key %lu\n", key);
                        #if MULTI_DATA_STREAM
                        cudaMemcpyAsync((void*)gpu_addr, (const void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, _bid), PAGE_SIZE, 
                                        cudaMemcpyHostToDevice, stream_mngr->data_stream_to_device[queue_index % NUM_DATA_STREAMS]);
                        cudaStreamSynchronize(stream_mngr->data_stream_to_device[queue_index % NUM_DATA_STREAMS]);
                        #else
                        cudaMemcpyAsync((void*)gpu_addr, (const void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, _bid), PAGE_SIZE, 
                                        cudaMemcpyHostToDevice, stream_mngr->data_stream_to_device);
                        cudaStreamSynchronize(stream_mngr->data_stream_to_device);
                        #endif
                        //fprintf(stderr, "fetch from host: start transfer for key %lu done\n", key);
                        fail = false;
                        
                        this->host_rw_queue->entries[queue_index].u.CacheReq.is_cached_page_dirty = false;
                        // CHIA-HAO: make invalid so that this entry can be reused
                        if (host_cache_state[_bid].is_dirty) {
                            host_cache_state[_bid].is_dirty = false;
                            host_cache_state[_bid].state.exchange(HOST_CACHE_ENTRY_INVALID, simt::std::memory_order_release);
                            
                            this->host_rw_queue->entries[queue_index].u.CacheReq.is_cached_page_dirty = true;
                            //fprintf(stderr, "moved to gpu!!! key %lu\n", key);
                            
                            // Chia-Hao: added on 070823
                            //__atomic_add_fetch(num_idle_slots_h, 1, __ATOMIC_SEQ_CST);
                            total_dirty_fetches.fetch_add(1, std::memory_order_seq_cst);
                        }
                        else {
                            total_clean_fetches.fetch_add(1, std::memory_order_seq_cst);
                        }
                        total_cu_fetches.fetch_add(1, std::memory_order_seq_cst);

                        host_cache_state[_bid].lock.exchange(HOST_CACHE_ENTRY_UNLOCKED, simt::std::memory_order_release);
                        fetch_from_host_count.fetch_add(1, std::memory_order_acq_rel);
                        
                        // Chia-Hao: commented out on 070823
                        __atomic_add_fetch(num_idle_slots_h, 1, __ATOMIC_SEQ_CST);
                    }
                    else {
                        this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_FETCH_FAILED;
                        host_cache_state[_bid].lock.exchange(HOST_CACHE_ENTRY_UNLOCKED, simt::std::memory_order_release);
                        //goto exit_upon_failure;
                        break;
                    }
                }
            }
            //fprintf(stderr, "fetch from host (key %lu)\n", key);
        } while (fail);
        
        if (!fail) {
            this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_SUCCEEDED;
            total_hits.fetch_add(1, std::memory_order_seq_cst);
            //fprintf(stderr, "Fetch from host: %lu succedded\n", key);
        }
        else {
            this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_FETCH_FAILED;
            //fprintf(stderr, "Fetch from host: [%u][%u] failed\n", key>>32, key&0xffffffff);
        }
        total_fetches.fetch_add(1, std::memory_order_seq_cst);
    }
    else if (type == GPU_RW_FETCH_FROM_HOST_PERF_TEST) {
        bool fail = true;
        // TODO: bug here, should not mod HOST_MEM_NUM_PAGES
        uint64_t _bid = this->host_rw_queue->entries[queue_index].bid % HOST_MEM_NUM_PAGES;
        
        // check if address is still valid
        do {
            uint32_t lock = host_cache_state[_bid].lock.load(simt::std::memory_order_relaxed);
            if (lock == HOST_CACHE_ENTRY_UNLOCKED) {
                uint32_t state = host_cache_state[_bid].state.load(simt::std::memory_order_acquire);
                lock = host_cache_state[_bid].lock.exchange(HOST_CACHE_ENTRY_LOCKED, simt::std::memory_order_acquire);
                /* Get the lock of this entry */
                if (lock == HOST_CACHE_ENTRY_UNLOCKED) {
                    /* Check if the tag matches */
                    /* Start to transfer from host to GPU */
                    //fprintf(stderr, "fetch from host perf test: %lu\n", _bid);
                    #if TIME_BREAK_DOWN_FOR_PERF_TEST
                    this->perf_test_time_profile_vec[queue_index].transfer_begin = std::chrono::high_resolution_clock::now();
                    #endif

                    #if MULTI_DATA_STREAM
                    cudaMemcpyAsync((void*)gpu_addr, (const void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, _bid), PAGE_SIZE, 
                                    cudaMemcpyHostToDevice, stream_mngr->data_stream_to_device[queue_index % NUM_DATA_STREAMS]);
                    cudaStreamSynchronize(stream_mngr->data_stream_to_device[queue_index % NUM_DATA_STREAMS]);
                    #else
                    cudaMemcpyAsync((void*)gpu_addr, (const void*)NVM_PTR_OFFSET(host_mem, PAGE_SIZE, _bid), PAGE_SIZE, 
                                    cudaMemcpyHostToDevice, stream_mngr->data_stream_to_device);
                    cudaStreamSynchronize(stream_mngr->data_stream_to_device);
                    #endif
                    //fprintf(stderr, "fetch from host perf test done: %lu\n", _bid);
                    
                    #if TIME_BREAK_DOWN_FOR_PERF_TEST
                    this->perf_test_time_profile_vec[queue_index].transfer_end = std::chrono::high_resolution_clock::now();
                    #endif

                    fail = false;
                    host_cache_state[_bid].lock.exchange(HOST_CACHE_ENTRY_UNLOCKED, simt::std::memory_order_release);

                    valid_entry_count.fetch_sub((uint64_t)1, std::memory_order_acquire);
                    __atomic_sub_fetch(num_idle_slots_h, 1, __ATOMIC_SEQ_CST);
                }
            }
        } while (fail);
        
        if (!fail) {
            this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_SUCCEEDED;
        }
        else {
            this->host_rw_queue->entries[queue_index].u.CacheReq.return_code = HOST_CACHE_FETCH_FAILED;
        }
    }
    else if (type == GPU_RW_FETCH_TO_HOST_BY_HOST) {
        // TODO:
    }
    else if (type == GPU_RW_LOGGING) {
        loggingOnHost(this->host_rw_queue->entries[queue_index].u.LoggingReq.arrayIdx, 
                      this->host_rw_queue->entries[queue_index].u.LoggingReq.pageIdx,
                      this->host_rw_queue->entries[queue_index].u.LoggingReq.actionType);
    }
    else if (type == GPU_RW_ARBITRARY_LOGGING) {
        //pending_reqs_num.fetch_add((int64_t)1, std::memory_order_acq_rel);
        int64_t n = pending_reqs_num.load(std::memory_order_acquire);
        //fprintf(stderr, "thread count %lu (#pending reqs %ld)\n", this->host_rw_queue->entries[queue_index].u.ArbitraryLoggingReq.threadCount, n);
        //std::stringstream ss;
        //ss << std::dec << this->host_rw_queue->entries[queue_index].u.ArbitraryLoggingReq.threadCount << std::endl;
        //std::cout << ss.str();
        //pending_reqs_num.fetch_sub((int64_t)1, std::memory_order_acq_rel);
    }
    else if (type == GPU_RW_GET_REQS_NUM) {
        this->host_rw_queue->entries[queue_index].u.CacheReq.num_reqs = (int)pending_reqs_num.load(std::memory_order_acquire);
    }
    else if (type == GPU_RW_SAMPLE_MEM) {
        //this->mem_sample_collector->push_samples(key, this->host_rw_queue->entries[queue_index].u.SampleMemReq.virt_timestamp_dist);
    }
    
req_ready:
    if (type != GPU_RW_PIN_PAGE && type != GPU_RW_UNPIN_PAGE) {
        pending_reqs_num.fetch_sub((int64_t)1, std::memory_order_acq_rel);
    }

//req_ready:
    //*/
    this->host_rw_queue->entries[queue_index].status = GPU_RW_READY;
    __sync_synchronize();
    //printf("%s: queue index %d Finished (%lu)\n", __func__, queue_index, this->host_rw_queue->entries[queue_index].u.ArbitraryLoggingReq.threadCount);   
    //printf("[Tier-2] Valid Entry Count %lu\n", valid_entry_count.load(std::memory_order_acquire));

    #if TIME_BREAK_DOWN_FOR_PERF_TEST
    this->perf_test_time_profile_vec[queue_index].handle_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> transfer_time = std::chrono::duration_cast<std::chrono::duration<double>>(
            this->perf_test_time_profile_vec[queue_index].transfer_end - this->perf_test_time_profile_vec[queue_index].transfer_begin);
    std::chrono::duration<double> handle_time = std::chrono::duration_cast<std::chrono::duration<double>>(
            this->perf_test_time_profile_vec[queue_index].handle_end - this->perf_test_time_profile_vec[queue_index].handle_begin);
    std::chrono::duration<double> total_time = std::chrono::duration_cast<std::chrono::duration<double>>(
            this->perf_test_time_profile_vec[queue_index].handle_end - this->perf_test_time_profile_vec[queue_index].thread_launch);
    /*
    fprintf(stderr, "thread launch %f us, update time %f us, transfer time %f us (total time %f us)\n", (total_time-handle_time)*1000000, 
            (handle_time-transfer_time)*1000000, transfer_time.count()*1000000, total_time.count()*1000000);
    */
    #endif


    return;
//exit_upon_failure:
//    std::cerr << "Failed at handleRequest" << std::endl;
//    exit(1);
}

void HostCache::debugPrint()
{
    for (int i = 0; i < GPU_RW_SIZE; i++) {
        fprintf(stderr, "%d status %s\n", i, enum_rw_status[this->host_rw_queue->entries[i].status]);
    }
}

void HostCache::registerRangesLBA(uint64_t starting_lba)
{
    this->ranges_starting_lba.push_back(starting_lba);
}


void HostCache::threadLoop(int tid)
{
#if PRE_LAUNCH_THREADS 
    uint32_t queue_index;
    while (1) {
        queue_index = this->global_qid.load(std::memory_order_acquire);
        //printf("queue %u\n", queue_index);
        //if (tid == 0) debugPrint();
        if (this->host_rw_queue->entries[queue_index & (GPU_RW_SIZE-1)].status == GPU_RW_PENDING) {
            CPU_PRINT("Pending Request at Queue Slot %d\n", queue_index);
            //fprintf(stderr, "pending queue %u\n", queue_index);
            bool exchanged = 
                this->global_qid.compare_exchange_weak(queue_index, queue_index+1, std::memory_order_release);
            if (!exchanged) 
                continue;
            //fprintf(stderr, "got queue %u\n", queue_index);
            queue_index %= GPU_RW_SIZE;
            this->host_rw_queue->entries[queue_index].status = GPU_RW_PROCESSING;
            #if TIME_BREAK_DOWN_FOR_PERF_TEST
            this->perf_test_time_profile_vec[queue_index].thread_launch = std::chrono::high_resolution_clock::now();
            #endif
            handleRequest(queue_index);
            //FlushGPUDirect();
        }
        // Chia-Hao: 1117
        if (revoke_runtime) break;
    }
#endif
}

void HostCache::threadLoop2(int tid)
{
#if PRE_LAUNCH_THREADS 
    uint32_t queue_index;
    while (true) {
        for (queue_index = tid; queue_index < GPU_RW_SIZE; queue_index = (queue_index+NUM_CPU_CORES)&(GPU_RW_SIZE-1) ) {
            if (this->host_rw_queue->entries[queue_index].status == GPU_RW_PENDING) {
                this->host_rw_queue->entries[queue_index].status = GPU_RW_PROCESSING;
                #if TIME_BREAK_DOWN_FOR_PERF_TEST
                this->perf_test_time_profile_vec[queue_index].thread_launch = std::chrono::high_resolution_clock::now();
                #endif
                handleRequest(queue_index);
            }
        }
    }
#endif
}


void HostCache::launchThreadLoop()
{
    int tid = 0;
    //int req_count = 0;
    for (tid = 0; tid < NUM_CPU_CORES; tid++) {
        std::thread handler_thread(&HostCache::threadLoop, std::ref(*this), tid);
        //std::thread handler_thread(&HostCache::threadLoop2, std::ref(*this), tid);
        handler_thread.detach();
    }
    while(!revoke_runtime) sleep(1);
}

void HostCache::mainLoop()
{
    int queue_index = 0;
    int req_count = 0;
    //cu_context_init();
    CPU_PRINT("mainLoop starts... \n");
    do {
        // parse reqeusts from GPU
        if (this->host_rw_queue->entries[queue_index].status == GPU_RW_WAIT_FLUSH) {
            //FlushGPUDirect();
            err = cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice, cudaFlushGPUDirectRDMAWritesToOwner);
            if (err != cudaSuccess) fprintf(stderr, "Flush GPU Direct RDMA wirtes failed.\n");
            this->host_rw_queue->entries[queue_index].status = GPU_RW_READY;
            __sync_synchronize();
        }
        if (this->host_rw_queue->entries[queue_index].status == GPU_RW_PENDING) {
            //CPU_PRINT("Pending Request at Queue Slot %d\n", queue_index);
            //fprintf(stderr, "Pending Request at Queue Slot %d (req_count %d)\n", queue_index, req_count);
            this->host_rw_queue->entries[queue_index].status = GPU_RW_PROCESSING;
            #if TIME_BREAK_DOWN_FOR_PERF_TEST
            this->perf_test_time_profile_vec[queue_index].thread_launch = std::chrono::high_resolution_clock::now();
            #endif
            
            // spawn another to address the request
            std::thread handler_thread(&HostCache::handleRequest, std::ref(*this), queue_index);
            handler_thread.detach();
            //printf("queue index %u\n", queue_index);
            //handleRequest(queue_index);

            req_count++;
            //printf("req_count %d\n", req_count);
        }
        //FlushGPUDirect();
        queue_index = (queue_index + 1) & (GPU_RW_SIZE-1);
    } while (!revoke_runtime);
}

void HostCache::preLoadData(int fd, size_t page_size)
{
    if (preload) {
        // Host buffers
        lseek(fd, 0, SEEK_SET);
        for (int i = 0; i < GPU_RW_SIZE; i++) {
            // TODO: Fix size 2m now
            //cudaHostAlloc(&(this->host_buffer[i]), PAGE_SIZE_2M, cudaHostAllocDefault);
            ssize_t read_size = read(fd, this->host_buffer[i], page_size);
            if (read_size < 0) {
                fprintf(stderr, "Pre-Load data (fd %d) failed %ld (%s)\n", fd, read_size, strerror(errno));
                exit(-1);
            }
        }
    }
}


static std::atomic<uint64_t> dirty_page_count;

static inline uint64_t get_lba(HostCache* hc, uint64_t key)
{
    uint32_t range_id = (key >> 32) & 0xffffffff;
    uint32_t range_page_id = (key) & 0xffffffff;
    uint64_t lba = host_cache->ranges_starting_lba[range_id] + range_page_id*(PAGE_SIZE/512);
    
    return lba;
}

static void* host_flush_page(void* arg)
{
    uint64_t start_page = *(uint64_t*)(arg);
    uint32_t queue_id = start_page % NUM_NVME_QUEUES;

    //printf("%s: start page %lu (num pages %lu)\n", __func__, start_page, host_cache->num_pages);
    for (uint32_t page = start_page; page < (host_cache->num_pages); page += NUM_FLUSH_THREADS) {
        //fprintf(stderr, "%s: page %lu (%lu) is_dirty? %u, valid? %u\n", __func__, page, host_cache->num_pages,
        //host_cache->host_cache_state[page].is_dirty,
        //host_cache->host_cache_state[page].state == HOST_CACHE_ENTRY_VALID);
        if (host_cache->host_cache_state[page].state == HOST_CACHE_ENTRY_VALID && host_cache->host_cache_state[page].is_dirty) {
        //fprintf(stderr, "%s: page %lu (%lu)\n", __func__, page, host_cache->num_pages);
            uint64_t lba = get_lba(host_cache, host_cache->host_cache_state[page].tag);
            //print_content(host_cache->host_mem, page);
#if USER_SPACE_ZERO_COPY
            write_data_from_hc(host_cache, lba, PAGE_SIZE/512, page, queue_id); 
#else
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); 
#endif
            dirty_page_count.fetch_add(1);
        }
    }

    return NULL;;
}

static void* host_flush_page_batch(void* arg)
{
    uint64_t start_page = *(uint64_t*)(arg);
    uint32_t queue_id = start_page % NUM_NVME_QUEUES;
    uint16_t cid[16];

    for (uint32_t page = start_page; page < (host_cache->num_pages); page += NUM_FLUSH_THREADS*32) {
        for (uint32_t i = 0; i < 32; i++) {
            uint32_t v_page = page + NUM_FLUSH_THREADS*i;
            if (host_cache->host_cache_state[v_page].state == HOST_CACHE_ENTRY_VALID && host_cache->host_cache_state[v_page].is_dirty) {
                // TODO:
                cid[i] = submit_io_async(host_cache, v_page*(PAGE_SIZE/512)/*starting_lba*/, PAGE_SIZE/512, v_page, queue_id);
                dirty_page_count.fetch_add(1);
            }
        }

        for (uint32_t i = 0; i < 32; i++) {
            poll_io(host_cache, cid[i], queue_id); 
        }
    }

    return NULL;;
}


static void* start_host_runtime_thread(void* arg) 
{
    host_cache = new HostCache((Controller*)arg, pc_mem_size);
    //host_cache->setGDSHandler((GDS_HANDLER*)arg);
    CPU_PRINT("Start Host Cache Main Loop\n");
    
    #if PRE_LAUNCH_THREADS
    host_cache->launchThreadLoop();
    #else
    host_cache->mainLoop();
    #endif

    delete host_cache;
    printf("Host Cache deleted...\n");
    return NULL;
}

HostCache* createHostCache(Controller* ctrl, size_t _pc_mem_size)
{
    int ret;

    pc_mem_size = _pc_mem_size;
    //GDS_HANDLER* gds_handler = new GDS_HANDLER();
    ret = pthread_create(&host_runtime_thread, NULL, &start_host_runtime_thread, (void*)ctrl);
    if (ret != 0) {
        handle_error_en(ret, "Create Host Runtime Failed...");   
    }
    while (host_cache == NULL);
    //pthread_join(host_runtime_thread, NULL);

    // for host p2p
    //#if !EXCLUDE_SPIN
    //spin_init();
    //#endif

    CPU_PRINT("Create Host Cache Done!\n");
    return (HostCache*)host_cache;
}

void flushHostCache()
{
    #if 1
    for (uint64_t i = 0; i < NUM_FLUSH_THREADS; i++) {
        //fprintf(stderr, "launching new thread for flushing %lu\n", i);
        flush_thread_args[i] = i;
        int err = pthread_create(&(flush_thread[i]), NULL, &host_flush_page, (void*)&flush_thread_args[i]);
        if (err != 0) {
            fprintf(stderr, "launching new thread for flushing %lu failed\n", i);
        }
    }

    for (uint64_t i = 0; i < NUM_FLUSH_THREADS; i++) {
        pthread_join(flush_thread[i], NULL);
    }

    printf("Dirty pages count : %lu\n", dirty_page_count.load());

    #endif
    #if 0
    for (uint32_t page = 0; page < (host_cache->num_pages); page += NUM_FLUSH_THREADS) {
        //fprintf(stderr, "%s: page %lu (%lu)\n", __func__, page, host_cache->num_pages);
        if (host_cache->host_cache_state[page].state == HOST_CACHE_ENTRY_VALID && host_cache->host_cache_state[page].is_dirty) {
            write_data_from_hc(host_cache, page*PAGE_SIZE/512, PAGE_SIZE/512, page); 
        }
    }
    #endif
}

__global__ void set_last_iteration_flag()
{
    last_iteration = true;
}

void lastIteration(cudaStream_t* s)
{
    bool val = true;
    //cudaMemset(&last_iteration, false, sizeof(bool));
    //cudaMemcpyToSymbolAsync(last_iteration, &val, sizeof(bool), 0, cudaMemcpyHostToDevice, *s);
    set_last_iteration_flag<<<1,1,0,*s>>>();
    std::cerr << "set last iteration done...\n";
}  

void revokeHostRuntime()
{
    //cuCtxDestroy(ctx);
    //PRINT_CU_ERROR("cuCtxDestroy");
    std::cerr << "Revoking host runtime...\n";
    revoke_runtime = true;

    //for (uint64_t i = 0; i < NUM_FLUSH_THREADS; i++) {
    //    pthread_create(&flush_thread[i], NULL, &host_flush_page, (void*)&i);
    //}

    //int st = pthread_cancel(host_runtime_thread);
    //if (st != 0) std::cerr << "Revoking failed...\n";
    pthread_join(host_runtime_thread, NULL);

    //for (uint64_t i = 0; i < NUM_FLUSH_THREADS; i++) {
    //    pthread_join(flush_thread[i], NULL);
    //}

    //delete host_cache;
    std::cerr << "Revoke host runtime done.\n";
}

void preLoadData(int fd, size_t page_size)
{
    host_cache->preLoadData(fd, page_size);
}

//void registerGPUMem(void* gpu_addr, size_t length)
//{
//    #if !EXCLUDE_SPIN
//    register_gpu_mem(gpu_addr, length);
//    #endif
//}

std::unordered_map<uint64_t,std::map<uint32_t,uint64_t>> page_bursts;
std::unordered_map<uint64_t,std::map<uint32_t,uint64_t>> reuse_dist_after_burst;

void loadBurstProfileFile(std::string& filename) 
{
    std::string line;
    uint64_t key;
    uint32_t size;
    uint32_t burst_count;
    uint32_t reuse_dist;
    std::fstream fs(filename.c_str(), std::fstream::in);
    if (fs.is_open()) {
        do {
            fs >> line;
            key = stol(line);
            fs >> line;
            size = stoi(line);
            //std::cout << "key " << key << "; size " << size << std::endl;
            for (uint32_t i = 0; i < size; i++) {
                fs >> line;
                burst_count = stoi(line);
                fs >> line;
                reuse_dist = stoi(line);
                page_bursts[key][i] = burst_count;
                reuse_dist_after_burst[key][i] = reuse_dist;
                //std::cout << "index " << i << "; burst " << burst_count << std::endl;
            }
        } while (!fs.eof());
        fs.close();
    }
    else {
        std::cerr << "Burst Profile File open failed..." << std::endl;
        exit(1);
    }
    std::cout << "Burst Profile File processed..." << std::endl;
}


#endif
