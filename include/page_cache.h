#ifndef __PAGE_CACHE_H__
#define __PAGE_CACHE_H__

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif


#include "util.h"
#include "host_util.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include "buffer.h"
#include "ctrl.h"
#include <iostream>
#include "nvm_parallel_queue.h"
#include "nvm_cmd.h"

// CHIA-HAO
#include "host_cache.h"
#include "helper.h"

#define FREE 2
// enum locality {HIGH_SPATIAL, LOW_SPATIAL, MEDIUM_SPATIAL};
// template <typname T>
// stryct array_t {
//   range_t<T>* ranges;
//   uint32_t n_ranges;

//   void add_range(start_idx, end_idx, locality l )
//   {
//   ranges.push_back(new range(star))
//   }
// }

// enum page_state {USE = 1U, USE_DIRTY = ((1U << 31) | 1), VALID_DIRTY = (1U << 31),
//     VALID = 0U, INVALID = (UINT_MAX & 0x7fffffff),
//     BUSY = ((UINT_MAX & 0x7fffffff)-1)};

enum data_dist_t {REPLICATE = 0, STRIPE = 1};

// #define USE (1ULL)
// #define VALID_DIRTY (1ULL << 31)
// #define USE_DIRTY (VALID_DIRTY | USE)
// #define VALID (0ULL)
// #define INVALID (0xffffffffULL)
// #define BUSY ((0xffffffffULL)-1)

#define ALL_CTRLS 0xffffffffffffffff

//broken
#define VALID_DIRTY (1ULL << 31)
#define USE_DIRTY (VALID_DIRTY | USE)

#define INVALID 0x00000000U
#define VALID   0x80000000U
#define BUSY    0x40000000U
#define DIRTY   0x20000000U
#define PINNTED 0x10000000U

#define CNT_SHIFT (29ULL)
//#define CNT_MASK 0x1fffffffU
//#define CNT_SHIFT (28ULL)
#define CNT_MASK 0x7ffffffU
#define VALID_MASK 0x7
#define BUSY_MASK 0xb
#define DISABLE_BUSY_ENABLE_VALID 0xc0000000U
#define DISABLE_BUSY_MASK 0xbfffffffU
#define NV_NB 0x00U
#define NV_B 0x01U
#define V_NB 0x02U
#define V_B 0x03U

// CHIA-HAO
#define CACHED_BY_HOST  0x08000000U
#define UNDER_PREFETCH  0x04000000U
#define RRPV_MAX 3
#define ZC_THRESHOLD (8)

struct page_cache_t;

struct page_cache_d_t;

//typedef padded_struct_pc* page_states_t;

template <typename T>
class Prefetcher;

template <typename T>
struct range_t;

template<typename T>
struct array_d_t;

template <typename T>
struct range_d_t;

/*struct data_page_t {
  simt::atomic<uint64_t, simt::thread_scope_device>  state; //state
  //
  uint32_t offset;

  };
*/
typedef PageReuseTable page_reuse_hist_t;

typedef struct __align__(32) {
    uint32_t num_burst;
    uint32_t curr_burst;
    uint32_t* burst_count;
    uint32_t* reuse_dist_after_burst;
} burst_profile_t;

typedef struct __align__(32) {
    simt::atomic<uint32_t, simt::thread_scope_device>  state; //state
                                                              //
    simt::atomic<uint32_t, simt::thread_scope_device>  access_in_burst;                                                          
    uint32_t offset;
    int32_t bid;
    //uint8_t pad[32-4-4];

    //uint32_t burst_idx;
    //uint32_t burst_arr_size;
    //uint32_t* burst_count;
    //uint32_t* reuse_dist_after_burst;
    burst_profile_t burst_profile;

} __attribute__((aligned (32))) data_page_t;

typedef data_page_t* pages_t;

template<typename T>
struct returned_cache_page_t {
    T* addr;
    uint32_t size;
    uint32_t offset;

    T operator[](size_t i) const {
        if (i < size)
            return addr[i];
        else
            return 0;
    }

    T& operator[](size_t i) {
        if (i < size)
            return addr[i];
        else
            return addr[0];
    }
};
#define THREAD_ 0
#define SHARED_ 1
#define GLOBAL_ 2

//#ifdef __CUDACC__
#define TID ( (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)))
//#else
//#define TID 0
//#endif

//#ifdef __CUDACC__
#define BLKSIZE ( (blockDim.x * blockDim.y * blockDim.z) )
//#else
//#define BLKSIZE 1
//#endif

#ifdef __CUDACC__
#define SYNC (loc != THREAD_ ? __syncthreads() : (void)0)
#else
#define SYNC (void)0
#endif




#define INVALID_ 0x00000000
#define VALID_ 0x80000000
#define BUSY_ 0x40000000
#define CNT_MASK_ 0x3fffffff
#define INVALID_MASK_ 0x7fffffff
#define DISABLE_BUSY_MASK_ 0xbfffffff

// CHIA-HAO
#define USE_HOST_CACHE 0
#define ENABLE_TMM 1
#define PRINT_TMM 0
#define ALWAYS_EVICT 0
#define ACCESS_HOST_CACHE_WITH_ZERO_COPY 1
#define SPECIAL_EVICTION_FOR_CLEAN_PAGES 1
#define MAX_SAMPLES_APPLIED 1


#define MAX_TRY_SAMPLES 3276800 
//#if USE_HOST_CACHE
//#endif 

template<simt::thread_scope _scope = simt::thread_scope_device>
struct tlb_entry {
    uint64_t global_id;
    simt::atomic<uint32_t, _scope> state;
    data_page_t* page = nullptr;

    __forceinline__
    __host__ __device__
    tlb_entry() { init(); }

    __forceinline__
    __host__ __device__
    void init() {
        global_id = 0;
        state.store(0, simt::memory_order_relaxed);
        page = nullptr;
    }

    __forceinline__
    __device__
    void release(const uint32_t count) {
//		    if (global_id == 515920192)
//			printf("--(2)st: %llx\tcount: %llu\n", (unsigned long long) state.load(simt::memory_order_relaxed), (unsigned long long) count);

        state.fetch_sub(count, simt::memory_order_release); }

    __forceinline__
    __device__
    void release() { if (page != nullptr)  {
//		    if (global_id == 515920192)
//			printf("--(1)st: %llx\tcount: %llu\n", (unsigned long long) state.load(simt::memory_order_relaxed), (unsigned long long) 1);

            page->state.fetch_sub(1, simt::memory_order_release); }}



};




template<typename T, size_t n = 32, simt::thread_scope _scope = simt::thread_scope_device, size_t loc = GLOBAL_>
struct tlb {
    tlb_entry<_scope> entries[n];
    array_d_t<T>* array = nullptr;

    __forceinline__
    __host__ __device__
    tlb() {}
/*
  __forceinline__
  __device__
  tlb(array_d_t<T>* a) { init(a); }
*/
    __forceinline__
    __device__
    void init(array_d_t<T>* a) {
        //SYNC;
//	__syncthreads();
        if (n) {
            size_t tid = TID;
            if (tid == 0)
                array = a;
            for (; tid < n; tid+=BLKSIZE)
                entries[tid].init();
        }

//	__syncthreads();
//        SYNC;

    }

    __forceinline__
    __device__
    void fini() {
        //      SYNC;
//	__syncthreads();

        if (n) {
            size_t tid = TID;
            for (; tid < n; tid+=BLKSIZE)
                entries[tid].release();
        }
//	__syncthreads();

//        SYNC;
    }

    __forceinline__
    __device__
    ~tlb() {  }

    __forceinline__
    __device__
    T* acquire(const size_t i, const size_t gid, size_t& start, size_t& end, range_d_t<T>* range, const size_t page_) {

        //size_t gid = array->get_page_gid(i);
        uint32_t lane = lane_id();
        size_t ent = gid % n;
        tlb_entry<_scope>* entry = entries + ent;
        uint32_t mask = __activemask();
        uint32_t eq_mask = __match_any_sync(mask, gid);
        eq_mask &= __match_any_sync(mask, (uint64_t)this);
        uint32_t master = __ffs(eq_mask) - 1;
        uint32_t count = __popc(eq_mask);
        uint64_t base_master, base;
        if (lane == master) {
            uint64_t c = 0;
            bool cont = false;
            uint32_t st;
            do {

                //lock;
                do {
                    st = entry->state.fetch_or(VALID_, simt::memory_order_acquire);
                    if ((st & VALID_) == 0)
                        break;
                    __nanosleep(100);
                } while (true);

                if ((entry->page != nullptr) && (gid == entry->global_id)) {
//		    if (gid == 515920192)
//			printf("++(1)st: %llx\tst&Val: %llx\tcount: %llu\n", (unsigned long long) st, (unsigned long long) (st & VALID_), (unsigned long long) count);

                    st += count;

                    base_master = (uint64_t) range->get_cache_page_addr(entry->page->offset);

                    entry->state.store(st, simt::memory_order_release);
                    break;
                }
                else if(((entry->page == nullptr)) || (((st & 0x3fffffff) == 0))) {
//		    if (gid == 515920192)
//			printf("++(2)st: %llx\tst&Val: %llx\tVal: %llx\tcount: %llu\n", (unsigned long long) st, (unsigned long long) (st & VALID_), (unsigned long long) VALID_, (unsigned long long) count);
//
                    if (entry->page != nullptr)
                        entry->page->state.fetch_sub(1, simt::memory_order_release);
                    data_page_t* page = nullptr;// = (data_page_t*)0xffffffffffffffff;
                    base_master = (uint64_t) array->acquire_page_(i, page, start, end, range, page_);
                    if (((uint64_t) page == 0xffffffffffffffff) || (page == nullptr))
                        printf("failure\n");
                    entry->page = page;
                    entry->global_id = gid;
                    st += count;
                    entry->state.store(st, simt::memory_order_release);
                    break;

                }
                else {
                    if (++c % 100000 == 0)
                        printf("c: %llu\ttid: %llu\twanted_gid: %llu\tgot_gid: %llu\tst: %llx\tst&0x7: %llx\n", (unsigned long long) c, (unsigned long long) (TID), (unsigned long long) gid, (unsigned long long) entry->global_id, (unsigned long long) st, (unsigned long long) (st & 0x7fffffff));
                    entry->state.store(st, simt::memory_order_relaxed);
                    __nanosleep(100);

                }

            } while(true);


        }

        base_master = __shfl_sync(eq_mask,  base_master, master);

        return (T*) base_master;

    }

    __forceinline__
    __device__
    void release(const size_t gid) {
        //size_t gid = array->get_page_gid(i);
        uint32_t lane = lane_id();
        uint32_t mask = __activemask();
        uint32_t eq_mask = __match_any_sync(mask, gid);
        eq_mask &= __match_any_sync(mask, (uint64_t)this);
        uint32_t master = __ffs(eq_mask) - 1;
        uint32_t count = __popc(eq_mask);

        size_t ent = gid % n;
        tlb_entry<_scope>* entry = entries + ent;
        if (lane == master)
            entry->release(count);
        __syncwarp(eq_mask);

    }

};

template<typename T, size_t n = 32, simt::thread_scope _scope = simt::thread_scope_device, size_t loc = GLOBAL_>
struct bam_ptr_tlb {
    tlb<T,n,_scope,loc>* tlb_ = nullptr;
    array_d_t<T>* array = nullptr;
    range_d_t<T>* range;
    size_t page;
    size_t start = 0;
    size_t end = 0;
    size_t gid = 0;
    //int64_t range_id = -1;
    T* addr = nullptr;

    __forceinline__
    __host__ __device__
    bam_ptr_tlb(array_d_t<T>* a, tlb<T,n,_scope,loc>* t) { init(a, t); }

    __forceinline__
    __device__
    ~bam_ptr_tlb() { fini(); }

    __forceinline__
    __host__ __device__
    void init(array_d_t<T>* a, tlb<T,n,_scope,loc>* t) { array = a; tlb_ = t; }

    __forceinline__
    __device__
    void fini(void) {
        if (addr) {

            tlb_->release(gid);
            addr = nullptr;
        }

    }

    __forceinline__
    __device__
    void update_page(const size_t i) {
        ////printf("++++acquire: i: %llu\tpage: %llu\tstart: %llu\tend: %llu\trange: %llu\n",
//            (unsigned long long) i, (unsigned long long) page, (unsigned long long) start, (unsigned long long) end, (unsigned long long) range_id);
        fini(); //destructor
        array->get_page_gid(i, range, page, gid);
        addr = (T*) tlb_->acquire(i, gid, start, end, range, page);
//        //printf("----acquire: i: %llu\tpage: %llu\tstart: %llu\tend: %llu\trange: %llu\n",
//            (unsigned long long) i, (unsigned long long) page, (unsigned long long) start, (unsigned long long) end, (unsigned long long) range_id);
    }

    __forceinline__
    __device__
    T operator[](const size_t i) const {
        if ((i < start) || (i >= end)) {
            update_page(i);
        }
        return addr[i-start];
    }

    __forceinline__
    __device__
    T& operator[](const size_t i) {
        if ((i < start) || (i >= end)) {
            update_page(i);
            range->mark_page_dirty(page);
        }
        return addr[i-start];
    }
};


template<typename T>
struct bam_ptr {
    data_page_t* page = nullptr;
    array_d_t<T>* array = nullptr;
    size_t start = 0;
    size_t end = 0;
    int64_t range_id = -1;
    T* addr = nullptr;

    __forceinline__
    __host__ __device__
    bam_ptr(array_d_t<T>* a) { init(a); }

    __forceinline__
    __host__ __device__
    ~bam_ptr() { fini(); }

    __forceinline__
    __host__ __device__
    void init(array_d_t<T>* a) { array = a; }

    __forceinline__
    __host__ __device__
    void fini(void) {
        if (page) {
            array->release_page(page, range_id, start);
            page = nullptr;
        }

    }

    __forceinline__
    __host__ __device__
    T* update_page(const size_t i) {
        ////printf("++++acquire: i: %llu\tpage: %llu\tstart: %llu\tend: %llu\trange: %llu\n",
//            (unsigned long long) i, (unsigned long long) page, (unsigned long long) start, (unsigned long long) end, (unsigned long long) range_id);
        fini(); //destructor
        addr = (T*) array->acquire_page(i, page, start, end, range_id);
//        //printf("----acquire: i: %llu\tpage: %llu\tstart: %llu\tend: %llu\trange: %llu\n",
//            (unsigned long long) i, (unsigned long long) page, (unsigned long long) start, (unsigned long long) end, (unsigned long long) range_id);
        return addr;
    }

    __forceinline__
    __host__ __device__
    T operator[](const size_t i) const {
        if ((i < start) || (i >= end)) {
           T* tmpaddr =  update_page(i);
        }
        return addr[i-start];
    }
    
    __host__ __device__
    T* memref(size_t i) {
        T* ret_; 
        if ((i < start) || (i >= end)) {
           ret_ =  update_page(i);
        }
        return ret_;
    }


    __forceinline__
    __host__ __device__
    T& operator[](const size_t i) {
        if ((i < start) || (i >= end)) {
            update_page(i);
            //page->state.fetch_or(DIRTY, simt::memory_order_relaxed);
        }
        return addr[i-start];
    }
    
    // CHIA-HAO
    __forceinline__
    __host__ __device__
    void write(const size_t i, T val) {
        if ((i < start) || (i >= end)) {
            update_page(i);
            page->state.fetch_or(DIRTY, simt::memory_order_relaxed);
        }
        addr[i-start] = val;
    }

};

typedef struct __align__(32) {
    simt::atomic<uint32_t, simt::thread_scope_device>  page_take_lock; //state
    //
    // CHIA-HAO
    int32_t  bid = -1;
    uint64_t page_translation;

    uint8_t pad[32-8];

} __attribute__((aligned (32))) cache_page_t;
/*
  struct cache_page_t {
  simt::atomic<uint32_t, simt::thread_scope_device>  page_take_lock;
  uint32_t  page_translation;
  uint8_t   range_id;
  };
*/
struct page_cache_d_t {
    uint8_t* base_addr;
    uint64_t page_size;
    uint64_t page_size_minus_1;
    uint64_t page_size_log;
    uint64_t n_pages;
    uint64_t n_pages_minus_1;
    cache_page_t* cache_pages;
    //uint32_t* page_translation;         //len = num of pages in cache
    //padded_struct_pc* page_translation;         //len = num of pages in cache
    //padded_struct_pc* page_take_lock;      //len = num of pages in cache
    padded_struct_pc* page_ticket;
    uint64_t* prp1;                  //len = num of pages in cache
    uint64_t* prp2;                  //len = num of pages in cache if page_size = ctrl.page_size *2
    //uint64_t* prp_list;              //len = num of pages in cache if page_size > ctrl.page_size *2
    uint64_t    ctrl_page_size;
    uint64_t  range_cap;
    //uint64_t  range_count;
    pages_t*   ranges;
    pages_t*   h_ranges;
    // CHIA-HAO
    page_reuse_hist_t** ranges_reuse_hist;
    page_reuse_hist_t** h_ranges_reuse_hist;
    uint64_t ring_count;
    
#if MAX_SAMPLES_APPLIED
    uint64_t try_samples = 0;
#endif
    //uint64_t tier2_occupied_count;

    uint64_t n_ranges;
    uint64_t n_ranges_bits;
    uint64_t n_ranges_mask;
    uint64_t n_cachelines_for_states;

    uint64_t* ranges_page_starts;
    data_dist_t* ranges_dists;
    simt::atomic<uint64_t, simt::thread_scope_device>* ctrl_counter;

    simt::atomic<uint64_t, simt::thread_scope_device>* q_head;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_tail;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_lock;
    simt::atomic<uint64_t, simt::thread_scope_device>* extra_reads;
    
    simt::atomic<uint64_t, simt::thread_scope_device>* evict_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device>* virt_timestamp;
    simt::atomic<uint64_t, simt::thread_scope_device>* virt_timestamp_sampled;
    simt::atomic<int64_t, simt::thread_scope_device>* tier2_occupied_count;
#if ACCESS_HOST_CACHE_WITH_ZERO_COPY
    simt::atomic<int64_t, simt::thread_scope_device>* simul_reqs_count;
#endif
#if GET_GOLDEN_REUSE_DISTANCE
    simt::atomic<uint64_t, simt::thread_scope_device>* total_pred_count;
    simt::atomic<uint64_t, simt::thread_scope_device>* accurate_count;
#endif
    //simt::atomic<int64_t, simt::thread_scope_device>* rrpv_increment;

    Controller** d_ctrls;
    uint64_t n_ctrls;
    bool prps;

    uint64_t n_blocks_per_page;
    
    // Chia-Hao: statistics for pages
#if SPECIAL_EVICTION_FOR_CLEAN_PAGES
    uint64_t total_pages_num = 0;
    simt::atomic<uint64_t, simt::thread_scope_device>* dirty_pages_evicted;
    simt::atomic<uint64_t, simt::thread_scope_device>* clean_pages_evicted;
    #define special_handling_threshold() ((2*1024*1024*1024)/65536)
    #define all_pages_evicted() (dirty_pages_evicted->load() + clean_pages_evicted->load())

    // Chia-Hao: 080623
    simt::atomic<uint64_t, simt::thread_scope_device>* unique_page_evict_num;
#endif
    

    __forceinline__
    __device__
    cache_page_t* get_cache_page(const uint32_t page) const;

    __forceinline__
    __device__
    uint32_t find_slot(uint64_t address, uint64_t range_id, const uint32_t queue_, int32_t* bid = NULL);

    // CHIA-HAO
    __forceinline__
    __device__
    //uint32_t find_slot_prefetching();
    uint32_t find_slot_prefetching(uint64_t address, uint64_t range_id, const uint32_t queue_, int32_t* bid = NULL);
    

    __forceinline__
    __device__
    uint64_t get_cache_page_addr(const uint32_t page) const;

    __forceinline__
    __device__
    void evict_all_to_hc();

};


__device__ void read_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);
__device__ void write_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);

__forceinline__
__device__
uint64_t get_backing_page_(const uint64_t page_start, const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist) {
    uint64_t page = page_start;
    if (dist == STRIPE) {
        page += page_offset / n_ctrls;
    }
    else if (dist == REPLICATE) {
        page += page_offset;
    }

    return page;
}
__forceinline__
__device__
uint64_t get_backing_ctrl_(const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist) {
    uint64_t ctrl;

    if (dist == STRIPE) {
        ctrl = page_offset % n_ctrls;
    }
    else if (dist == REPLICATE) {
        ctrl = ALL_CTRLS;
    }
    return ctrl;

}

__global__
void __flush(page_cache_d_t* pc) {
    uint64_t page = threadIdx.x + blockIdx.x * blockDim.x;

    if (page < pc->n_pages) {
        uint64_t previous_global_address = pc->cache_pages[page].page_translation;
        //uint8_t previous_range = this->cache_pages[page].range_id;
        uint64_t previous_range = previous_global_address & pc->n_ranges_mask;
        uint64_t previous_address = previous_global_address >> pc->n_ranges_bits;
        //uint32_t new_state = BUSY;

        uint32_t expected_state = pc->ranges[previous_range][previous_address].state.load(simt::memory_order_relaxed);

        uint32_t d = expected_state & DIRTY;
        uint32_t smid = get_smid();
        if (d) {

            uint64_t ctrl = get_backing_ctrl_(previous_address, pc->n_ctrls, pc->ranges_dists[previous_range]);
            //uint64_t get_backing_page(const uint64_t page_start, const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist) {
            uint64_t index = get_backing_page_(pc->ranges_page_starts[previous_range], previous_address, pc->n_ctrls, pc->ranges_dists[previous_range]);
            // //printf("Eviciting range_id: %llu\tpage_id: %llu\tctrl: %llx\tindex: %llu\n",
            //        (unsigned long long) previous_range, (unsigned long long)previous_address,
            //        (unsigned long long) ctrl, (unsigned long long) index);
            if (ctrl == ALL_CTRLS) {
                for (ctrl = 0; ctrl < pc->n_ctrls; ctrl++) {
                    Controller* c = pc->d_ctrls[ctrl];
                    uint32_t queue = smid % (c->n_qps);
                    
                    // Chia-Hao: 070423
                    if ((expected_state & CACHED_BY_HOST) != 0) {
                        //printf("page[%u][%u] cached by host\n", previous_range, previous_address);
                    }

                    pc->evict_cnt->fetch_add(1, simt::memory_order_relaxed);
                    write_data(pc, (c->d_qps)+queue, (index*pc->n_blocks_per_page), pc->n_blocks_per_page, page);
                    c->write_io_counter.fetch_add(1, simt::memory_order_relaxed);
                }
            }
            else {

                Controller* c = pc->d_ctrls[ctrl];
                uint32_t queue = smid % (c->n_qps);

                //index = ranges_page_starts[previous_range] + previous_address;

                pc->evict_cnt->fetch_add(1, simt::memory_order_relaxed);

                write_data(pc, (c->d_qps)+queue, (index*pc->n_blocks_per_page), pc->n_blocks_per_page, page);
                c->write_io_counter.fetch_add(1, simt::memory_order_relaxed);
            }

            pc->ranges[previous_range][previous_address].state.fetch_and(~DIRTY);

        }
    }
}

// CHIA-HAO
//#if USE_HOST_CACHE
__global__
void __flush_to_hc(page_cache_d_t* pc) {
    uint64_t page = threadIdx.x + blockIdx.x * blockDim.x;

    if (page < pc->n_pages) {
        uint32_t previous_global_address = pc->cache_pages[page].page_translation;
        //uint8_t previous_range = this->cache_pages[page].range_id;
        uint32_t previous_range = previous_global_address & pc->n_ranges_mask;
        uint32_t previous_address = previous_global_address >> pc->n_ranges_bits;
        //uint32_t new_state = BUSY;

        uint32_t expected_state = pc->ranges[previous_range][previous_address].state.load(simt::memory_order_relaxed);

        uint32_t d = expected_state & DIRTY;
        uint32_t smid = get_smid();
        if (d) {

            uint64_t ctrl = get_backing_ctrl_(previous_address, pc->n_ctrls, pc->ranges_dists[previous_range]);
            //uint64_t get_backing_page(const uint64_t page_start, const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist) {
            uint64_t index = get_backing_page_(pc->ranges_page_starts[previous_range], previous_address, pc->n_ctrls, pc->ranges_dists[previous_range]);
            // //printf("Eviciting range_id: %llu\tpage_id: %llu\tctrl: %llx\tindex: %llu\n",
            //        (unsigned long long) previous_range, (unsigned long long)previous_address,
            //        (unsigned long long) ctrl, (unsigned long long) index);
            if (ctrl == ALL_CTRLS) {
                for (ctrl = 0; ctrl < pc->n_ctrls; ctrl++) {
                    Controller* c = pc->d_ctrls[ctrl];
                    uint32_t queue = smid % (c->n_qps);
    
                    pc->evict_cnt->fetch_add(1, simt::memory_order_relaxed);
                    write_data(pc, (c->d_qps)+queue, (index*pc->n_blocks_per_page), pc->n_blocks_per_page, page);
                    c->write_io_counter.fetch_add(1, simt::memory_order_relaxed);
                    //accessHostCache((void*)pc->get_cache_page_addr(page), ((((uint64_t)previous_range)<<32)|(uint64_t)previous_address), GPU_RW_EVICT_TO_HOST, &(pc->ranges[previous_range][previous_address].bid));

                    //int bid = 0;
                    //accessHostCache((void*)pc->get_cache_page_addr(page), GPU_RW_FETCH_FROM_HOST, &bid);
                    //pc->ranges[previous_range][previous_address].state.fetch_or(CACHED_BY_HOST, simt::memory_order_acquire);
                }
            }
            else {

                Controller* c = pc->d_ctrls[ctrl];
                uint32_t queue = smid % (c->n_qps);

                //index = ranges_page_starts[previous_range] + previous_address;

                pc->evict_cnt->fetch_add(1, simt::memory_order_relaxed);
                write_data(pc, (c->d_qps)+queue, (index*pc->n_blocks_per_page), pc->n_blocks_per_page, page);
                
                c->write_io_counter.fetch_add(1, simt::memory_order_relaxed);
                //accessHostCache((void*)pc->get_cache_page_addr(page), ((((uint64_t)previous_range)<<32) | (uint64_t)previous_address), GPU_RW_EVICT_TO_HOST, &(pc->ranges[previous_range][previous_address].bid));
                //pc->ranges[previous_range][previous_address].state.fetch_or(CACHED_BY_HOST, simt::memory_order_acquire);
            }

            pc->ranges[previous_range][previous_address].state.fetch_and(~DIRTY);

        }
    }
}
//#endif

struct page_cache_t {


    //void* d_pc;

    //BufferPtr prp2_list_buf;
    page_cache_d_t pdt;
    //void* d_pc;
    //BufferPtr prp2_list_buf;
    //bool prps;
    pages_t*   h_ranges;
    page_reuse_hist_t** h_ranges_reuse_hist;

    uint64_t* h_ranges_page_starts;
    data_dist_t* h_ranges_dists;
    page_cache_d_t* d_pc_ptr;

    DmaPtr pages_dma;
    DmaPtr prp_list_dma;
    BufferPtr prp1_buf;
    BufferPtr prp2_buf;
    BufferPtr cache_pages_buf;
    //BufferPtr page_translation_buf;
    //BufferPtr page_take_lock_buf;
    BufferPtr ranges_buf;
    BufferPtr ranges_reuse_hist_buf;
    BufferPtr pc_buff;
    BufferPtr d_ctrls_buff;
    BufferPtr ranges_page_starts_buf;
    BufferPtr ranges_dists_buf;

    BufferPtr page_ticket_buf;
    BufferPtr ctrl_counter_buf;
    BufferPtr q_head_buf;
    BufferPtr q_tail_buf;
    BufferPtr q_lock_buf;
    BufferPtr evict_cnt_buf;
    BufferPtr virt_time_buf;
    BufferPtr virt_time_sampled_buf;
    BufferPtr tier2_occupied_count_buf;
    BufferPtr simul_reqs_count_buf;
    BufferPtr total_pred_count_buf;
    BufferPtr accurate_count_buf;
    BufferPtr extra_reads_buf;

#if SPECIAL_EVICTION_FOR_CLEAN_PAGES
    BufferPtr dirty_pages_evicted_buf;
    BufferPtr clean_pages_evicted_buf;
    BufferPtr unique_page_evict_num_buf;
#endif

    // Chia-Hao: added on 070823
#if SPECIAL_EVICTION_FOR_CLEAN_PAGES
    uint64_t total_pages_num = 0;
#endif
    // Chia-Hao: prefetcher
    //Prefetcher* prefetcher;

    void print_reset_stats(void) {
        uint64_t v = 0;
        cuda_err_chk(cudaMemcpy(&v, pdt.extra_reads, sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaMemcpyDeviceToHost));

        cuda_err_chk(cudaMemset(pdt.extra_reads, 0, sizeof(simt::atomic<uint64_t, simt::thread_scope_device>)));

//        printf("Cache Extra Reads: %llu\n", v);
    }

    void print_stats() 
    {
        simt::atomic<uint64_t, simt::thread_scope_device> evict_cnt;
        #if GET_GOLDEN_REUSE_DISTANCE
        simt::atomic<uint64_t, simt::thread_scope_device> total_pred_cnt;
        simt::atomic<uint64_t, simt::thread_scope_device> accurate_cnt;
        #endif
        cuda_err_chk(cudaMemcpy(&pdt, d_pc_ptr, sizeof(page_cache_d_t), cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaMemcpy(&evict_cnt, pdt.evict_cnt, sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaMemcpyDeviceToHost));

        #if GET_GOLDEN_REUSE_DISTANCE
        cuda_err_chk(cudaMemcpy(&total_pred_cnt, pdt.total_pred_count, sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaMemcpy(&accurate_cnt, pdt.accurate_count, sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaMemcpyDeviceToHost));
        #endif
        std::cout << "*********************************" << std::endl;
        std::cout << "******* Page Cache Stats ********" << std::endl;
        std::cout << std::dec << "#Evicts : "  << evict_cnt.load(simt::memory_order_acquire)
            << std::endl;
        #if GET_GOLDEN_REUSE_DISTANCE
        std::cout << std::dec << "#Total Pred : "  << total_pred_cnt.load(simt::memory_order_acquire) << std::endl;
        std::cout << std::dec << "#Accurate : "  << accurate_cnt.load(simt::memory_order_acquire) << std::endl;
        std::cout << "Accuracy : "  << (double)accurate_cnt.load()/(double)total_pred_cnt.load()*100.0 << " %"<< std::endl;
        #endif
        std::cout << "*********************************" << std::endl;
    }


    void flush_cache() {
        size_t threads = 64;
        size_t n_blocks = (pdt.n_pages + threads - 1) / threads;

        __flush<<<n_blocks, threads>>>(d_pc_ptr);


    }
    
    void flush_cache_to_hc(cudaStream_t *kernelStream) {
        size_t threads = 64;
        size_t n_blocks = (pdt.n_pages + threads - 1) / threads;
        //fprintf(stderr, "%s\n", __func__);
        __flush_to_hc<<<n_blocks, threads, 0, *kernelStream>>>(d_pc_ptr);
        //__flush<<<n_blocks, threads, 0, *kernelStream>>>(d_pc_ptr);
    }


    template <typename T>
    void add_range(range_t<T>* range) {
        range->rdt.range_id  = pdt.n_ranges++;
        h_ranges[range->rdt.range_id] = range->rdt.pages;
        h_ranges_reuse_hist[range->rdt.range_id] = range->rdt.pages_reuse_hist;
        h_ranges_page_starts[range->rdt.range_id] = range->rdt.page_start;
        h_ranges_dists[range->rdt.range_id] = range->rdt.dist;
        cuda_err_chk(cudaMemcpy(pdt.ranges_page_starts, h_ranges_page_starts, pdt.n_ranges * sizeof(uint64_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(pdt.ranges, h_ranges, pdt.n_ranges* sizeof(pages_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(pdt.ranges_reuse_hist, h_ranges_reuse_hist, pdt.n_ranges* sizeof(page_reuse_hist_t*), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(pdt.ranges_dists, h_ranges_dists, pdt.n_ranges* sizeof(data_dist_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
    }

    page_cache_t(const uint64_t ps, const uint64_t np, const uint32_t cudaDevice, const Controller& ctrl, const uint64_t max_range, const std::vector<Controller*>& ctrls) {

        ctrl_counter_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_head_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_tail_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_lock_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        evict_cnt_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        virt_time_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        virt_time_sampled_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        tier2_occupied_count_buf = createBuffer(sizeof(simt::atomic<int64_t, simt::thread_scope_device>), cudaDevice);
#if ACCESS_HOST_CACHE_WITH_ZERO_COPY
        simul_reqs_count_buf = createBuffer(sizeof(simt::atomic<int64_t, simt::thread_scope_device>), cudaDevice);
#endif  

#if GET_GOLDEN_REUSE_DISTANCE
        total_pred_count_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        accurate_count_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
#endif
        extra_reads_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        pdt.ctrl_counter = (simt::atomic<uint64_t, simt::thread_scope_device>*)ctrl_counter_buf.get();
        pdt.page_size = ps;
        pdt.q_head = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_head_buf.get();
        pdt.q_tail = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_tail_buf.get();
        pdt.q_lock = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_lock_buf.get();
        pdt.evict_cnt = (simt::atomic<uint64_t, simt::thread_scope_device>*)evict_cnt_buf.get();
        //pdt.virt_timestamp = (simt::atomic<uint64_t, simt::thread_scope_device>*)evict_cnt_buf.get();
        pdt.virt_timestamp = (simt::atomic<uint64_t, simt::thread_scope_device>*)virt_time_buf.get();
        //pdt.virt_timestamp_sampled = (simt::atomic<uint64_t, simt::thread_scope_device>*)evict_cnt_buf.get();
        pdt.virt_timestamp_sampled = (simt::atomic<uint64_t, simt::thread_scope_device>*)virt_time_sampled_buf.get();
        pdt.tier2_occupied_count = (simt::atomic<int64_t, simt::thread_scope_device>*)tier2_occupied_count_buf.get();
#if ACCESS_HOST_CACHE_WITH_ZERO_COPY
        pdt.simul_reqs_count = (simt::atomic<int64_t, simt::thread_scope_device>*)simul_reqs_count_buf.get();
#endif
#if GET_GOLDEN_REUSE_DISTANCE
        pdt.total_pred_count = (simt::atomic<uint64_t, simt::thread_scope_device>*)total_pred_count_buf.get();
        pdt.accurate_count = (simt::atomic<uint64_t, simt::thread_scope_device>*)accurate_count_buf.get();
#endif
#if SPECIAL_EVICTION_FOR_CLEAN_PAGES
        dirty_pages_evicted_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        clean_pages_evicted_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        pdt.dirty_pages_evicted = (simt::atomic<uint64_t, simt::thread_scope_device>*)dirty_pages_evicted_buf.get();
        pdt.clean_pages_evicted = (simt::atomic<uint64_t, simt::thread_scope_device>*)clean_pages_evicted_buf.get();
        // Chia-Hao: 080623
        unique_page_evict_num_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        pdt.unique_page_evict_num = (simt::atomic<uint64_t, simt::thread_scope_device>*)unique_page_evict_num_buf.get();
#endif
        pdt.extra_reads = (simt::atomic<uint64_t, simt::thread_scope_device>*)extra_reads_buf.get();
        pdt.page_size_minus_1 = ps - 1;
        pdt.n_pages = np;
        pdt.ctrl_page_size = ctrl.ctrl->page_size;
        pdt.n_pages_minus_1 = np - 1;
        pdt.n_ctrls = ctrls.size();
        d_ctrls_buff = createBuffer(pdt.n_ctrls * sizeof(Controller*), cudaDevice);
        pdt.d_ctrls = (Controller**) d_ctrls_buff.get();
        pdt.n_blocks_per_page = (ps/ctrl.blk_size);
        pdt.n_cachelines_for_states = np/STATES_PER_CACHELINE;
        //pdt.rrpv_increment = 0;
        for (size_t k = 0; k < pdt.n_ctrls; k++)
            cuda_err_chk(cudaMemcpy(pdt.d_ctrls+k, &(ctrls[k]->d_ctrl_ptr), sizeof(Controller*), cudaMemcpyHostToDevice));
        //n_ctrls = ctrls.size();
        //d_ctrls_buff = createBuffer(n_ctrls * sizeof(Controller*), cudaDevice);
        //d_ctrls = (Controller**) d_ctrls_buff.get();
        //for (size_t k = 0; k < n_ctrls; k++)
        //    cuda_err_chk(cudaMemcpy(d_ctrls+k, &(ctrls[k]->d_ctrl_ptr), sizeof(Controller*), cudaMemcpyHostToDevice));

        pdt.range_cap = max_range;
        pdt.n_ranges = 0;
        pdt.n_ranges_bits = (max_range == 1) ? 1 : std::log2(max_range);
        pdt.n_ranges_mask = max_range-1;
        std::cout << "n_ranges_bits: " << std::dec << pdt.n_ranges_bits << std::endl;
        std::cout << "n_ranges_mask: " << std::dec << pdt.n_ranges_mask << std::endl;

        pdt.page_size_log = std::log2(ps);
        ranges_buf = createBuffer(max_range * sizeof(pages_t), cudaDevice);
        pdt.ranges = (pages_t*)ranges_buf.get();
        
        //ranges_reuse_hist_buf = createBuffer(max_range * sizeof(page_reuse_hist_t*), cudaDevice);
        //pdt.ranges_reuse_hist = (page_reuse_hist_t**)ranges_reuse_hist_buf.get();
        cudaMalloc(&(pdt.ranges_reuse_hist), max_range*sizeof(page_reuse_hist_t*));
        h_ranges = new pages_t[max_range];
        h_ranges_reuse_hist = new page_reuse_hist_t*[max_range];
        pdt.ring_count = 0;

        h_ranges_page_starts = new uint64_t[max_range];
        std::memset(h_ranges_page_starts, 0, max_range * sizeof(uint64_t));

        //pages_translation_buf = createBuffer(np * sizeof(uint32_t), cudaDevice);
        //pdt.page_translation = (uint32_t*)page_translation_buf.get();
        //page_translation_buf = createBuffer(np * sizeof(padded_struct_pc), cudaDevice);
        //page_translation = (padded_struct_pc*)page_translation_buf.get();

        //page_take_lock_buf = createBuffer(np * sizeof(padded_struct_pc), cudaDevice);
        //pdt.page_take_lock =  (padded_struct_pc*)page_take_lock_buf.get();

        cache_pages_buf = createBuffer(np * sizeof(cache_page_t), cudaDevice);
        pdt.cache_pages = (cache_page_t*)cache_pages_buf.get();

        ranges_page_starts_buf = createBuffer(max_range * sizeof(uint64_t), cudaDevice);
        pdt.ranges_page_starts = (uint64_t*) ranges_page_starts_buf.get();

        page_ticket_buf = createBuffer(1 * sizeof(padded_struct_pc), cudaDevice);
        pdt.page_ticket =  (padded_struct_pc*)page_ticket_buf.get();
        //std::vector<padded_struct_pc> tps(np, FREE);
        cache_page_t* tps = new cache_page_t[np];
        for (size_t i = 0; i < np; i++)
            tps[i].page_take_lock = FREE;
        cuda_err_chk(cudaMemcpy(pdt.cache_pages, tps, np*sizeof(cache_page_t), cudaMemcpyHostToDevice));
        delete tps;

        ranges_dists_buf = createBuffer(max_range * sizeof(data_dist_t), cudaDevice);
        pdt.ranges_dists = (data_dist_t*)ranges_dists_buf.get();
        h_ranges_dists = new data_dist_t[max_range];

        uint64_t cache_size = ps*np;
        this->pages_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(cache_size, 1UL << 16), cudaDevice);
        pdt.base_addr = (uint8_t*) this->pages_dma.get()->vaddr;
        std::cout << "pages_dma: " << std::hex << this->pages_dma.get()->vaddr << "\t" << this->pages_dma.get()->ioaddrs[0] << std::endl;
        std::cout << "HEREN\n";
        const uint32_t uints_per_page = ctrl.ctrl->page_size / sizeof(uint64_t);
        if ((pdt.page_size > (ctrl.ctrl->page_size * uints_per_page)) || (np == 0) || (pdt.page_size < ctrl.ns.lba_data_size))
            throw error(string("page_cache_t: Can't have such page size or number of pages"));
        if (ps <= this->pages_dma.get()->page_size) {
            std::cout << "Cond1\n";
            uint64_t how_many_in_one = ctrl.ctrl->page_size/ps;
            this->prp1_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp1 = (uint64_t*) this->prp1_buf.get();


            std::cout << np << " " << sizeof(uint64_t) << " " << how_many_in_one << " " << this->pages_dma.get()->n_ioaddrs <<std::endl;
            uint64_t* temp = new uint64_t[how_many_in_one *  this->pages_dma.get()->n_ioaddrs];
            std::memset(temp, 0, how_many_in_one *  this->pages_dma.get()->n_ioaddrs);
            if (temp == NULL)
                std::cout << "NULL\n";

            for (size_t i = 0; (i < this->pages_dma.get()->n_ioaddrs) ; i++) {
                for (size_t j = 0; (j < how_many_in_one); j++) {
                    temp[i*how_many_in_one + j] = ((uint64_t)this->pages_dma.get()->ioaddrs[i]) + j*ps;
                    //std::cout << std::dec << "\ti: " << i << "\tj: " << j << "\tindex: "<< (i*how_many_in_one + j) << "\t" << std::hex << (((uint64_t)this->pages_dma.get()->ioaddrs[i]) + j*ps) << std::dec << std::endl;
                }
            }
            cuda_err_chk(cudaMemcpy(pdt.prp1, temp, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            delete temp;
            //std::cout << "HERE1\n";
            //free(temp);
            //std::cout << "HERE2\n";
            pdt.prps = false;
        }

        else if ((ps > this->pages_dma.get()->page_size) && (ps <= (this->pages_dma.get()->page_size * 2))) {
            this->prp1_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp1 = (uint64_t*) this->prp1_buf.get();
            this->prp2_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp2 = (uint64_t*) this->prp2_buf.get();
            //uint64_t* temp1 = (uint64_t*) malloc(np * sizeof(uint64_t));
            uint64_t* temp1 = new uint64_t[np * sizeof(uint64_t)];
            std::memset(temp1, 0, np * sizeof(uint64_t));
            //uint64_t* temp2 = (uint64_t*) malloc(np * sizeof(uint64_t));
            uint64_t* temp2 = new uint64_t[np * sizeof(uint64_t)];
            std::memset(temp2, 0, np * sizeof(uint64_t));
            for (size_t i = 0; i < np; i++) {
                temp1[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i*2]);
                temp2[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i*2+1]);
            }
            cuda_err_chk(cudaMemcpy(pdt.prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(pdt.prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));

            delete temp1;
            delete temp2;
            pdt.prps = true;
        }
        else {
            this->prp1_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp1 = (uint64_t*) this->prp1_buf.get();
            uint32_t prp_list_size =  ctrl.ctrl->page_size  * np;
            this->prp_list_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(prp_list_size, 1UL << 16), cudaDevice);
            this->prp2_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp2 = (uint64_t*) this->prp2_buf.get();
            uint64_t* temp1 = new uint64_t[np * sizeof(uint64_t)];
            uint64_t* temp2 = new uint64_t[np * sizeof(uint64_t)];
            uint64_t* temp3 = new uint64_t[prp_list_size];
            std::memset(temp1, 0, np * sizeof(uint64_t));
            std::memset(temp2, 0, np * sizeof(uint64_t));
            std::memset(temp3, 0, prp_list_size);
            uint32_t how_many_in_one = ps /  ctrl.ctrl->page_size ;
            for (size_t i = 0; i < np; i++) {
                temp1[i] = ((uint64_t) this->pages_dma.get()->ioaddrs[i*how_many_in_one]);
                temp2[i] = ((uint64_t) this->prp_list_dma.get()->ioaddrs[i]);
                for(size_t j = 0; j < (how_many_in_one-1); j++) {
                    temp3[i*uints_per_page + j] = ((uint64_t) this->pages_dma.get()->ioaddrs[i*how_many_in_one + j + 1]);
                }
            }
            /*
              for (size_t i = 0; i < this->pages_dma.get()->n_ioaddrs; i+=how_many_in_one) {
              temp1[i/how_many_in_one] = ((uint64_t)this->pages_dma.get()->ioaddrs[i]);
              temp2[i/how_many_in_one] = ((uint64_t)this->prp_list_dma.get()->ioaddrs[i]);
              for (size_t j = 0; j < (how_many_in_one-1); j++) {

              temp3[(i/how_many_in_one)*uints_per_page + j] = ((uint64_t)this->pages_dma.get()->ioaddrs[i+1+j]);
              }
              }
            */

            std::cout << "Done creating PRP\n";
            cuda_err_chk(cudaMemcpy(pdt.prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(pdt.prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(this->prp_list_dma.get()->vaddr, temp3, prp_list_size, cudaMemcpyHostToDevice));

            delete temp1;
            delete temp2;
            delete temp3;
            pdt.prps = true;
        }


        pc_buff = createBuffer(sizeof(page_cache_d_t), cudaDevice);
        d_pc_ptr = (page_cache_d_t*)pc_buff.get();
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
        std::cout << "Finish Making Page Cache\n";

    }

    ~page_cache_t() {
        delete h_ranges;
        delete h_ranges_reuse_hist;
        delete h_ranges_page_starts;
        delete h_ranges_dists;
    }





};


template <typename T>
struct range_d_t {
    uint64_t index_start;
    uint64_t count;
    uint64_t range_id;
    uint64_t array_id;
    uint64_t page_start_offset;
    uint64_t page_size;
    uint64_t page_start;
    uint64_t page_count;
    size_t n_elems_per_page;
    data_dist_t dist;
    uint8_t* src;

    simt::atomic<uint64_t, simt::thread_scope_device> access_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> miss_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> hit_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> read_io_cnt;

    simt::atomic<uint64_t, simt::thread_scope_device> simultaneous_cnt;

    pages_t pages;
    page_reuse_hist_t* pages_reuse_hist;
    //padded_struct_pc* page_addresses;
    //uint32_t* page_addresses;
    //padded_struct_pc* page_vals;  //len = num of pages for data
    //void* self_ptr;
    page_cache_d_t cache;
    //range_d_t(range_t<T>* rt);
    __forceinline__
    __device__
    uint64_t get_backing_page(const size_t i) const;
    __forceinline__
    __device__
    uint64_t get_backing_ctrl(const size_t i) const;
    __forceinline__
    __device__
    uint64_t get_sector_size() const;
    __forceinline__
    __device__
    uint64_t get_page(const size_t i) const;
    __forceinline__
    __device__
    uint64_t get_subindex(const size_t i) const;
    __forceinline__
    __device__
    uint64_t get_global_address(const size_t page) const;
    __forceinline__
    __device__
    void release_page(const size_t pg) const;
    __forceinline__
    __device__
    void release_page(const size_t pg, const uint32_t count) const;
    __forceinline__
    __device__
    uint64_t acquire_page(const size_t pg, const uint32_t count, const bool write, const uint32_t ctrl, const uint32_t queue) ;
    // CHIA-HAO
    __forceinline__
    __device__
    uint64_t find_page(const size_t pg, const uint32_t count, const bool write, const uint32_t ctrl, const uint32_t queue, bool& nv_nb) ;
    /////////////
    __forceinline__
    __device__
    uint64_t find_page_without_data_transfer(const size_t pg, const uint32_t count, const bool write, const uint32_t ctrl_, const uint32_t queue, uint32_t& scheme);
    __forceinline__
    __device__
    void get_page_from_ssd(const size_t pg, const bool write, const uint32_t ctrl, const uint32_t queue, uint64_t page_trans);
    __forceinline__
    __device__
    void write_page_to_ssd(const size_t pg, const bool write, const uint32_t ctrl, const uint32_t queue, uint64_t page_trans);
    __forceinline__
    __device__
    void update_page_state(const uint64_t pg, uint32_t new_s);
    __forceinline__
    __device__
    void disable_page_state(const uint64_t pg, uint32_t state);
    __forceinline__
    __device__
    void write_done(const size_t pg, const uint32_t count) const;
    __forceinline__
    __device__
    T operator[](const size_t i) ;
    __forceinline__
    __device__
    void operator()(const size_t i, const T val);
    __forceinline__
    __device__
    cache_page_t* get_cache_page(const size_t pg) const;
    __forceinline__
    __device__
    uint64_t get_cache_page_addr(const uint32_t page_trans) const;
    __forceinline__
    __device__
    void mark_page_dirty(const size_t index);
};

template <typename T>
struct range_t {
    range_d_t<T> rdt;

    range_d_t<T>* d_range_ptr;
    page_cache_d_t* cache;

    BufferPtr pages_buff;
    BufferPtr pages_reuse_hist_buff;
    //BufferPtr page_addresses_buff;

    BufferPtr range_buff;


    #if APPLY_BURST_PROFILE
    range_t(uint64_t is, uint64_t count, uint64_t ps, uint64_t pc, uint64_t pso, uint64_t p_size, page_cache_t* c_h, uint32_t cudaDevice, data_dist_t dist = REPLICATE, uint64_t array_id = 0);
    #else
    range_t(uint64_t is, uint64_t count, uint64_t ps, uint64_t pc, uint64_t pso, uint64_t p_size, page_cache_t* c_h, uint32_t cudaDevice, data_dist_t dist = REPLICATE);
    #endif


};

template <typename T>
#if APPLY_BURST_PROFILE
range_t<T>::range_t(uint64_t is, uint64_t count, uint64_t ps, uint64_t pc, uint64_t pso, uint64_t p_size, page_cache_t* c_h, 
uint32_t cudaDevice, data_dist_t dist, uint64_t array_id) {
#else
range_t<T>::range_t(uint64_t is, uint64_t count, uint64_t ps, uint64_t pc, uint64_t pso, uint64_t p_size, page_cache_t* c_h, 
uint32_t cudaDevice, data_dist_t dist) {
#endif
    (void)p_size;
    rdt.access_cnt = 0;
    rdt.miss_cnt = 0;
    rdt.hit_cnt = 0;
    rdt.read_io_cnt = 0;
    rdt.index_start = is;
    rdt.count = count;
    //range_id = (c_h->range_count)++;
    rdt.page_start = ps;
    rdt.page_count = pc;
    rdt.page_size = c_h->pdt.page_size;
    rdt.page_start_offset = pso;
    rdt.dist = dist;
    size_t s = pc;//(rdt.page_end-rdt.page_start);//*page_size / c_h->page_size;
    rdt.n_elems_per_page = rdt.page_size / sizeof(T);
    cache = (page_cache_d_t*) c_h->d_pc_ptr;
    pages_buff = createBuffer(s * sizeof(data_page_t), cudaDevice);
    rdt.pages = (pages_t) pages_buff.get();
    
    pages_reuse_hist_buff = createBuffer(s * sizeof(page_reuse_hist_t), cudaDevice);
    rdt.pages_reuse_hist = (page_reuse_hist_t*) pages_reuse_hist_buff.get();

    //std::vector<padded_struct_pc> ts(s, INVALID);
    data_page_t* ts = new data_page_t[s];
    for (size_t i = 0; i < s; i++) {
        ts[i].state = INVALID;
    }
    ////printf("S value: %llu\n", (unsigned long long)s);
    cuda_err_chk(cudaMemcpy(rdt.pages//_states
                            , ts, s * sizeof(data_page_t), cudaMemcpyHostToDevice));
    delete ts;

    // CHIA-HAO
    #if APPLY_BURST_PROFILE
    uint64_t key;
    uint32_t* burst_count;
    uint32_t* reuse_dist;
    rdt.array_id = array_id;
    for (uint64_t page = 0; page < rdt.page_count; page++) {
        key = (array_id << 32) | page;
        burst_profile_t burst_profile_h;
        burst_profile_h.num_burst = page_bursts[key].size();
        burst_profile_h.curr_burst = 0;
        burst_count = (uint32_t*)malloc(sizeof(uint32_t)*burst_profile_h.num_burst);
        reuse_dist = (uint32_t*)malloc(sizeof(uint32_t)*burst_profile_h.num_burst);
        for (uint32_t i = 0; i < burst_profile_h.num_burst; i++) {
            burst_count[i] = page_bursts[key][i];
            reuse_dist[i] = reuse_dist_after_burst[key][i];
        }

        cuda_err_chk(cudaMalloc(&(burst_profile_h.burst_count), sizeof(uint32_t)*burst_profile_h.num_burst));
        cuda_err_chk(cudaMalloc(&(burst_profile_h.reuse_dist_after_burst), sizeof(uint32_t)*burst_profile_h.num_burst));
        cuda_err_chk(cudaMemcpy((void*)(burst_profile_h.burst_count), (void*)burst_count, sizeof(uint32_t)*burst_profile_h.num_burst, cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy((void*)(burst_profile_h.reuse_dist_after_burst), (void*)reuse_dist, sizeof(uint32_t)*burst_profile_h.num_burst, cudaMemcpyHostToDevice));
        //cuda_err_chk(cudaMemcpy(burst_profile_h.reuse_dist_after_burst, burst_count_profile[page].burst_count, sizeof(uint32_t)*burst_profile_h.num_burst, cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy((void*)((unsigned char*)(rdt.pages) + page*sizeof(data_page_t) + offsetof(data_page_t, burst_profile)), (void*)&(burst_profile_h), sizeof(burst_profile_t), cudaMemcpyHostToDevice));

        free(burst_count);
        //if (array_id == 0 && page == 0) {
        //    std::cout << "key " << key << ", num burst " << burst_profile_h.num_burst << ", burst_count " << burst_profile_h.burst_count << std::endl;
        //    printf("copy to %p\n", (unsigned char*)(rdt.pages) + 0);
        //}
    }
    std::cout << "Loaded Burst Profile..." << std::endl;
    #endif

    //page_addresses_buff = createBuffer(s * sizeof(uint32_t), cudaDevice);
    //rdt.page_addresses = (uint32_t*) page_addresses_buff.get();
    //page_addresses_buff = createBuffer(s * sizeof(padded_struct_pc), cudaDevice);
    //page_addresses = (padded_struct_pc*) page_addresses_buff.get();

    range_buff = createBuffer(sizeof(range_d_t<T>), cudaDevice);
    d_range_ptr = (range_d_t<T>*)range_buff.get();
    //rdt.range_id  = c_h->pdt.n_ranges++;


    cuda_err_chk(cudaMemcpy(d_range_ptr, &rdt, sizeof(range_d_t<T>), cudaMemcpyHostToDevice));
    
#if SPECIAL_EVICTION_FOR_CLEAN_PAGES
    c_h->total_pages_num += rdt.page_count;
    c_h->pdt.total_pages_num += rdt.page_count;
#endif
    c_h->add_range(this);

    rdt.cache = c_h->pdt;
    cuda_err_chk(cudaMemcpy(d_range_ptr, &rdt, sizeof(range_d_t<T>), cudaMemcpyHostToDevice));
    
}




template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_backing_page(const size_t page_offset) const {
    return get_backing_page_(page_start, page_offset, cache.n_ctrls, dist);
}




template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_backing_ctrl(const size_t page_offset) const {
    return get_backing_ctrl_(page_offset, cache.n_ctrls, dist);
}

template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_sector_size() const {
    return page_size;
}


template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_page(const size_t i) const {
    uint64_t index = ((i - index_start) * sizeof(T) + page_start_offset) >> (cache.page_size_log);
    return index;
}
template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_subindex(const size_t i) const {
    uint64_t index = ((i - index_start) * sizeof(T) + page_start_offset) & (cache.page_size_minus_1);
    return index;
}
template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_global_address(const size_t page) const {
    return ((page << cache.n_ranges_bits) | range_id);
}
template <typename T>
__forceinline__
__device__
void range_d_t<T>::release_page(const size_t pg) const {
    uint64_t index = pg;
    pages[index].state.fetch_sub(1, simt::memory_order_release);
}

template <typename T>
__forceinline__
__device__
void range_d_t<T>::release_page(const size_t pg, const uint32_t count) const {
    uint64_t index = pg;
    pages[index].state.fetch_sub(count, simt::memory_order_release);
}

template <typename T>
__forceinline__
__device__
cache_page_t* range_d_t<T>::get_cache_page(const size_t pg) const {
    uint32_t page_trans = pages[pg].offset;
    return cache.get_cache_page(page_trans);
}

template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_cache_page_addr(const uint32_t page_trans) const {
    return ((uint64_t)((cache.base_addr+(page_trans * cache.page_size))));
}

template <typename T>
__forceinline__
__device__
void range_d_t<T>::mark_page_dirty(const size_t index) {
    pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
}


template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::acquire_page(const size_t pg, const uint32_t count, const bool write, const uint32_t ctrl_, const uint32_t queue) {
    uint64_t index = pg;
    //uint32_t global_address = (index << cache.n_ranges_bits) | range_id;
    //access_cnt.fetch_add(count, simt::memory_order_relaxed);
    access_cnt.fetch_add(count, simt::memory_order_relaxed);
    bool fail = true;
    unsigned int ns = 8;
    //bool miss = false;
    //T ret;
    uint64_t j = 0;
    uint64_t read_state,st,st_new;
    uint64_t reuse_dist;
    read_state = pages[index].state.fetch_add(count, simt::memory_order_acquire);
    

#if USE_HOST_CACHE && ENABLE_TMM && GET_GOLDEN_REUSE_DISTANCE
    //if ((read_state & (VALID|BUSY)) == 0 && this->pages_reuse_hist[index].evicted_before) {
        //printf("gpu re-ref::: %lu\n", index);
    //    this->pages_reuse_hist[index].actual_reuse_dist_upon_re_ref = pushMemSampleToHost(((this->range_id<<32)|index), GPU_RW_SAMPLE_MEM_RE_REF);
    //}
    //if (((read_state >> (CNT_SHIFT+1)) & 0x03) != NV_NB)
        reuse_dist = pushMemSampleToHost(((this->range_id<<32)|index), GPU_RW_SAMPLE_MEM);
#endif

    // Timestamp...
    #if USE_HOST_CACHE && ENABLE_TMM
    //#if USE_HOST_CACHE
    uint64_t curr_virt_timestamp = cache.virt_timestamp->fetch_add(1, simt::memory_order_relaxed);
    uint64_t curr_virt_timestamp_sampled;

#if DYNAMIC_SAMPLE_FREQ
    if (cache.ring_count < 32 && curr_virt_timestamp && (curr_virt_timestamp % (*dynamic_sample_freq) == 0)) {
#else
    if (cache.ring_count < 32 && curr_virt_timestamp && (curr_virt_timestamp % SAMPLE_FREQ == 0)) {
    //if (cache.ring_count < 64 && curr_virt_timestamp && (curr_virt_timestamp % SAMPLE_FREQ == 0)) {
#endif
        curr_virt_timestamp_sampled = cache.virt_timestamp_sampled->fetch_add(1, simt::memory_order_relaxed);
        uint64_t timestamp_diff = curr_virt_timestamp_sampled-this->pages_reuse_hist[index].last_virt_timestamp_sampled+1;
        //printf("timestamp diff %lu (%lu - %lu)\n", timestamp_diff, curr_virt_timestamp_sampled, this->pages_reuse_hist[index].last_virt_timestamp_sampled);
        //uint64_t timestamp_diff = curr_virt_timestamp-this->pages_reuse_hist[index].curr_virt_timestamp+1;
        if (timestamp_diff >= MID_PREDICTOR_THRESHOLD && this->pages_reuse_hist[index].last_virt_timestamp_sampled) {
            #if !SPECIAL_EVICTION_FOR_CLEAN_PAGES
            //enqueue_mem_samples(mem_samples_ring_buf, ((this->range_id<<32)|index), timestamp_diff, NULL, &cache.ring_count);
            enqueue_mem_samples(mem_samples_ring_buf, ((this->range_id<<32)|index), curr_virt_timestamp, NULL, &cache.ring_count);
            #else 
            //enqueue_mem_samples_blocked(mem_samples_ring_buf, ((this->range_id<<32)|index), timestamp_diff, NULL, &cache.ring_count);
            enqueue_mem_samples_blocked(mem_samples_ring_buf, ((this->range_id<<32)|index), curr_virt_timestamp, NULL, &cache.ring_count);
            #endif
        }
        this->pages_reuse_hist[index].last_virt_timestamp_sampled = curr_virt_timestamp_sampled+1;
    }
    this->pages_reuse_hist[index].curr_virt_timestamp = curr_virt_timestamp;
    #endif   

    do {
        st = (read_state >> (CNT_SHIFT+1)) & 0x03;
        switch (st) {
            //invalid
        case NV_NB:
        {
            st_new = pages[index].state.fetch_or(BUSY, simt::memory_order_acquire);
            if ((st_new & BUSY) == 0) {
                
            #if USE_HOST_CACHE && ENABLE_TMM && GET_GOLDEN_REUSE_DISTANCE
                if (this->pages_reuse_hist[index].evicted_before) {
                    //printf("gpu re-ref::: %lu\n", index);
                    this->pages_reuse_hist[index].actual_reuse_dist_upon_re_ref = pushMemSampleToHost(((this->range_id<<32)|index), GPU_RW_SAMPLE_MEM_RE_REF);
    
                }
                //this->pages_reuse_hist[index].actual_reuse_dist_upon_re_ref = pushMemSampleToHost(((this->range_id<<32)|index), GPU_RW_SAMPLE_MEM);
            #endif

                uint32_t page_trans = cache.find_slot(index, range_id, queue);
                //fill in
                uint64_t ctrl = get_backing_ctrl(index);
                if (ctrl == ALL_CTRLS)
                    ctrl = cache.ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (cache.n_ctrls);
                //ctrl = ctrl_;
                uint64_t b_page = get_backing_page(index);
                Controller* c = cache.d_ctrls[ctrl];
                c->access_counter.fetch_add(1, simt::memory_order_relaxed);
                //read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
#if USE_HOST_CACHE
                if ((st_new & CACHED_BY_HOST) != 0x0) {
                    //printf("access host cache %lu: %u\n", ((uint64_t)range_id<<32)|index, st_new);
                    int ret = accessHostCache((void*)get_cache_page_addr(page_trans), ((range_id<<32)|index)/*key*/, GPU_RW_FETCH_FROM_HOST, &(pages[index].bid), false);
                    if (ret != HOST_CACHE_SUCCEEDED) {
                        // TODO
                        //printf("access host cache failed... %lu\n", ((uint64_t)range_id<<32)|index);
                        pages[index].state.fetch_xor(CACHED_BY_HOST, simt::memory_order_relaxed);
                        read_data(&cache, (c->d_qps)+queue, ((b_page)*cache.n_blocks_per_page), cache.n_blocks_per_page, page_trans);
                        read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
                    }
                    //else printf("access host cache succed... %lu\n", ((uint64_t)range_id<<32)|index);

                    // CHIA-HAO: regardless this fetching is successful or not, we will view tier-2 available space becomes more
                    cache.tier2_occupied_count->fetch_sub(1, simt::memory_order_acquire);
                }
                else {
#endif
                    //printf("fetch data from ssd (key %lu)\n", (range_id<<32)|index);
                    read_data(&cache, (c->d_qps)+queue, ((b_page)*cache.n_blocks_per_page), cache.n_blocks_per_page, page_trans);
                    read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
                    //printf("fetch data from ssd (key %lu) done\n", (range_id<<32)|index);
#if USE_HOST_CACHE
                }
#endif
                pages[index].offset = page_trans;
                miss_cnt.fetch_add(count, simt::memory_order_relaxed);
                if (write)
                    pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
                pages[index].state.fetch_xor(DISABLE_BUSY_ENABLE_VALID, simt::memory_order_release);

                //if (page_trans >= 32768) printf("base_master %lu\n", page_trans);
                return page_trans;

                fail = false;
            }
            break;
        }
            //valid
        case V_NB:
        {
            if (write && ((read_state & DIRTY) == 0))
                pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
            //uint32_t page_trans = pages[index].offset.load(simt::memory_order_acquire);
            uint32_t page_trans = pages[index].offset;
            hit_cnt.fetch_add(count, simt::memory_order_relaxed);
            #if APPLY_BURST_PROFILE
            pages[index].access_in_burst.fetch_add(1, simt::memory_order_relaxed);
            #endif

            return page_trans;
            fail = false;

            break;
        }
        case NV_B:
        case V_B:
            //if (kernel_iter >= 1) printf("no need to fetch: %lu done\n", index);
        default:
            break;
        }
        if (fail) {
            //printf("failed : %lu (st %u)\n", (range_id<<32)|index, st);
            //if ((++j % 1000000) == 0)
            //    printf("failed to acquire_page: j: %llu\tcnt_shift+1: %llu\tpage: %llu\tread_state: %llx\tst: %llx\tst_new: %llx\n", (unsigned long long)j, (unsigned long long) (CNT_SHIFT+1), (unsigned long long) index, (unsigned long long)read_state, (unsigned long long)st, (unsigned long long)st_new);
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
            __nanosleep(ns);
            if (ns < 256) {
                ns *= 2;
            }
#endif
            read_state = pages[index].state.load(simt::memory_order_acquire);
        }

    } while (fail);
    return 0;
}

// CHIA-HAO: this function only addresses cache replacement and eviction 
template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::find_page(const size_t pg, const uint32_t count, const bool write, const uint32_t ctrl_, const uint32_t queue, bool& nv_nb_and_cached_by_hc) {
    uint64_t index = pg;
    access_cnt.fetch_add(count, simt::memory_order_relaxed);
    bool fail = true;
    int hc_ret = HOST_CACHE_SUCCEEDED;
    unsigned int ns = 8;
    uint64_t read_state, st, st_new;
    uint64_t reuse_dist;
    nv_nb_and_cached_by_hc = false;
    read_state = pages[index].state.fetch_add(count, simt::memory_order_acquire);

#if USE_HOST_CACHE && ENABLE_TMM && GET_GOLDEN_REUSE_DISTANCE
    reuse_dist = pushMemSampleToHost(((this->range_id<<32)|index), GPU_RW_SAMPLE_MEM);
#endif

#if USE_HOST_CACHE && ENABLE_TMM
    //#if USE_HOST_CACHE
    uint64_t curr_virt_timestamp = cache.virt_timestamp->fetch_add(1, simt::memory_order_relaxed);
    uint64_t curr_virt_timestamp_sampled;
#if DYNAMIC_SAMPLE_FREQ
    if (cache.ring_count < 32 && curr_virt_timestamp && (curr_virt_timestamp % (*dynamic_sample_freq) == 0)) {
#else
    #if MAX_SAMPLES_APPLIED
    if (cache.try_samples < MAX_TRY_SAMPLES && cache.ring_count < 32 && curr_virt_timestamp && (curr_virt_timestamp % SAMPLE_FREQ == 0)) {
    #else
    if (cache.ring_count < 16 && curr_virt_timestamp && (curr_virt_timestamp % SAMPLE_FREQ == 0)) {
    #endif
#endif
        // Time threshold here to avoid overheads
        //if (get_elapsed_seconds() > 420) {
        //    cache.ring_count = 32;
        //}

        curr_virt_timestamp_sampled = cache.virt_timestamp_sampled->fetch_add(1, simt::memory_order_acq_rel);
        uint64_t timestamp_diff = (curr_virt_timestamp_sampled <= this->pages_reuse_hist[index].last_virt_timestamp_sampled) ? 1 : curr_virt_timestamp_sampled-this->pages_reuse_hist[index].last_virt_timestamp_sampled+1;
        //uint64_t timestamp_diff = curr_virt_timestamp-this->pages_reuse_hist[index].curr_virt_timestamp+1;
        //printf("timestamp diff %lu (%lu - %lu)\n", timestamp_diff, curr_virt_timestamp_sampled, this->pages_reuse_hist[index].last_virt_timestamp_sampled);
        if (timestamp_diff >= MID_PREDICTOR_THRESHOLD && this->pages_reuse_hist[index].last_virt_timestamp_sampled) {
        //printf("timestamp diff %lu (%lu - %lu)\n", timestamp_diff, curr_virt_timestamp_sampled, this->pages_reuse_hist[index].last_virt_timestamp_sampled);
            //enqueue_mem_samples(mem_samples_ring_buf, ((uint64_t)(this->range_id<<32)|index), timestamp_diff, NULL, &cache.ring_count);
            //enqueue_mem_samples_blocked(mem_samples_ring_buf, ((this->range_id<<32)|index), timestamp_diff, NULL, &cache.ring_count);

#if !SPECIAL_EVICTION_FOR_CLEAN_PAGES
            enqueue_mem_samples(mem_samples_ring_buf, ((this->range_id<<32)|index), timestamp_diff, NULL, &cache.ring_count);
#else 
            //enqueue_mem_samples_blocked(mem_samples_ring_buf, ((this->range_id<<32)|index), timestamp_diff, NULL, &cache.ring_count);
            enqueue_mem_samples(mem_samples_ring_buf, ((this->range_id<<32)|index), timestamp_diff, NULL, &cache.ring_count);
#endif
        }
        else if (get_elapsed_seconds() > 120 && cache.ring_count < 2 && this->pages_reuse_hist[index].last_virt_timestamp_sampled) {
            //enqueue_mem_samples(mem_samples_ring_buf, ((this->range_id<<32)|index), timestamp_diff, NULL, &cache.ring_count);
            //printf("...");
        }
        #if MAX_SAMPLES_APPLIED
        cache.try_samples++;
        #endif
        // @0520: use max to update
        //this->pages_reuse_hist[index].last_virt_timestamp_sampled = curr_virt_timestamp_sampled+1;
        this->pages_reuse_hist[index].last_virt_timestamp_sampled = max(this->pages_reuse_hist[index].last_virt_timestamp_sampled, curr_virt_timestamp_sampled+1);
    }
    this->pages_reuse_hist[index].curr_virt_timestamp = curr_virt_timestamp;
#endif 

    do {
        st = (read_state >> (CNT_SHIFT+1)) & 0x03;
        //printf("--- st %x for page %u\n", st, (range_id<<32)|index);
        switch (st) {
        //invalid
        case NV_NB:
        {
            st_new = pages[index].state.fetch_or(BUSY, simt::memory_order_acquire);
            if ((st_new & BUSY) == 0) {
            #if USE_HOST_CACHE && ENABLE_TMM && GET_GOLDEN_REUSE_DISTANCE
                if (this->pages_reuse_hist[index].evicted_before) {
                    //printf("gpu re-ref::: %lu\n", index);
                    this->pages_reuse_hist[index].actual_reuse_dist_upon_re_ref = pushMemSampleToHost(((this->range_id<<32)|index), GPU_RW_SAMPLE_MEM_RE_REF);
    
                }
                //this->pages_reuse_hist[index].actual_reuse_dist_upon_re_ref = pushMemSampleToHost(((this->range_id<<32)|index), GPU_RW_SAMPLE_MEM);
            #endif

            #if GET_PAGE_FAULT_RATE
                atomicAdd((int*)&page_fault_count, 1);
                if ((clock() - page_fault_clock) >= 100000) {
                    page_fault_clock = clock();
                    printf("%lu %d\n", page_fault_clock, page_fault_count);
                }
            #endif

                uint32_t page_trans = cache.find_slot(index, range_id, queue);
                uint64_t ctrl = get_backing_ctrl(index);
                if (ctrl == ALL_CTRLS)
                    ctrl = cache.ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (cache.n_ctrls);
                uint64_t b_page = get_backing_page(index);
                Controller* c = cache.d_ctrls[ctrl];
                c->access_counter.fetch_add(1, simt::memory_order_relaxed);
                
                uint64_t simul_reqs = 0;
                if ((st_new & CACHED_BY_HOST) != 0x0 && count == 32) {
                    simul_reqs = simultaneous_cnt.fetch_add(1, simt::memory_order_acquire);
                    //printf("simul reqs %lu\n", simul_reqs);
                }

                //read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
            
                #define zc_cond(count, simul_reqs) ((count != 32) || (simul_reqs < 8))
                bool is_cached_page_dirty = false;
                //printf("simultaneous cnt %lu for page %lu\n", simultaneous_cnt.load(), (range_id<<32|index));
                //if ((st_new & CACHED_BY_HOST) != 0x0 && !(count == 32 && range_id == 0)) {
                if ((st_new & CACHED_BY_HOST) != 0x0 && zc_cond(count, simul_reqs)) {
                //if ((st_new & CACHED_BY_HOST) != 0x0) {
                    hc_ret = accessHostCache((void*)get_cache_page_addr(page_trans), ((range_id<<32)|index)/*key*/, GPU_RW_FETCH_FROM_HOST, &(pages[index].bid), false, &is_cached_page_dirty);
                    if (hc_ret != HOST_CACHE_SUCCEEDED) {
                        pages[index].state.fetch_xor(CACHED_BY_HOST, simt::memory_order_relaxed);
                        read_data(&cache, (c->d_qps)+queue, ((b_page)*cache.n_blocks_per_page), cache.n_blocks_per_page, page_trans);
                        read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
                    }
                }
                else if ((st_new & CACHED_BY_HOST) == 0x0) {
                    read_data(&cache, (c->d_qps)+queue, ((b_page)*cache.n_blocks_per_page), cache.n_blocks_per_page, page_trans);
                    //printf("read data from find_page: %lu\n", (range_id<<32)|index);
                    read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
                }
              
                
                pages[index].offset = page_trans;
                miss_cnt.fetch_add(count, simt::memory_order_relaxed);
                if (write)
                    pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
                
                if ((st_new & CACHED_BY_HOST) != 0x0 && count == 32) {
                    simultaneous_cnt.fetch_sub(1, simt::memory_order_acquire);
                }

                if ((st_new & CACHED_BY_HOST) == 0x0) {
                    pages[index].state.fetch_xor(DISABLE_BUSY_ENABLE_VALID, simt::memory_order_release);
                }
                else if ((st_new & CACHED_BY_HOST) != 0x0 && zc_cond(count, simul_reqs)) {
                    if (write || (hc_ret == HOST_CACHE_SUCCEEDED && is_cached_page_dirty)) {
                        pages[index].state.fetch_xor(DISABLE_BUSY_ENABLE_VALID | CACHED_BY_HOST, simt::memory_order_release);
                        // Chia-Hao: need to set the dirty bit again
                        pages[index].state.fetch_or(DIRTY, simt::memory_order_release);
                        //printf("make %lu (dirty) disable_busy_enable_valid (%x)\n", ((range_id<<32)|index), pages[index].state.load());
                    }
                    else {
                        pages[index].state.fetch_xor(DISABLE_BUSY_ENABLE_VALID, simt::memory_order_release);
                        // Chia-Hao: wierd...
                        //pages[index].state.fetch_xor(DISABLE_BUSY_ENABLE_VALID | CACHED_BY_HOST, simt::memory_order_release);
                        //printf("make %lu disable_busy_enable_valid (%x)\n", ((range_id<<32)|index), pages[index].state.load());
                    }
                    //printf("make %lu disable_busy_enable_valid (%x)\n", index, pages[index].state.load());
                }
                else {
                    nv_nb_and_cached_by_hc = true;
                }
                
            #if GET_PAGE_FAULT_RATE
                atomicSub((int*)&page_fault_count, 1);
            #endif
                //if ((st_new & CACHED_BY_HOST) != 0x0 && count == 32) {
                //    simultaneous_cnt.fetch_sub(1, simt::memory_order_acquire);
                //}
                return page_trans;

                fail = false;
            }
            break;
        }
            //valid
        case V_NB:
        {
            if (write && ((read_state & DIRTY) == 0))
                pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
            //uint32_t page_trans = pages[index].offset.load(simt::memory_order_acquire);
            uint32_t page_trans = pages[index].offset;
            // while (cache.page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
            //     __nanosleep(100);
            //hit_cnt.fetch_add(count, simt::memory_order_relaxed);
            hit_cnt.fetch_add(count, simt::memory_order_relaxed);
            #if APPLY_BURST_PROFILE
            pages[index].access_in_burst.fetch_add(1, simt::memory_order_relaxed);

            //if (pages[index].burst_profile.num_burst > 0) 
            //    printf("array %lu - index %lu - num_burst %u - burst_count %u\n", array_id, index, pages[index].burst_profile.num_burst, pages[index].burst_profile.burst_count[0]);
            #endif

            //cache.cache_pages[pages[index].offset].rrpv = 0;

            return page_trans;

            //pages[index].fetch_sub(1, simt::memory_order_release);
            fail = false;

            break;
        }
        case NV_B:
        case V_B:
            //if (kernel_iter >= 1) printf("no need to fetch: %lu done\n", index);
        default:
            break;
        }
        if (fail) {
            //printf("failed : %lu\n", (range_id<<32)|index);
            //if ((++j % 1000000) == 0)
            //    printf("failed to acquire_page: j: %llu\tcnt_shift+1: %llu\tpage: %llu\tread_state: %llx\tst: %llx\tst_new: %llx\n", (unsigned long long)j, (unsigned long long) (CNT_SHIFT+1), (unsigned long long) index, (unsigned long long)read_state, (unsigned long long)st, (unsigned long long)st_new);
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
            __nanosleep(ns);
            if (ns < 32) {
                ns *= 2;
            }
#endif
            read_state = pages[index].state.load(simt::memory_order_acquire);
        }

    } while (fail);
    return 0;
}


template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::find_page_without_data_transfer(const size_t pg, const uint32_t count, const bool write, const uint32_t ctrl_, const uint32_t queue, uint32_t& scheme) {
    uint64_t index = pg;
    access_cnt.fetch_add(count, simt::memory_order_relaxed);
    bool fail = true;
    int hc_ret = HOST_CACHE_SUCCEEDED;
    unsigned int ns = 8;
    uint64_t read_state, st, st_new;
    read_state = pages[index].state.fetch_add(count, simt::memory_order_acquire);
    do {
        st = (read_state >> (CNT_SHIFT+1)) & 0x03;
        //printf("--- st %x for page %u\n", st, (range_id<<32)|index);
        switch (st) {
        //invalid
        case NV_NB:
        {
            st_new = pages[index].state.fetch_or(BUSY, simt::memory_order_acquire);
            if ((st_new & BUSY) == 0) {

        //printf("--- st %x for page %u\n", st, (range_id<<32)|index);
                uint32_t page_trans = cache.find_slot(index, range_id, queue);
                uint64_t ctrl = get_backing_ctrl(index);
                if (ctrl == ALL_CTRLS)
                    ctrl = cache.ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (cache.n_ctrls);
                uint64_t b_page = get_backing_page(index);
                Controller* c = cache.d_ctrls[ctrl];
                c->access_counter.fetch_add(1, simt::memory_order_relaxed);
                
                uint64_t simul_reqs = 0x1 << 31;

                read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
            
                //#define ZC_THRESHOLD (8)
                if ((st_new & CACHED_BY_HOST) != 0x0/* && !(count >= 32 && simul_reqs >= ZC_THRESHOLD)*/) {
                    scheme = ZERO_COPY;
                }
                else if ((st_new & CACHED_BY_HOST) == 0x0) {
                    scheme = READ_FROM_SSD;
                }
              
                pages[index].offset = page_trans;
                if (write)
                    pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
                
                return page_trans;

                fail = false;
            }
            break;
        }
            //valid
        case V_NB:
        {
            if (write && ((read_state & DIRTY) == 0))
                pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
            uint32_t page_trans = pages[index].offset;
            hit_cnt.fetch_add(count, simt::memory_order_relaxed);
            #if APPLY_BURST_PROFILE
            pages[index].access_in_burst.fetch_add(1, simt::memory_order_relaxed);
            #endif

            return page_trans;

            fail = false;

            break;
        }
        case NV_B:
        case V_B:
            //if (kernel_iter >= 1) printf("no need to fetch: %lu done\n", index);
        default:
            break;
        }
        if (fail) {
            //printf("failed : %lu\n", (range_id<<32)|index);
            //if ((++j % 1000000) == 0)
            //    printf("failed to acquire_page: j: %llu\tcnt_shift+1: %llu\tpage: %llu\tread_state: %llx\tst: %llx\tst_new: %llx\n", (unsigned long long)j, (unsigned long long) (CNT_SHIFT+1), (unsigned long long) index, (unsigned long long)read_state, (unsigned long long)st, (unsigned long long)st_new);
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
            __nanosleep(ns);
            if (ns < 32) {
                ns *= 2;
            }
#endif
            read_state = pages[index].state.load(simt::memory_order_acquire);
        }

    } while (fail);
    return 0;
}


template <typename T>
__forceinline__
__device__
void range_d_t<T>::get_page_from_ssd(const size_t pg, const bool write, const uint32_t ctrl, const uint32_t queue, uint64_t page_trans)
{
    uint64_t index = pg;
    uint64_t b_page = get_backing_page(index);
    Controller* c = cache.d_ctrls[ctrl];
    read_data(&cache, (c->d_qps)+queue, ((b_page)*cache.n_blocks_per_page), cache.n_blocks_per_page, page_trans);
    read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
}

// Chia-Hao
template <typename T>
__forceinline__
__device__
void range_d_t<T>::write_page_to_ssd(const size_t pg, const bool write, const uint32_t ctrl, const uint32_t queue, uint64_t page_trans)
{
    uint64_t index = pg;
    uint64_t b_page = get_backing_page(index);
    Controller* c = cache.d_ctrls[ctrl];
    write_data(&cache, (c->d_qps)+queue, ((b_page)*cache.n_blocks_per_page), cache.n_blocks_per_page, page_trans);
    c->write_io_counter.fetch_add(1, simt::memory_order_relaxed);
}


template <typename T>
__forceinline__
__device__
void range_d_t<T>::update_page_state(const uint64_t pg, uint32_t new_s) {
    pages[pg].state.fetch_xor(new_s, simt::memory_order_release);
}

template <typename T>
__forceinline__
__device__
void range_d_t<T>::disable_page_state(const uint64_t pg, uint32_t state) {
    pages[pg].state.fetch_and(~state, simt::memory_order_release);
}

template<typename T>
struct array_d_t {
    uint64_t n_elems;
    uint64_t start_offset;
    uint64_t n_ranges;
    uint64_t array_idx;
    uint8_t *src;

    // CHIA-HAO

    range_d_t<T>* d_ranges;

    __forceinline__
    __device__
    void get_page_gid(const uint64_t i, range_d_t<T>*& r_, size_t& pg, size_t& gid) const {
        int64_t r = find_range(i);
        r_ = d_ranges+r;

        if (r != -1) {
            r_ = d_ranges+r;
            pg = r_->get_page(i);
            gid = r_->get_global_address(pg);
        }
        else {
            r_ = nullptr;
            printf("here\n");
        }
    }
    __forceinline__
    __device__
    void memcpy(const uint64_t i, const uint64_t count, T* dest) {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;

        uint32_t ctrl;
        uint32_t queue;

        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = 0xffffffff;
#endif
            uint32_t leader = 0;
            if (lane == leader) {
                page_cache_d_t* pc = &(r_->cache);
                ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
                queue = get_smid() % (pc->d_ctrls[ctrl]->n_qps);
            }
            ctrl = __shfl_sync(mask, ctrl, leader);
            queue = __shfl_sync(mask, queue, leader);

            uint64_t page = r_->get_page(i);
            //uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);
            //uint64_t p_s = r_->page_size;

            uint32_t active_cnt = 32;
            uint32_t eq_mask = mask;
            int master = 0;
            uint64_t base_master;
            uint64_t base;
            //bool memcpyflag_master;
            //bool memcpyflag;
            uint32_t count = 1;
            if (master == lane) {
                //std::pair<uint64_t, bool> base_memcpyflag;
                base = r_->acquire_page(page, count, false, ctrl, queue);
                base_master = base;
//                //printf("++tid: %llu\tbase: %p  page:%llu\n", (unsigned long long) threadIdx.x, base_master, (unsigned long long) page);
            }
            base_master = __shfl_sync(eq_mask,  base_master, master);

            //if (threadIdx.x == 63) {
            ////printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            //
            ulonglong4* src_ = (ulonglong4*) r_->get_cache_page_addr(base_master);
            ulonglong4* dst_ = (ulonglong4*) dest;
            warp_memcpy<ulonglong4>(dst_, src_, 512/32);

            __syncwarp(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);

        }

    }
    __forceinline__
    __device__
    int64_t find_range(const size_t i) const {
        int64_t range = -1;
        int64_t k = 0;
        for (; k < n_ranges; k++) {
            //printf("find range %ld [%lu - %lu] for index %ld\n", k, d_ranges[k].index_start, d_ranges[k].count, i);
            if ((d_ranges[k].index_start <= i) && (d_ranges[k].count > i)) {
                range = k;
                break;
            }

        }
        return range;
    }
    // Chia-Hao:
    __forceinline__
    __device__
    int zero_copy(const int64_t r, const uint64_t page, uint32_t eq_mask, const uint32_t lane, const int master, uint64_t base_master, bool dirty, uint32_t queue) const
    {
        ///*
        //printf("zero copy\n");
        auto r_ = d_ranges + r;
        int ret = HOST_CACHE_SUCCEEDED;
        uint32_t ctrl = 0;
        void* data_out = (void*)r_->get_cache_page_addr(base_master);
        void* data_in = get_host_cache_addr(r_->pages[page].bid);
        bool is_cached_page_dirty = false;

        if (master == lane) {
            ret = sendReqToHostCache(((r_->range_id<<32)|page), r_->pages[page].bid, GPU_RW_PIN_PAGE, &is_cached_page_dirty);
        }
        //ret = __shfl_sync(eq_mask, ret, master);
        ret = __shfl_sync(0xffffffff, ret, master);

        //
        if (ret == HOST_CACHE_PIN_FAILED) {
            // Pin failed, then use another way to get data
            if (master == lane) {
                r_->get_page_from_ssd(page, dirty, ctrl, queue, base_master);
                r_->update_page_state(page, (DISABLE_BUSY_ENABLE_VALID | CACHED_BY_HOST));
            }
            //__syncwarp(eq_mask);
            __syncwarp(0xffffffff);
        }
        else  
        {
            // Start coordinating threads to transfer
            //copyFromHostCache(data_in, data_out, r_->page_size, eq_mask);
            copyFromHostCacheVec(data_in, data_out, r_->page_size, 0xffffffff);

            // unlock the page
            if (master == lane) {
                ret = sendReqToHostCache(((r_->range_id<<32)|page), r_->pages[page].bid, GPU_RW_UNPIN_PAGE);
                if (dirty || is_cached_page_dirty) 
                    r_->update_page_state(page, DISABLE_BUSY_ENABLE_VALID | CACHED_BY_HOST);
                else {
                    // Chia-Hao: wierd..
                    //r_->update_page_state(page, DISABLE_BUSY_ENABLE_VALID | CACHED_BY_HOST);
                    r_->update_page_state(page, DISABLE_BUSY_ENABLE_VALID);
                }
            }
            //__syncwarp(eq_mask);
            __syncwarp(0xffffffff);
        }
        //*/
                //auto r_ = d_ranges + r;
                //if (master == lane) {
                //    r_->update_page_state(page, DISABLE_BUSY_ENABLE_VALID);
                //}
                //__syncwarp(eq_mask);
        //}
        return 0;
    }

    // Chia-Hao: new @070823
    __forceinline__
    __device__
    int evict_and_zero_copy(const int64_t r, const uint64_t page_out, const uint64_t page_in, uint32_t eq_mask, const uint32_t lane, const int master, uint64_t base_master, bool dirty, uint32_t queue) const
    {
        ///*
        //printf("zero copy\n");
        auto r_ = d_ranges + r;
        int ret = HOST_CACHE_SUCCEEDED;
        uint32_t ctrl = 0;
        void* destination = (void*)r_->get_cache_page_addr(base_master);
        void* data_in = get_host_cache_addr(r_->pages[page_in].bid);
        bool is_cached_page_dirty = false;

        // Eviction first
        if (master == lane) {
            ret = sendReqToHostCache(((r_->range_id<<32)|page_out), r_->pages[page_out].bid, GPU_RW_ALLOCATE_PAGE, &is_cached_page_dirty);
        }
        ret = __shfl_sync(0xffffffff, ret, master);
        //
        if (ret == HOST_CACHE_ALLOCATE_FAILED) {
            if (master == lane) {
                r_->write_page_to_ssd(page_out, dirty, ctrl, queue, base_master);
                r_->update_page_state(page_out, (DISABLE_BUSY_ENABLE_VALID));
            }
        }
        else  
        {
            // Start coordinating threads to transfer
            void* data_out = get_host_cache_addr(r_->pages[page_out].bid);
            copyFromHostCacheVec(destination, data_out, r_->page_size, 0xffffffff);
            if (master == lane) {
                r_->update_page_state(page_out, DISABLE_BUSY_ENABLE_VALID | CACHED_BY_HOST);
            }
        }
        __syncwarp(0xffffffff);


        // Bring new page
        if (master == lane) {
            ret = sendReqToHostCache(((r_->range_id<<32)|page_in), r_->pages[page_in].bid, GPU_RW_PIN_PAGE, &is_cached_page_dirty);
        }
        ret = __shfl_sync(0xffffffff, ret, master);

        if (ret == HOST_CACHE_PIN_FAILED) {
            // Pin failed, then use another way to get data
            if (master == lane) {
                r_->get_page_from_ssd(page_in, dirty, ctrl, queue, base_master);
                r_->update_page_state(page_in, (DISABLE_BUSY_ENABLE_VALID | CACHED_BY_HOST));
            }
            __syncwarp(0xffffffff);
        }
        else  
        {
            // Start coordinating threads to transfer
            copyFromHostCacheVec(data_in, destination, r_->page_size, 0xffffffff);

            // unlock the page
            if (master == lane) {
                ret = sendReqToHostCache(((r_->range_id<<32)|page_in), r_->pages[page_in].bid, GPU_RW_UNPIN_PAGE);
                if (dirty || is_cached_page_dirty) 
                    r_->update_page_state(page_in, DISABLE_BUSY_ENABLE_VALID | CACHED_BY_HOST);
                else {
                    // Chia-Hao: wierd..
                    //r_->update_page_state(page_in, DISABLE_BUSY_ENABLE_VALID | CACHED_BY_HOST);
                    r_->update_page_state(page_in, DISABLE_BUSY_ENABLE_VALID);
                }
            }
            __syncwarp(0xffffffff);
        }
        return 0;
    }

    
    __forceinline__
    __device__
    int zero_copy_test(const int64_t r, const uint64_t page, uint32_t eq_mask, const uint32_t lane, const int master, uint64_t base_master, bool dirty, uint32_t queue) const
    {
        ///*
        auto r_ = d_ranges + r;
        int ret = HOST_CACHE_SUCCEEDED;
        uint32_t ctrl = 0;
        void* data_in = (void*)r_->get_cache_page_addr(base_master);
        void* data_out = get_host_cache_addr(r_->pages[page].bid);
        
        //copyFromHostCache(data_in, data_out, r_->page_size, eq_mask);
        copyFromHostCacheVec(data_in, data_out, r_->page_size, 0xffffffff);
        if (master == lane) {
            r_->update_page_state(page, DISABLE_BUSY_ENABLE_VALID);
        }

        return 0;
    }



    __forceinline__
    __device__
    void coalesce_page_2(const uint32_t lane, const uint32_t mask, const int64_t r, const uint64_t page, const uint64_t gaddr, const bool write,
                       uint32_t& eq_mask, int& master, uint32_t& count, uint64_t& base_master) const {
        uint32_t ctrl;
        uint32_t queue;
        uint32_t leader = __ffs(mask) - 1;
        auto r_ = d_ranges+r;
        if (lane == leader) {
            page_cache_d_t* pc = &(r_->cache);
            ctrl = 0;//pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
            queue = get_smid() % (pc->d_ctrls[0]->n_qps);
        }

        ctrl = 0; //__shfl_sync(mask, ctrl, leader);
        queue = __shfl_sync(mask, queue, leader);

        uint32_t active_cnt = __popc(mask);
        eq_mask = __match_any_sync(mask, gaddr);
        eq_mask &= __match_any_sync(mask, (uint64_t)this);
        master = __ffs(eq_mask) - 1;

        uint32_t dirty = __any_sync(eq_mask, write);

        uint64_t base;
        count = __popc(eq_mask);
        

        uint32_t st = r_->pages[page].state.load(simt::memory_order_acquire); 
        st = __shfl_sync(eq_mask, st, master);
        auto range_id = r_->range_id;
        
        uint32_t scheme = 0;
        // check if the page valid and get the cache address if swap-in is needed
        if (master == lane) {
            //printf("find page for %lu (warp id %u, lane %u, master %u, logical tid %u, st %x, count %u)\n", (r_->range_id<<32)|page, warp_id(), lane, master, blockIdx.x*blockDim.x+threadIdx.x, st, count);
            base = r_->find_page_without_data_transfer(page, count, dirty, ctrl, queue, scheme);
            //base = r_->acquire_page(page, count, dirty, ctrl, queue);
            base_master = base;
        }
        // sync in the warp
        base_master = __shfl_sync(eq_mask, base_master, master);
        scheme = __shfl_sync(eq_mask, scheme, master);
        
        ///*
        if (scheme == ZERO_COPY) {
            zero_copy(r, page, eq_mask, lane, master, base_master, dirty, queue);
        }
        else if (scheme == READ_FROM_SSD) {
            if (master == lane) {     
                Controller* c = (r_->cache).d_ctrls[ctrl];
                uint64_t b_page = r_->get_backing_page(page);
                read_data(&(r_->cache), (c->d_qps)+queue, ((b_page)*r_->cache.n_blocks_per_page), r_->cache.n_blocks_per_page, base_master);
                //pages[index].state.fetch_xor(DISABLE_BUSY_ENABLE_VALID, simt::memory_order_release);
                r_->update_page_state(page, DISABLE_BUSY_ENABLE_VALID);
            }
            __syncwarp(eq_mask);
        }
    }


    __forceinline__
    __device__
    void coalesce_page(const uint32_t lane, const uint32_t mask, const int64_t r, const uint64_t page, const uint64_t gaddr, const bool write,
                       uint32_t& eq_mask, int& master, uint32_t& count, uint64_t& base_master) const {
        uint32_t ctrl;
        uint32_t queue;
        uint32_t leader = __ffs(mask) - 1;
        auto r_ = d_ranges+r;
        if (lane == leader) {
            page_cache_d_t* pc = &(r_->cache);
            ctrl = 0;//pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
            queue = get_smid() % (pc->d_ctrls[0]->n_qps);
        }

        ctrl = 0; //__shfl_sync(mask, ctrl, leader);
        queue = __shfl_sync(mask, queue, leader);


        uint32_t active_cnt = __popc(mask);
        eq_mask = __match_any_sync(mask, gaddr);
        eq_mask &= __match_any_sync(mask, (uint64_t)this);
        master = __ffs(eq_mask) - 1;

        uint32_t dirty = __any_sync(eq_mask, write);

        uint64_t base;
        //bool memcpyflag_master;
        //bool memcpyflag;
        count = __popc(eq_mask);
        
        // CHIA-HAO
#if USE_HOST_CACHE && LOGGING_RT_INFO 
        if (master == lane) loggingRtInfo((uint64_t)count);
#endif
        

#if USE_HOST_CACHE && ACCESS_HOST_CACHE_WITH_ZERO_COPY
        uint32_t st = r_->pages[page].state.load(simt::memory_order_acquire); 
        st = __shfl_sync(eq_mask, st, master);
        auto range_id = r_->range_id;
        //if (r_->range_id == 0 && page < 65536 && ((st&CACHED_BY_HOST) != 0)) {
        //    printf("st of %lu : %x (st & CACHED_BY_HOST: %u) (count %u)\n", (r_->range_id<<32)|page, st, st&CACHED_BY_HOST, count);
        //}
        
        //if (((st & CACHED_BY_HOST) != 0x0) && count == 32 && r_->cache.simul_reqs_count->load(simt::memory_order_acquire) > 8) {
        //if (false) {
        if (true) {
            //if (st != 0) printf("st of %lu : %u\n", (r_->range_id<<32)|page, st);
            bool nv_nb_and_cached_by_hc = false;
            // check if the page valid and get the cache address if swap-in is needed
            if (master == lane) {
                //printf("find page for %lu (warp id %u, lane %u, master %u, logical tid %u, st %x, count %u)\n", (r_->range_id<<32)|page, warp_id(), lane, master, blockIdx.x*blockDim.x+threadIdx.x, st, count);
                
                base = r_->find_page(page, count, dirty, ctrl, queue, nv_nb_and_cached_by_hc);
                //base = r_->acquire_page(page, count, dirty, ctrl, queue);
                
                //if (nv_nb_and_cached_by_hc) {
                //    printf("zero copy nv nb: [%lu][%lu] (state %x)\n", range_id, page, r_->pages[page].state);
                //}
                if (!nv_nb_and_cached_by_hc && dirty) {
                    // Chia-Hao: wierd...
                    r_->disable_page_state(page, CACHED_BY_HOST);
                }
                
                base_master = base;
            }
            // sync in the warp
            base_master = __shfl_sync(eq_mask, base_master, master);
            nv_nb_and_cached_by_hc = __shfl_sync(eq_mask, nv_nb_and_cached_by_hc, master);
            // if not valid and not busy, start copying from the host using zero-copy
            
            //int ret = (nv_nb_and_cached_by_hc) ? zero_copy(r, page, eq_mask, lane, master, base_master, dirty, queue) : 1;
            if (!nv_nb_and_cached_by_hc) {
            //if (false) {
                return;
            }//* reconvergence point? even if they dont exec the above code... */
            else {
                zero_copy(r, page, eq_mask, lane, master, base_master, dirty, queue);
                //zero_copy_test(r, page, eq_mask, lane, master, base_master, dirty, queue);
                //assert(false);
                return;
            }
        }
        else {
//#else
#endif
        ///*
            if (master == lane) {
                //std::pair<uint64_t, bool> base_memcpyflag;
                //printf("++tid: %llu\tbase: %p  page:%llu (r %lld)\n", (unsigned long long) threadIdx.x, base_master, (unsigned long long) page, r);
                base = r_->acquire_page(page, count, dirty, ctrl, queue);
                base_master = base;
                //printf("++tid: %llu\tbase: %p  page:%llu done (r %lld)\n", (unsigned long long) threadIdx.x, base_master, (unsigned long long) page, r);
                #if USE_HOST_CACHE && PROFILE
                loggingPageAction(r_->range_id, page, PAGE_ACTION_ACCESS);
                #endif

                //enqueue_mem_samples(mem_samples_ring_buf, (r_->range_id<<32|page), virt_time_diff, NULL);
            }
            base_master = __shfl_sync(eq_mask,  base_master, master);
        //*/
//#endif
#if USE_HOST_CACHE && ACCESS_HOST_CACHE_WITH_ZERO_COPY
        }
#endif
    }

    __forceinline__
    __device__
    returned_cache_page_t<T> get_raw(const size_t i) const {
        returned_cache_page_t<T> ret;
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;


        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);



            ret.addr = (T*) r_->get_cache_page_addr(base_master);
            ret.size = r_->get_sector_size()/sizeof(T);
            ret.offset = subindex/sizeof(T);
            //ret.page = page;
            __syncwarp(mask);


        }
        return ret;
    }
    __forceinline__
    __device__
    void release_raw(const size_t i) const {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;


        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            uint32_t active_cnt = __popc(mask);
            eq_mask = __match_any_sync(mask, gaddr);
            eq_mask &= __match_any_sync(mask, (uint64_t)this);
            master = __ffs(eq_mask) - 1;
            count = __popc(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);



        }
    }

    __forceinline__
    __device__
    void* acquire_page_(const size_t i, data_page_t*& page_, size_t& start, size_t& end, range_d_t<T>* r_, const size_t page) const {
        //uint32_t lane = lane_id();



        void* ret = nullptr;
        page_ = nullptr;
        if (r_) {
            //uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);
            page_cache_d_t* pc = &(r_->cache);
            uint32_t ctrl = 0;//pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
            uint32_t queue = get_smid() % (pc->d_ctrls[0]->n_qps);
            uint64_t base_master = r_->acquire_page(page, 1, false, ctrl, queue);
            //coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);

            page_ = &r_->pages[base_master];


            ret = (void*)r_->get_cache_page_addr(base_master);
            start = r_->n_elems_per_page * page;
            end = start +r_->n_elems_per_page;// * (page+1);
            //ret.page = page;

        }
        return ret;
    }
    __forceinline__
    __device__
    void* acquire_page(const size_t i, data_page_t*& page_, size_t& start, size_t& end, int64_t& r) const {
        uint32_t lane = lane_id();
        r = find_range(i);
        auto r_ = d_ranges+r;

        void* ret = nullptr;
        page_ = nullptr;
        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);
            //page_ = &r_->pages[base_master];
            page_ = &r_->pages[page];


            ret = (void*)r_->get_cache_page_addr(base_master);
            start = r_->n_elems_per_page * page + r_->index_start;
            end = start +r_->n_elems_per_page;// * (page+1);
            //ret.page = page;
            __syncwarp(mask);
        }
        return ret;
    }

    __forceinline__
    __device__
    void release_page(data_page_t* page_, const int64_t r, const size_t i) const {
        uint32_t lane = lane_id();
        auto r_ = d_ranges+r;

        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t gaddr = r_->get_global_address(page);

            uint32_t active_cnt = __popc(mask);
            eq_mask = __match_any_sync(mask, gaddr);
            eq_mask &= __match_any_sync(mask, (uint64_t)this);
            master = __ffs(eq_mask) - 1;
            count = __popc(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);



        }
    }

    __forceinline__
    __device__
    T seq_read(const size_t i) const {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;
        T ret;
        
        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);
            //coalesce_page_2(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);

            //if (threadIdx.x == 63) {
            //printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            #if PROFILE
            //loggingPageAction(0, page, PAGE_ACTION_ACCESS);
            #endif

            ret = ((T*)(r_->get_cache_page_addr(base_master)+subindex))[0];
            __syncwarp(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);

        }
        return ret;
    }
    __forceinline__
    __device__
    T* get_ptr(const size_t i) const {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;
        T* ret;
        
        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);
            //coalesce_page_2(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);

            //if (threadIdx.x == 63) {
            //printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            #if PROFILE
            //loggingPageAction(0, page, PAGE_ACTION_ACCESS);
            #endif

            ret = ((T*)(r_->get_cache_page_addr(base_master)+subindex));
            __syncwarp(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);

        }
        return ret;
    }


    __forceinline__
    __device__
    void seq_write(const size_t i, const T val) const {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;


        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, true, eq_mask, master, count, base_master);
            //coalesce_page_2(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);

            //if (threadIdx.x == 63) {
            ////printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            ((T*)(r_->get_cache_page_addr(base_master)+subindex))[0] = val;
            __syncwarp(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);

        }
    }
    __forceinline__
    __device__
    T operator[](size_t i) const {
        return seq_read(i);
        // size_t k = 0;
        // bool found = false;
        // for (; k < n_ranges; k++) {
        //     if ((d_ranges[k].index_start <= i) && (d_ranges[k].index_end > i)) {
        //         found = true;
        //         break;
        //     }

        // }
        // if (found)
        //     return (((d_ranges[k]))[i-d_ranges[k].index_start]);
    }
    __forceinline__
    __device__
    void operator()(size_t i, T val) const {
        seq_write(i, val);
        // size_t k = 0;
        // bool found = false;
        // uint32_t mask = __activemask();
        // for (; k < n_ranges; k++) {
        //     if ((d_ranges[k].index_start <= i) && (d_ranges[k].index_end > i)) {
        //         found = true;
        //         break;
        //     }
        // }
        // __syncwarp(mask);
        // if (found)
        //     ((d_ranges[k]))(i-d_ranges[k].index_start, val);
    }


    __forceinline__
    __device__
    T AtomicAdd(const size_t i, const T val) const {
        //uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;

        T old_val = 0;

        uint32_t ctrl;
        uint32_t queue;

        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t leader = __ffs(mask) - 1;
            if (lane == leader) {
                page_cache_d_t* pc = &(r_->cache);
                ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
                queue = get_smid() % (pc->d_ctrls[ctrl]->n_qps);
            }
            ctrl = __shfl_sync(mask, ctrl, leader);
            queue = __shfl_sync(mask, queue, leader);

            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);


            uint64_t gaddr = r_->get_global_address(page);
            //uint64_t p_s = r_->page_size;

            uint32_t active_cnt = __popc(mask);
            uint32_t eq_mask = __match_any_sync(mask, gaddr);
            eq_mask &= __match_any_sync(mask, (uint64_t)this);
            int master = __ffs(eq_mask) - 1;
            uint64_t base_master;
            uint64_t base;
            //bool memcpyflag_master;
            //bool memcpyflag;
            uint32_t count = __popc(eq_mask);
            if (master == lane) {
                base = r_->acquire_page(page, count, true, ctrl, queue);
                base_master = base;
                //    //printf("++tid: %llu\tbase: %llu  memcpyflag_master:%llu\n", (unsigned long long) threadIdx.x, (unsigned long long) base_master, (unsigned long long) memcpyflag_master);
            }
            base_master = __shfl_sync(eq_mask,  base_master, master);

            //if (threadIdx.x == 63) {
            ////printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            // ((T*)(base_master+subindex))[0] = val;
            old_val = atomicAdd((T*)(r_->get_cache_page_addr(base_master)+subindex), val);
            // //printf("AtomicAdd: tid: %llu\tpage: %llu\tsubindex: %llu\tval: %llu\told_val: %llu\tbase_master: %llx\n",
            //        (unsigned long long) tid, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) val,
            //     (unsigned long long) old_val, (unsigned long long) base_master);
            __syncwarp(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);
        }

        return old_val;
    }




};

template<typename T>
struct array_t {
    array_d_t<T> adt;

    //range_t<T>** d_ranges;
    array_d_t<T>* d_array_ptr;



    BufferPtr d_array_buff;
    BufferPtr d_ranges_buff;
    BufferPtr d_d_ranges_buff;

    void print_reset_stats(void) {
        std::vector<range_d_t<T>> rdt(adt.n_ranges);
        //range_d_t<T>* rdt = new range_d_t<T>[adt.n_ranges];
        cuda_err_chk(cudaMemcpy(rdt.data(), adt.d_ranges, adt.n_ranges*sizeof(range_d_t<T>), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < adt.n_ranges; i++) {

            std::cout << std::dec << "#READ IOs: "  << rdt[i].read_io_cnt 
                                  << "\t#Accesses:" << rdt[i].access_cnt
                                  << "\t#Misses:"   << rdt[i].miss_cnt 
                                  << "\tMiss Rate:" << ((float)rdt[i].miss_cnt/rdt[i].access_cnt)
                                  << "\t#Hits: "    << rdt[i].hit_cnt 
                                  << "\tHit Rate:"  << ((float)rdt[i].hit_cnt/rdt[i].access_cnt) 
                                  << "\tCLSize:"    << rdt[i].page_size 
                                  << std::endl;
            std::cout << "*********************************" << std::endl;
            rdt[i].read_io_cnt = 0;
            rdt[i].access_cnt = 0;
            rdt[i].miss_cnt = 0;
            rdt[i].hit_cnt = 0;
        }
        cuda_err_chk(cudaMemcpy(adt.d_ranges, rdt.data(), adt.n_ranges*sizeof(range_d_t<T>), cudaMemcpyHostToDevice));
    }

    array_t(const uint64_t num_elems, const uint64_t disk_start_offset, const std::vector<range_t<T>*>& ranges, uint32_t cudaDevice, uint64_t array_idx) {
        adt.n_elems = num_elems;
        adt.start_offset = disk_start_offset;
        adt.array_idx = array_idx;

        adt.n_ranges = ranges.size();
        d_array_buff = createBuffer(sizeof(array_d_t<T>), cudaDevice);
        d_array_ptr = (array_d_t<T>*) d_array_buff.get();

        //d_ranges_buff = createBuffer(n_ranges * sizeof(range_t<T>*), cudaDevice);
        d_d_ranges_buff = createBuffer(adt.n_ranges * sizeof(range_d_t<T>), cudaDevice);
        adt.d_ranges = (range_d_t<T>*)d_d_ranges_buff.get();
        //d_ranges = (range_t<T>**) d_ranges_buff.get();
        for (size_t k = 0; k < adt.n_ranges; k++) {
            //cuda_err_chk(cudaMemcpy(d_ranges+k, &(ranges[k]->d_range_ptr), sizeof(range_t<T>*), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(adt.d_ranges+k, (ranges[k]->d_range_ptr), sizeof(range_d_t<T>), cudaMemcpyDeviceToDevice));
        }

        cuda_err_chk(cudaMemcpy(d_array_ptr, &adt, sizeof(array_d_t<T>), cudaMemcpyHostToDevice));
    }

};

__forceinline__
__device__
cache_page_t* page_cache_d_t::get_cache_page(const uint32_t page) const {
    return &this->cache_pages[page];
}

// CHIA-HAO
__forceinline__
__device__
uint64_t page_cache_d_t::get_cache_page_addr(const uint32_t page) const {
    return ((uint64_t)((this->base_addr+(page * this->page_size))));
}

__forceinline__
__device__
void page_cache_d_t::evict_all_to_hc() 
{
    for (uint64_t page = 0; page < this->n_pages; page++) {
        uint32_t global_address = this->cache_pages[page].page_translation;
        uint32_t range = global_address & n_ranges_mask;
        uint32_t address = global_address >> n_ranges_bits;

        uint64_t state = this->ranges[range][address].state.load(simt::memory_order_acquire);
        uint64_t ctrl = get_backing_ctrl_(address, n_ctrls, ranges_dists[range]);
        uint64_t index = get_backing_page_(ranges_page_starts[range], address, n_ctrls, ranges_dists[range]);

        for (ctrl = 0; ctrl < n_ctrls; ctrl++) {
            Controller* c = this->d_ctrls[ctrl];
            uint32_t queue = 0;
            int ret;
            if (state & DIRTY) {
                ret = accessHostCache((void*)get_cache_page_addr(page), ((((uint64_t)range)<<32)|(uint64_t)address), 
                        GPU_RW_EVICT_TO_HOST, &(ranges[range][address].bid), true);
            }
            else {
                ret = accessHostCache((void*)get_cache_page_addr(page), ((((uint64_t)range)<<32)|(uint64_t)address), 
                        GPU_RW_EVICT_TO_HOST, &(ranges[range][address].bid), false);
            }

            if (ret == HOST_CACHE_SUCCEEDED) {
                this->ranges[range][address].state.fetch_or(CACHED_BY_HOST, simt::memory_order_acquire);
                this->ranges[range][address].state.fetch_and(~(VALID | BUSY), simt::memory_order_acquire);
                //printf("Evict Succ ([%u][%u])\n", range, address);
            }
            else {
                printf("Evict Failed!!! ([%u][%u])\n", range, address);
            }
        }
    }
}

__forceinline__
__device__
uint32_t page_cache_d_t::find_slot(uint64_t address, uint64_t range_id, const uint32_t queue_, int32_t* bid) {
    bool fail = true;
    uint64_t count = 0;
    uint32_t global_address =(uint32_t) ((address << n_ranges_bits) | range_id); //not elegant. but hack
    uint32_t page = 0;
    unsigned int ns = 8;
	//uint64_t j = 0;
    uint64_t expected_state = VALID;
    uint64_t new_expected_state = 0;
    //int64_t rrpv_increment_s = 0;

#if GET_PAGE_FAULT_RATE
    int cec = atomicAdd((int*)&concurrent_evict_count, 1);
    __threadfence();
    if ((clock() - evict_clock) >= 100000) {
    //if (true) {
        evict_clock = clock();
        printf("ev %lu %d\n", evict_clock, cec);
    }
#endif


    do {
        page = page_ticket->fetch_add(1, simt::memory_order_relaxed)  % (this->n_pages);
        bool lock = false;
        uint32_t v = this->cache_pages[page].page_take_lock.load(simt::memory_order_relaxed);

        // Update timestamp upon re-references
        //#if USE_HOST_CACHE
        #if USE_HOST_CACHE && ENABLE_TMM
        if (ranges_reuse_hist[range_id][address].evicted_before) {
            update_timestamp_upon_re_reference(&(ranges_reuse_hist[range_id][address]), this->virt_timestamp->load(simt::memory_order_relaxed));

            #if GET_GOLDEN_REUSE_DISTANCE
            uint64_t actual_remaining_reuse_dist = ranges_reuse_hist[range_id][address].actual_reuse_dist_upon_re_ref - ranges_reuse_hist[range_id][address].dist_from_ref_to_eviction;
            bool is_prediction_correct = getPredictionResult(actual_remaining_reuse_dist, ranges_reuse_hist[range_id][address].curr_predicted_tier); 
            print_reuse_history((ranges_reuse_hist[range_id][address]), ranges_reuse_hist[range_id][address].curr_predicted_tier);
            printf("actual remaining reuse dist of page %lu : %lu (correct? %d) (dist_on_re-ref %lu, dist_on_eviction %lu) (curr predicted tier %u, estimated_remaining_reuse_dist %lu, remaining_virt_timestamp_dist %lu) (reasons: %s)\n", ((((uint64_t)range_id)<<32)|(uint64_t)address), actual_remaining_reuse_dist, is_prediction_correct, ranges_reuse_hist[range_id][address].actual_reuse_dist_upon_re_ref, ranges_reuse_hist[range_id][address].dist_from_ref_to_eviction, ranges_reuse_hist[range_id][address].curr_predicted_tier, ranges_reuse_hist[range_id][address].estimated_remaining_reuse_dist, ranges_reuse_hist[range_id][address].remaining_virt_timestamp_dist, tier_prediction_reasons_str[ranges_reuse_hist[range_id][address].tier_prediction_reason]);
            if (is_prediction_correct) {
                accurate_count->fetch_add(1, simt::memory_order_relaxed);
            }
            total_pred_count->fetch_add(1, simt::memory_order_relaxed);
            printf("prediction accuracy so far ... %f\n", (float)accurate_count->load()/(float)total_pred_count->load());
            // */
            #endif

            //update_timestamp_upon_re_reference(&(ranges_reuse_hist[range_id][address]), this->virt_timestamp->load(simt::memory_order_relaxed));
            #if PRINT_TMM
            printf("[%lu][%lu] estimated reuse distance: %lu (tier %u) (virt timestamp diff %lu, slope %f, offset %f)\n", range_id, address, ranges_reuse_hist[range_id][address].estimated_remaining_reuse_dist, ranges_reuse_hist[range_id][address].last_predicted_tier, virt_timestamp->load(simt::memory_order_relaxed)-ranges_reuse_hist[range_id][address].last_eviction_virt_timestamp, curr_reg_info());
            #endif

            //if (ranges_reuse_hist[range_id][address].thrashing_count >= 2) {
            //    printf("[%lu][%lu] thrashing count %u\n", range_id, address, ranges_reuse_hist[range_id][address].thrashing_count);
            //}
        }
        #endif

        //not assigned to anyone yet
        if ( v == FREE ) {
            lock = this->cache_pages[page].page_take_lock.compare_exchange_weak(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if ( lock ) {
                this->cache_pages[page].page_translation = global_address;
                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
                fail = false;
            }
            //printf("thread %lu - FREE - find slot for %lu (v %u, lock %u)\n", (uint64_t)(blockDim.x*blockIdx.x+threadIdx.x), address, v, lock);
        }
        //assigned to someone and was able to take lock
        else if ( v == UNLOCKED ) {
            lock = this->cache_pages[page].page_take_lock.compare_exchange_weak(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if (lock) {
                uint32_t previous_global_address = this->cache_pages[page].page_translation;
                uint32_t previous_range = previous_global_address & n_ranges_mask;
                uint32_t previous_address = previous_global_address >> n_ranges_bits;
                expected_state = this->ranges[previous_range][previous_address].state.load(simt::memory_order_relaxed);

                uint32_t cnt = expected_state & CNT_MASK;
                uint32_t b = expected_state & BUSY;
                
                if ((cnt == 0) && (b == 0) ) {
                    new_expected_state = this->ranges[previous_range][previous_address].state.fetch_or(BUSY, simt::memory_order_acquire);
                    if (((new_expected_state & BUSY ) == 0) ) {
                        if (((new_expected_state & CNT_MASK ) == 0) ) {
                            if ((new_expected_state & DIRTY)) {
                                uint64_t ctrl = get_backing_ctrl_(previous_address, n_ctrls, ranges_dists[previous_range]);
                                uint64_t index = get_backing_page_(ranges_page_starts[previous_range], previous_address, n_ctrls, ranges_dists[previous_range]);
                                //printf("Eviciting range_id: %llu\tpage_id: %llu\tctrl: %llx\tindex: %llu\n",
                                //        (unsigned long long) previous_range, (unsigned long long)previous_address,
                                //        (unsigned long long) ctrl, (unsigned long long) index);
                                if (ctrl == ALL_CTRLS) {
                                    for (ctrl = 0; ctrl < n_ctrls; ctrl++) {
                                        Controller* c = this->d_ctrls[ctrl];
                                        uint32_t queue = queue_ % (c->n_qps);
                                        //evict_cnt->fetch_add(1, simt::memory_order_relaxed);
                                    
                                    #if USE_HOST_CACHE && ENABLE_TMM
                                        bool reuse_place_decision = false;
                                        uint32_t tier_to_evict = which_tier_to_evict(&(ranges_reuse_hist[previous_range][previous_address]), this->virt_timestamp->load(simt::memory_order_relaxed), reuse_place_decision, true);

                                        if (reuse_place_decision) {
                                            //printf("[%u][%u] decision: %u\n", previous_range, previous_address, tier_to_evict);
                                            tier_bins_d->bins[tier_to_evict]++;
                                            
                                            // Chia-Hao: 091323
                                            if (tier_to_evict == Tier3_SSD && page_stealing()) {
                                                tier_to_evict = Tier2_CPU;
                                            }
                                        }

                                        #if GET_GOLDEN_REUSE_DISTANCE
                                        uint64_t actual_reuse_dist = pushMemSampleToHost(((((uint64_t)previous_range)<<32)|(uint64_t)previous_address), GPU_RW_SAMPLE_MEM_EVICTION);
                                        ranges_reuse_hist[previous_range][previous_address].dist_from_ref_to_eviction = actual_reuse_dist;
                                        ranges_reuse_hist[previous_range][previous_address].curr_predicted_tier = tier_to_evict;
                                        
                                        #endif
                                    
                                    // 072223
                                    #if 1
                                        //printf("all pages evicted %lu - total pages num %lu\n", all_pages_evicted(), total_pages_num);
                                        if (!reuse_place_decision && 
                                        //if (
                                            tier_to_evict == Tier3_SSD && all_pages_evicted() >= (total_pages_num>>1)
                                        && ranges_reuse_hist[previous_range][previous_address].evicted_before > 0
                                        ) {
                                            if (!idle_slots_too_few(0))
                                                tier_to_evict = Tier2_CPU;
                                            //printf("Special Eviction for clean pages...\n");
                                        }

                                    #endif
                                    
                                    //#if GET_PAGE_FAULT_RATE
                                    #if 0
                                        int cec = atomicAdd((int*)&concurrent_evict_count, 1);
                                        __threadfence();
                                        //if ((clock() - evict_clock) >= 100000) {
                                        if (true) {
                                            evict_clock = clock();
                                            printf("ev %lu %d\n", evict_clock, cec);
                                        }
                                    #endif
                                        //printf("Evict [%u][%u] to %s\n", previous_range, previous_address, mem_tier_str[tier_to_evict]);
                                        //printf("tier to evict %u, idle_slots %ld\n", tier_to_evict, *num_idle_slots);
                                        if (tier_to_evict == Tier2_CPU && !idle_slots_too_few(0)) 
                                        //if (tier_to_evict == Tier2_CPU) 
                                        {
                                            int ret = accessHostCache((void*)get_cache_page_addr(page), ((((uint64_t)previous_range)<<32)|(uint64_t)previous_address), GPU_RW_EVICT_TO_HOST, &(ranges[previous_range][previous_address].bid), true);
                                            if (ret == HOST_CACHE_SUCCEEDED) {
                                                //printf("Cache by host %lu dirty\n", ((((uint64_t)previous_range)<<32)|(uint64_t)previous_address));
                                            #if PRINT_TMM
                                                printf("[Tier-1 GPU] [%u][%u] goes to Tier 2 (last %s)\n", previous_range, previous_address, mem_tier_str[ranges_reuse_hist[previous_range][previous_address].last_predicted_tier]);
                                            #endif
                                                this->ranges[previous_range][previous_address].state.fetch_or(CACHED_BY_HOST, simt::memory_order_acquire);
                                                this->tier2_occupied_count->fetch_add(1, simt::memory_order_acquire);
                                            }
                                            else {
                                                // TODO:
                                                write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                                                c->write_io_counter.fetch_add(1, simt::memory_order_relaxed);
                                            }
                                        }
                                        else if (tier_to_evict == Tier1_GPU) {
                                        #if PRINT_TMM
                                            printf("[Tier-1 GPU] [%u][%u] goes to Tier 1 (last %s)\n", previous_range, previous_address, mem_tier_str[ranges_reuse_hist[previous_range][previous_address].last_predicted_tier]);
                                        #endif
                                            ranges_reuse_hist[previous_range][previous_address].evict_attempt_num++;
                                            goto unlock_page;
                                        }
                                        // Chia-Hao: bug@070423
                                        else {
                                        //else {
                                    #endif
                                            write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                                            c->write_io_counter.fetch_add(1, simt::memory_order_relaxed);
                                    #if USE_HOST_CACHE && ENABLE_TMM
                                            #if PRINT_TMM
                                            printf("[Tier-1 GPU] [%u][%u] goes to Tier 3 (last %s) (evicted num %u)\n", previous_range, previous_address, mem_tier_str[ranges_reuse_hist[previous_range][previous_address].last_predicted_tier], ranges_reuse_hist[previous_range][previous_address].evicted_before);
                                            #endif
                                            
                                            // Disable HC state: 100123
                                            this->ranges[previous_range][previous_address].state.fetch_and(~CACHED_BY_HOST, simt::memory_order_release);
                                        }
                                        
                                        //#if GET_PAGE_FAULT_RATE
                                        #if 0
                                        atomicSub((int*)&concurrent_evict_count, 1);
                                        #endif
                                    #endif
                                    }
                                }
                                else {
                                    Controller* c = this->d_ctrls[ctrl];
                                    uint32_t queue = queue_ % (c->n_qps);
                                    //index = ranges_page_starts[previous_range] + previous_address;
                                    write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                                    c->write_io_counter.fetch_add(1, simt::memory_order_relaxed);
                                }
                                evict_cnt->fetch_add(1, simt::memory_order_relaxed);
                            #if SPECIAL_EVICTION_FOR_CLEAN_PAGES
                                dirty_pages_evicted->fetch_add(1, simt::memory_order_acquire);
                            #endif

                            }
                            #if USE_HOST_CACHE && ENABLE_TMM
                            else {
                                // clean page
                                ///*
                                bool reuse_place_decision = false;
                                uint32_t tier_to_evict = which_tier_to_evict(&(ranges_reuse_hist[previous_range][previous_address]), this->virt_timestamp->load(simt::memory_order_relaxed), reuse_place_decision);

                                if (reuse_place_decision) {
                                    tier_bins_d->bins[tier_to_evict]++;

                                    // Chia-Hao: 091323
                                    if (tier_to_evict == Tier3_SSD && page_stealing()) {
                                        tier_to_evict = Tier2_CPU;
                                    }
                                }
                                // get "real" reuse distance if the flag is enabled
                                #if GET_GOLDEN_REUSE_DISTANCE
                                uint64_t actual_reuse_dist = pushMemSampleToHost(((((uint64_t)previous_range)<<32)|(uint64_t)previous_address), GPU_RW_SAMPLE_MEM_EVICTION);
                                ranges_reuse_hist[previous_range][previous_address].dist_from_ref_to_eviction = actual_reuse_dist;
                                ranges_reuse_hist[previous_range][previous_address].curr_predicted_tier = tier_to_evict;
                                #endif

                                #if SPECIAL_EVICTION_FOR_CLEAN_PAGES
                                //printf("all pages evicted %lu - total pages num %lu\n", all_pages_evicted(), total_pages_num);

                                if (!reuse_place_decision && 
                                //if (
                                    tier_to_evict == Tier3_SSD && all_pages_evicted() >= (total_pages_num>>1) && dirty_pages_evicted->load() < special_handling_threshold()
                                    && ranges_reuse_hist[previous_range][previous_address].evicted_before > 0
                                ) {
                                    if (!idle_slots_too_few(0))
                                        tier_to_evict = Tier2_CPU;
                                    //printf("Special Eviction for clean pages...[%u][%u]\n", previous_range, previous_address);
                                }
                                // Chia-Hao: 080623
                                else if (reuse_place_decision && tier_to_evict == Tier3_SSD && unique_page_evict_num->load() > 549824){
                                    //tier_to_evict = Tier2_CPU;
                                }

                                clean_pages_evicted->fetch_add(1, simt::memory_order_acquire);
                                #endif
                                
                                //#if GET_PAGE_FAULT_RATE
                                #if 0
                                int cec = atomicAdd((int*)&concurrent_evict_count, 1);
                                __threadfence();
                                //if ((clock() - evict_clock) >= 100000) {
                                if (true) {
                                    evict_clock = clock();
                                    printf("ev %lu %d\n", evict_clock, cec);
                                }
                                #endif
                                //printf("Evict [%u][%u] to %s\n", previous_range, previous_address, mem_tier_str[tier_to_evict]);
                                //printf("tier to evict %u, idle_slots %ld\n", tier_to_evict, *num_idle_slots);
                                if (tier_to_evict == Tier2_CPU && !idle_slots_too_few(64)) {
                                //if (previous_address < 18432) {
                                    if ((expected_state & CACHED_BY_HOST) == 0x0) { 
                                        //printf("Cache by host (clean)%lu\n", ((((uint64_t)previous_range)<<32)|(uint64_t)previous_address));
                                        int ret = accessHostCache((void*)get_cache_page_addr(page), ((((uint64_t)previous_range)<<32)|(uint64_t)previous_address), GPU_RW_EVICT_TO_HOST, &(ranges[previous_range][previous_address].bid), false);
                                        //printf("accessHostCache done for %lu (bid %lu)\n", ((((uint64_t)previous_range)<<32)|(uint64_t)previous_address), ranges[previous_range][previous_address].bid);
                                        //int ret = HOST_CACHE_SUCCEEDED;
                                        if (ret == HOST_CACHE_SUCCEEDED) {
                                            this->ranges[previous_range][previous_address].state.fetch_or(CACHED_BY_HOST, simt::memory_order_acquire);
                                            //printf("New state is %u\n", this->ranges[previous_range][previous_address].state.load());
                                            this->tier2_occupied_count->fetch_add(1, simt::memory_order_acquire);
                                        }
                                    }
                                #if PRINT_TMM
                                    printf("[Tier-1 GPU] [%u][%u] (clean) goes to Tier 2 (last %s)\n", previous_range, previous_address, mem_tier_str[ranges_reuse_hist[previous_range][previous_address].last_predicted_tier]);
                                #endif

                                    //printf("tier 2 occupied count %lu (remain %lu)\n", this->tier2_occupied_count->load(simt::memory_order_acquire), second_tier_mem_pages-this->tier2_occupied_count->load(simt::memory_order_acquire));
                                }
                                else if (tier_to_evict == Tier1_GPU) {
                                #if PRINT_TMM
                                    printf("[Tier-1 GPU] [%u][%u] (clean) goes to Tier 1 (last %s)\n", previous_range, previous_address, mem_tier_str[ranges_reuse_hist[previous_range][previous_address].last_predicted_tier]);
                                #endif
                                    ranges_reuse_hist[previous_range][previous_address].evict_attempt_num++;
                                    goto unlock_page;
                                }
                                else {
                                #if PRINT_TMM
                                    printf("[%u][%u] discarded (clean)\n", previous_range, previous_address);
                                #endif
                                }
                                //*/

                                //#if GET_PAGE_FAULT_RATE
                                #if 0
                                atomicSub((int*)&concurrent_evict_count, 1);
                                #endif
                            }
                            #endif
                            
                            // Update timestamp upon evictions
                            //#if USE_HOST_CACHE
                            #if USE_HOST_CACHE && ENABLE_TMM
                            
                            // Chia-Hao: 080623
                            if (ranges_reuse_hist[previous_range][previous_address].evicted_before == 0)  {
                                unique_page_evict_num->fetch_add(1, simt::memory_order_relaxed);
                            }

                            // Chia-Hao: 091323
                            //if (tier_bins_d->bins[3] >= 10000 && (this->virt_timestamp->load() % 100000) >= 0 && (this->virt_timestamp->load() % 100000) < 100) {
                            //    printf("Tier1 %lu, Tier2 %lu, Tier3 %lu\n", tier_bins_d->bins[1], tier_bins_d->bins[2], tier_bins_d->bins[3]);
                            //}

                            update_timestamp_upon_eviction(&(ranges_reuse_hist[previous_range][previous_address]), this->virt_timestamp->load(simt::memory_order_relaxed));
                            #endif
                            //printf("[%lu][%lu] eviction timestamp: %lu\n", range_id, address, virt_timestamp->load(simt::memory_order_relaxed));
                            //printf("[key %lu] eviction timestamp: %lu\n", ((uint64_t)range_id<<32|(uint64_t)address), virt_timestamp->load(simt::memory_order_relaxed));

                            fail = false;
                            //this->ranges[previous_range][previous_address].state.fetch_and(CNT_MASK, simt::memory_order_release);
                            this->ranges[previous_range][previous_address].state.fetch_and(CNT_MASK | CACHED_BY_HOST, simt::memory_order_release);

                            #if USE_HOST_CACHE && PROFILE
                            loggingPageAction(previous_range, previous_address, PAGE_ACTION_EVICT);
                            #endif
                            
                            evict_cnt->fetch_add(1, simt::memory_order_relaxed);
                            
                            #if GET_PAGE_FAULT_RATE
                            //atomicSub((int*)&concurrent_evict_count, 1);
                            //__threadfence();
                            #endif
                        }
                        else { 
unlock_page:
                            this->ranges[previous_range][previous_address].state.fetch_and(DISABLE_BUSY_MASK, simt::memory_order_release);
                        }
                    }
                }
                if (!fail) {
                    this->cache_pages[page].page_translation = global_address;
                }
                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
            }
        }
        count++;

        if (fail) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
             __nanosleep(ns);
             if (ns < 256) {
                 ns *= 2;
             }
#endif
            //   if ((j % 10000000) == 0) {
            //     printf("failed to find slot j: %llu\taddr: %llx\tpage: %llx\texpected_state: %llx\tnew_expected_date: %llx\n", (unsigned long long) j, (unsigned long long) address, (unsigned long long)page, (unsigned long long) expected_state, (unsigned long long) new_expected_state);
//            }
//	   expected_state = 0;
//	   new_expected_state = 0;
        }
        
        //if (fail) printf("thread %lu - find slot for %lu (range %lu) (v %u, lock %u, new_expected_state %lx)\n", (uint64_t)(blockDim.x*blockIdx.x+threadIdx.x), address, range_id, v, lock, (new_expected_state & CNT_MASK));
    } while(fail);

#if GET_PAGE_FAULT_RATE
    atomicSub((int*)&concurrent_evict_count, 1);
    __threadfence();
#endif 

    return page;

}

__forceinline__
__device__
uint32_t page_cache_d_t::find_slot_prefetching(uint64_t address, uint64_t range_id, const uint32_t queue_, int32_t* bid) {
    bool fail = true;
    uint64_t count = 0;
    uint32_t global_address =(uint32_t) ((address << n_ranges_bits) | range_id); //not elegant. but hack
    uint32_t page = 0;
    unsigned int ns = 8;
	//uint64_t j = 0;
    uint64_t expected_state = VALID;
    uint64_t new_expected_state = 0;
    //int64_t rrpv_increment_s = 0;

    do {
        page = page_ticket->fetch_add(1, simt::memory_order_relaxed)  % (this->n_pages);
        bool lock = false;
        uint32_t v = this->cache_pages[page].page_take_lock.load(simt::memory_order_relaxed);

        //not assigned to anyone yet
        if ( v == FREE ) {
            lock = this->cache_pages[page].page_take_lock.compare_exchange_weak(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if ( lock ) {
                this->cache_pages[page].page_translation = global_address;
                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
                fail = false;
            }
            //printf("thread %lu - FREE - find slot for %lu (v %u, lock %u)\n", (uint64_t)(blockDim.x*blockIdx.x+threadIdx.x), address, v, lock);
        }
        //assigned to someone and was able to take lock
        else if ( v == UNLOCKED ) {
            lock = this->cache_pages[page].page_take_lock.compare_exchange_weak(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if (lock) {
                uint32_t previous_global_address = this->cache_pages[page].page_translation;
                uint32_t previous_range = previous_global_address & n_ranges_mask;
                uint32_t previous_address = previous_global_address >> n_ranges_bits;
                expected_state = this->ranges[previous_range][previous_address].state.load(simt::memory_order_relaxed);

                uint32_t cnt = expected_state & CNT_MASK;
                uint32_t b = expected_state & BUSY;
                
                if ((cnt == 0) && (b == 0) ) {
                    new_expected_state = this->ranges[previous_range][previous_address].state.fetch_or(BUSY, simt::memory_order_acquire);
                    if (((new_expected_state & BUSY ) == 0) ) {
                        if (((new_expected_state & CNT_MASK ) == 0) ) {
                            if ((new_expected_state & DIRTY)) {
                                uint64_t ctrl = get_backing_ctrl_(previous_address, n_ctrls, ranges_dists[previous_range]);
                                uint64_t index = get_backing_page_(ranges_page_starts[previous_range], previous_address, n_ctrls, ranges_dists[previous_range]);
                                //printf("Eviciting range_id: %llu\tpage_id: %llu\tctrl: %llx\tindex: %llu\n",
                                //        (unsigned long long) previous_range, (unsigned long long)previous_address,
                                //        (unsigned long long) ctrl, (unsigned long long) index);
                                if (ctrl == ALL_CTRLS) {
                                    for (ctrl = 0; ctrl < n_ctrls; ctrl++) {
                                        Controller* c = this->d_ctrls[ctrl];
                                        uint32_t queue = queue_ % (c->n_qps);
                                        //evict_cnt->fetch_add(1, simt::memory_order_relaxed);
                                    
                                    #if USE_HOST_CACHE && ENABLE_TMM
                                        bool reuse_place_decision = false;
                                        uint32_t tier_to_evict = which_tier_to_evict(&(ranges_reuse_hist[previous_range][previous_address]), this->virt_timestamp->load(simt::memory_order_relaxed), reuse_place_decision, true);

                                        if (reuse_place_decision) {
                                            //printf("[%u][%u] decision: %u\n", previous_range, previous_address, tier_to_evict);
                                            tier_bins_d->bins[tier_to_evict]++;
                                            
                                            // Chia-Hao: 091323
                                            if (tier_to_evict == Tier3_SSD && page_stealing()) {
                                                tier_to_evict = Tier2_CPU;
                                            }
                                        }
                                    
                                    // 072223
                                    #if 1
                                        //printf("all pages evicted %lu - total pages num %lu\n", all_pages_evicted(), total_pages_num);
                                        if (!reuse_place_decision && 
                                        //if (
                                            tier_to_evict == Tier3_SSD && all_pages_evicted() >= (total_pages_num>>1)
                                        && ranges_reuse_hist[previous_range][previous_address].evicted_before > 0
                                        ) {
                                            if (!idle_slots_too_few(0))
                                                tier_to_evict = Tier2_CPU;
                                            //printf("Special Eviction for clean pages...\n");
                                        }

                                    #endif
                                    
                                        //printf("Evict [%u][%u] to %s\n", previous_range, previous_address, mem_tier_str[tier_to_evict]);
                                        //printf("tier to evict %u, idle_slots %ld\n", tier_to_evict, *num_idle_slots);
                                        if (tier_to_evict == Tier2_CPU && !idle_slots_too_few(0)) 
                                        //if (tier_to_evict == Tier2_CPU) 
                                        {
                                            int ret = accessHostCache((void*)get_cache_page_addr(page), ((((uint64_t)previous_range)<<32)|(uint64_t)previous_address), GPU_RW_EVICT_TO_HOST, &(ranges[previous_range][previous_address].bid), true);
                                            if (ret == HOST_CACHE_SUCCEEDED) {
                                                //printf("Cache by host %lu dirty\n", ((((uint64_t)previous_range)<<32)|(uint64_t)previous_address));
                                            #if PRINT_TMM
                                                printf("[Tier-1 GPU] [%u][%u] goes to Tier 2 (last %s)\n", previous_range, previous_address, mem_tier_str[ranges_reuse_hist[previous_range][previous_address].last_predicted_tier]);
                                            #endif
                                                this->ranges[previous_range][previous_address].state.fetch_or(CACHED_BY_HOST, simt::memory_order_acquire);
                                                this->tier2_occupied_count->fetch_add(1, simt::memory_order_acquire);
                                            }
                                            else {
                                                // TODO:
                                                write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                                                c->write_io_counter.fetch_add(1, simt::memory_order_relaxed);
                                            }
                                        }
                                        else if (tier_to_evict == Tier1_GPU) {
                                        #if PRINT_TMM
                                            printf("[Tier-1 GPU] [%u][%u] goes to Tier 1 (last %s)\n", previous_range, previous_address, mem_tier_str[ranges_reuse_hist[previous_range][previous_address].last_predicted_tier]);
                                        #endif
                                            ranges_reuse_hist[previous_range][previous_address].evict_attempt_num++;
                                            goto unlock_page;
                                        }
                                        // Chia-Hao: bug@070423
                                        else {
                                        //else {
                                    #endif
                                            write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                                            c->write_io_counter.fetch_add(1, simt::memory_order_relaxed);
                                    #if USE_HOST_CACHE && ENABLE_TMM
                                            #if PRINT_TMM
                                            printf("[Tier-1 GPU] [%u][%u] goes to Tier 3 (last %s) (evicted num %u)\n", previous_range, previous_address, mem_tier_str[ranges_reuse_hist[previous_range][previous_address].last_predicted_tier], ranges_reuse_hist[previous_range][previous_address].evicted_before);
                                            #endif
                                            
                                            // Disable HC state: 100123
                                            this->ranges[previous_range][previous_address].state.fetch_and(~CACHED_BY_HOST, simt::memory_order_release);
                                        }
                                        
                                    #endif
                                    }
                                }
                                else {
                                    Controller* c = this->d_ctrls[ctrl];
                                    uint32_t queue = queue_ % (c->n_qps);
                                    //index = ranges_page_starts[previous_range] + previous_address;
                                    write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                                    c->write_io_counter.fetch_add(1, simt::memory_order_relaxed);
                                }
                                evict_cnt->fetch_add(1, simt::memory_order_relaxed);
                            #if SPECIAL_EVICTION_FOR_CLEAN_PAGES
                                dirty_pages_evicted->fetch_add(1, simt::memory_order_acquire);
                            #endif

                            }
                            #if USE_HOST_CACHE && ENABLE_TMM
                            else {
                                // clean page
                                ///*
                                bool reuse_place_decision = false;
                                uint32_t tier_to_evict = which_tier_to_evict(&(ranges_reuse_hist[previous_range][previous_address]), this->virt_timestamp->load(simt::memory_order_relaxed), reuse_place_decision);

                                if (reuse_place_decision) {
                                    tier_bins_d->bins[tier_to_evict]++;

                                    // Chia-Hao: 091323
                                    if (tier_to_evict == Tier3_SSD && page_stealing()) {
                                        tier_to_evict = Tier2_CPU;
                                    }
                                }
                                
                                #if SPECIAL_EVICTION_FOR_CLEAN_PAGES
                                if (!reuse_place_decision && 
                                //if (
                                    tier_to_evict == Tier3_SSD && all_pages_evicted() >= (total_pages_num>>1) && dirty_pages_evicted->load() < special_handling_threshold()
                                    && ranges_reuse_hist[previous_range][previous_address].evicted_before > 0
                                ) {
                                    if (!idle_slots_too_few(0))
                                        tier_to_evict = Tier2_CPU;
                                    //printf("Special Eviction for clean pages...[%u][%u]\n", previous_range, previous_address);
                                }
                                // Chia-Hao: 080623
                                else if (reuse_place_decision && tier_to_evict == Tier3_SSD && unique_page_evict_num->load() > 549824){
                                    //tier_to_evict = Tier2_CPU;
                                }

                                clean_pages_evicted->fetch_add(1, simt::memory_order_acquire);
                                #endif
                                
                                //printf("tier to evict %u, idle_slots %ld\n", tier_to_evict, *num_idle_slots);
                                if (tier_to_evict == Tier2_CPU && !idle_slots_too_few(64)) {
                                //if (previous_address < 18432) {
                                    if ((expected_state & CACHED_BY_HOST) == 0x0) { 
                                        //printf("Cache by host (clean)%lu\n", ((((uint64_t)previous_range)<<32)|(uint64_t)previous_address));
                                        int ret = accessHostCache((void*)get_cache_page_addr(page), ((((uint64_t)previous_range)<<32)|(uint64_t)previous_address), GPU_RW_EVICT_TO_HOST, &(ranges[previous_range][previous_address].bid), false);
                                        //printf("accessHostCache done for %lu (bid %lu)\n", ((((uint64_t)previous_range)<<32)|(uint64_t)previous_address), ranges[previous_range][previous_address].bid);
                                        //int ret = HOST_CACHE_SUCCEEDED;
                                        if (ret == HOST_CACHE_SUCCEEDED) {
                                            this->ranges[previous_range][previous_address].state.fetch_or(CACHED_BY_HOST, simt::memory_order_acquire);
                                            //printf("New state is %u\n", this->ranges[previous_range][previous_address].state.load());
                                            this->tier2_occupied_count->fetch_add(1, simt::memory_order_acquire);
                                        }
                                    }
                                #if PRINT_TMM
                                    printf("[Tier-1 GPU] [%u][%u] (clean) goes to Tier 2 (last %s)\n", previous_range, previous_address, mem_tier_str[ranges_reuse_hist[previous_range][previous_address].last_predicted_tier]);
                                #endif

                                    //printf("tier 2 occupied count %lu (remain %lu)\n", this->tier2_occupied_count->load(simt::memory_order_acquire), second_tier_mem_pages-this->tier2_occupied_count->load(simt::memory_order_acquire));
                                }
                                else if (tier_to_evict == Tier1_GPU) {
                                #if PRINT_TMM
                                    printf("[Tier-1 GPU] [%u][%u] (clean) goes to Tier 1 (last %s)\n", previous_range, previous_address, mem_tier_str[ranges_reuse_hist[previous_range][previous_address].last_predicted_tier]);
                                #endif
                                    ranges_reuse_hist[previous_range][previous_address].evict_attempt_num++;
                                    goto unlock_page;
                                }
                                else {
                                #if PRINT_TMM
                                    printf("[%u][%u] discarded (clean)\n", previous_range, previous_address);
                                #endif
                                }
                                //*/

                            }
                            #endif
                            
                            // Update timestamp upon evictions
                            //#if USE_HOST_CACHE
                            #if USE_HOST_CACHE && ENABLE_TMM
                            
                            // Chia-Hao: 080623
                            if (ranges_reuse_hist[previous_range][previous_address].evicted_before == 0)  {
                                unique_page_evict_num->fetch_add(1, simt::memory_order_relaxed);
                            }

                            // Chia-Hao: 091323
                            //if (tier_bins_d->bins[3] >= 10000 && (this->virt_timestamp->load() % 100000) >= 0 && (this->virt_timestamp->load() % 100000) < 100) {
                            //    printf("Tier1 %lu, Tier2 %lu, Tier3 %lu\n", tier_bins_d->bins[1], tier_bins_d->bins[2], tier_bins_d->bins[3]);
                            //}

                            update_timestamp_upon_eviction(&(ranges_reuse_hist[previous_range][previous_address]), this->virt_timestamp->load(simt::memory_order_relaxed));
                            #endif
                            //printf("[%lu][%lu] eviction timestamp: %lu\n", range_id, address, virt_timestamp->load(simt::memory_order_relaxed));
                            //printf("[key %lu] eviction timestamp: %lu\n", ((uint64_t)range_id<<32|(uint64_t)address), virt_timestamp->load(simt::memory_order_relaxed));

                            fail = false;
                            //this->ranges[previous_range][previous_address].state.fetch_and(CNT_MASK, simt::memory_order_release);
                            this->ranges[previous_range][previous_address].state.fetch_and(CNT_MASK | CACHED_BY_HOST, simt::memory_order_release);

                            
                            evict_cnt->fetch_add(1, simt::memory_order_relaxed);
                            
                        }
                        else { 
unlock_page:
                            this->ranges[previous_range][previous_address].state.fetch_and(DISABLE_BUSY_MASK, simt::memory_order_release);
                        }
                    }
                }
                if (!fail) {
                    this->cache_pages[page].page_translation = global_address;
                }
                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
            }
        }
        count++;

        if (fail) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
             __nanosleep(ns);
             if (ns < 256) {
                 ns *= 2;
             }
#endif
            //   if ((j % 10000000) == 0) {
            //     printf("failed to find slot j: %llu\taddr: %llx\tpage: %llx\texpected_state: %llx\tnew_expected_date: %llx\n", (unsigned long long) j, (unsigned long long) address, (unsigned long long)page, (unsigned long long) expected_state, (unsigned long long) new_expected_state);
//            }
//	   expected_state = 0;
//	   new_expected_state = 0;
        }
        
        //if (fail) printf("thread %lu - find slot for %lu (range %lu) (v %u, lock %u, new_expected_state %lx)\n", (uint64_t)(blockDim.x*blockIdx.x+threadIdx.x), address, range_id, v, lock, (new_expected_state & CNT_MASK));
    } while(fail);

    return page;
}



inline __device__ void poll_async(QueuePair* qp, uint16_t cid, uint16_t sq_pos) {
    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    //sq_dequeue(&qp->sq, sq_pos);

    cq_dequeue(&qp->cq, cq_pos, &qp->sq);



    put_cid(&qp->sq, cid);
}

inline __device__ void access_data_async(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, const uint8_t opcode, uint16_t * cid, uint16_t* sq_pos) {
    nvm_cmd_t cmd;
    *cid = get_cid(&(qp->sq));
    ////printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, *cid, opcode, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    ////printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    *sq_pos = sq_enqueue(&qp->sq, &cmd);



}

inline __device__ void enqueue_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, nvm_cmd_t* cmd, const uint16_t cid, const uint64_t pc_pos, const uint64_t pc_prev_head) {
    nvm_cmd_rw_blks(cmd, starting_lba, 1);
    unsigned int ns = 8;
    do {
        //check if new head past pc_pos
        //cur_pc_head == new head
        //prev_pc_head == old head
        //pc_pos == position i wanna move the head past
        uint64_t cur_pc_head = pc->q_head->load(simt::memory_order_relaxed);
        //sec == true when cur_pc_head past pc_pos
        bool sec = ((cur_pc_head < pc_prev_head) && (pc_prev_head <= pc_pos)) ||
            ((pc_prev_head <= pc_pos) && (pc_pos < cur_pc_head)) ||
            ((pc_pos < cur_pc_head) && (cur_pc_head < pc_prev_head));

        if (sec) break;

        //if not
        uint64_t qlv = pc->q_lock->load(simt::memory_order_relaxed);
        //uint64_t qlv = pc->q_lock->load(simt::memory_order_acquire);
        //got lock
        if (qlv == 0) {
            qlv = pc->q_lock->fetch_or(1, simt::memory_order_acquire);
            if (qlv == 0) {
                //printf("s-lba-start %llu\n", starting_lba);
                uint64_t cur_pc_tail;// = pc->q_tail.load(simt::memory_order_acquire);

                uint16_t sq_pos = sq_enqueue(&qp->sq, cmd, pc->q_tail, &cur_pc_tail);
                //printf("s-lba-after-sq %llu (sq_pos %u)\n", starting_lba, sq_pos);
                uint32_t head, head_;
                uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);

                //printf("s-lba-after-cqpoll %llu (sq_pos %u)\n", starting_lba, cq_pos);
                pc->q_head->store(cur_pc_tail, simt::memory_order_release);
                pc->q_lock->store(0, simt::memory_order_release);
                pc->extra_reads->fetch_add(1, simt::memory_order_relaxed);
                cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);

                //printf("s-lba-after-cq-deque %llu (sq_pos %u)\n", starting_lba, cq_pos);


                break;
            }
        }
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
         __nanosleep(ns);
         if (ns < 256) {
             ns *= 2;
         }
         //printf(".");
         //printf("s-lba-end %llu\n", starting_lba);
#endif
    } while(true);

}

inline __device__ void read_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);

    //printf("%s: starting_lba %lu\n", __func__, starting_lba);

    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    //if ((threadIdx.x+blockIdx.x*blockDim.x) == 262142)
    //    printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    //printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
    uint32_t head, head_;
    uint64_t pc_pos;
    uint64_t pc_prev_head;
//printf("start_lba: %llu, sq_pos %llu\n", starting_lba, sq_pos);
    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);
//printf("start_lba: %llu, cq_pos %llu\n", starting_lba, cq_pos);

    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    pc_prev_head = pc->q_head->load(simt::memory_order_relaxed);
    pc_pos = pc->q_tail->fetch_add(1, simt::memory_order_acq_rel);

    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
    //sq_dequeue(&qp->sq, sq_pos);

//printf("cq_deque : start_lba: %llu\n", starting_lba);

    //enqueue_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, nvm_cmd_t* cmd, const uint16_t cid, const uint64_t pc_pos, const uint64_t pc_prev_head)
    enqueue_second(pc, qp, starting_lba, &cmd, cid, pc_pos, pc_prev_head);



    put_cid(&qp->sq, cid);


    //printf("%s: starting_lba %lu done\n", __func__, starting_lba);
}


inline __device__ void write_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);



    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    ////printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, cid, NVM_IO_WRITE, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    ////printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
    uint32_t head, head_;
    __attribute__ ((unused)) uint64_t pc_pos;
    __attribute__ ((unused)) uint64_t pc_prev_head;

    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);
    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    pc_prev_head = pc->q_head->load(simt::memory_order_relaxed);
    pc_pos = pc->q_tail->fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
    //sq_dequeue(&qp->sq, sq_pos);




    put_cid(&qp->sq, cid);

}

inline __device__ void access_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, const uint8_t opcode) {
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);



    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    ////printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, cid, opcode, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    ////printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);

    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq);
    //sq_dequeue(&qp->sq, sq_pos);




    put_cid(&qp->sq, cid);


}



//#ifndef __CUDACC__
//#undef __device__
//#undef __host__
//#undef __forceinline__
//#endif


#endif // __PAGE_CACHE_H__
