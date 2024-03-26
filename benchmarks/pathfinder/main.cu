#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <assert.h>
#include <chrono>
#include <vector>
#include "settings.h"
#include <page_cache.h>
#include <ctrl.h>
#include <buffer.h>

#ifdef TIMING
#include "timing.h"

#include "host_cache.h"

struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0;
#endif

const char* const sam_ctrls_paths[] = {"/dev/libnvm_vmalloc0"};

typedef enum {
    BASELINE = 0,
    BASELINE_PC = 1,
} impl_type;

typedef enum {
    GPUMEM = 0,
    UVM_READONLY = 1,
    UVM_DIRECT = 2,
    UVM_READONLY_NVLINK = 3,
    UVM_DIRECT_NVLINK = 4,
    DRAGON_MAP = 5,
    BAFS_DIRECT = 6,
} mem_type;

using namespace std::chrono; 

#define BLOCK_SIZE 512
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

//#define BENCH_PRINT
typedef int64_t TYPE;
std::vector<Controller*> ctrls;
page_cache_t* h_pc;
range_t<TYPE>* h_Wall_range;
range_t<TYPE>* h_Results_range[2];
std::vector<range_t<TYPE>*> vec_Wall_range(1);
std::vector<range_t<TYPE>*> vec_Results_range_1(1);
std::vector<range_t<TYPE>*> vec_Results_range_2(1);
array_t<TYPE>* h_Wall_array;
array_t<TYPE>* h_Results_array[2];

TYPE *gpuWall, *gpuResult[2];
HostCache* hc;

void run(int argc, char** argv);

uint64_t rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
uint64_t pyramid_height;
Settings settings; 

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

void
init(int argc, char** argv)
{
	cols = settings.cols;
	rows = settings.rows;
    pyramid_height = settings.pyramid_height;
	
    data = new int[rows*cols];
	wall = new int*[rows];
	for(int n = 0; n < rows; n++) {
		wall[n] = data + cols*n;
    }
	result = new int[cols];
	
	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            wall[i][j] = rand() % 10;
        }
    }
#ifdef BENCH_PRINT
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ",wall[i][j]);
        }
        printf("\n");
    }
#endif

    printf("init finished\n");
}

void init_page_cache()
{
    cols = settings.cols;
	rows = settings.rows;
    pyramid_height = settings.pyramid_height;

    ctrls.resize(settings.n_ctrls);
    cuda_err_chk(cudaSetDevice(settings.cudaDevice));
    printf("Queue Depth %lu\n", settings.queueDepth);
    for (uint32_t i = 0; i < settings.n_ctrls; i++) {
        ctrls[i] = new Controller(sam_ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);
    }
    printf("Controllers Created.\n");

    uint64_t pc_page_size = settings.pageSize; 
    uint64_t pc_pages = ceil((float)settings.maxPageCacheSize/pc_page_size); 

    printf("Initialization done. \n");
    fflush(stdout);

    uint64_t wall_count = cols*rows;
    uint64_t n_wall_pages = (wall_count*sizeof(TYPE)+pc_page_size-1) / pc_page_size;
    uint64_t results_count = cols;
    uint64_t n_results_pages = (results_count*sizeof(TYPE)+pc_page_size-1) / pc_page_size;
    h_pc = new page_cache_t(pc_page_size, pc_pages, settings.cudaDevice, ctrls[0][0], (uint64_t)64, ctrls);
    
    h_Wall_range = new range_t<TYPE>(0, wall_count, 0, n_wall_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_Wall_range[0] = h_Wall_range;
    h_Wall_array = new array_t<TYPE>(wall_count, 0, vec_Wall_range, settings.cudaDevice, 0);

    h_Results_range[0] = new range_t<TYPE>(0, results_count, n_wall_pages, n_results_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_Results_range_1[0] = h_Results_range[0];
    h_Results_array[0] = new array_t<TYPE>(results_count, n_wall_pages*pc_page_size, vec_Results_range_1, settings.cudaDevice, 1);
    
    
    h_Results_range[1] = new range_t<TYPE>(0, results_count, n_wall_pages+n_results_pages, n_results_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_Results_range_2[0] = h_Results_range[1];
    h_Results_array[1] = new array_t<TYPE>(results_count, (n_wall_pages+n_results_pages)*pc_page_size, vec_Results_range_2, settings.cudaDevice, 2);

    printf("Total size %lu GB\n", (n_wall_pages + n_results_pages*2)*pc_page_size / (1024*1024*1024));
    printf("Page Cache Initialized\n");
    fflush(stdout);

#if USE_HOST_CACHE
    std::cerr << "creating Host Cache...\n";
    hc = createHostCache(ctrls[0], settings.maxPageCacheSize);

    uint64_t offset = 0;
    hc->registerRangesLBA(offset/512); offset += n_wall_pages*pc_page_size;
    hc->registerRangesLBA(offset/512); offset += n_results_pages*pc_page_size;
    hc->registerRangesLBA(offset/512); offset += n_results_pages*pc_page_size;
#endif

    size_t size = rows*cols;
    //cudaMalloc((void**)&gpuResult[0], sizeof(TYPE)*cols);
    //cudaMalloc((void**)&gpuResult[1], sizeof(TYPE)*cols);
    //cudaMalloc((void**)&gpuWall, sizeof(TYPE)*size);
}

void print_stats() {
    h_Wall_array->print_reset_stats();
    h_Results_array[0]->print_reset_stats();
    h_Results_array[1]->print_reset_stats();

    revokeHostRuntime();
}

void post_free()
{
    //cudaFree(gpuResult[0]);
    //cudaFree(gpuResult[1]);
    //cudaFree(gpuWall);
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void dynproc_kernel(
                int iteration, 
                TYPE *gpuWall,
                TYPE *gpuSrc,
                TYPE *gpuResults,
                uint64_t cols, 
                uint64_t rows,
                int startStep,
                int border)
{
    __shared__ TYPE prev[BLOCK_SIZE];
    __shared__ TYPE result[BLOCK_SIZE];

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	
    // each block finally computes result for a small block
    // after N iterations. 
    // it is the non-overlapping small blocks that cover 
    // all the input data

    // calculate the small block size
	int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

    // calculate the boundary for the block according to 
    // the boundary of its small block
    int blkX = small_block_cols*bx-border;
    int blkXmax = blkX+BLOCK_SIZE-1;

    // calculate the global thread coordination
	int xidx = blkX+tx;
       
    // effective range within this block that falls within 
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

    int W = tx-1;
    int E = tx+1;
        
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1)) {
        prev[tx] = gpuSrc[xidx];
	}
	__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    
    bool computed;
    for (int i=0; i<iteration ; i++){ 
        computed = false;
        if (IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) && isValid) {
            computed = true;
            TYPE left = prev[W];
            TYPE up = prev[tx];
            TYPE right = prev[E];
            TYPE shortest = MIN(left, up);
            shortest = MIN(shortest, right);
            uint64_t index = cols*(startStep+i)+xidx;
            result[tx] = shortest + gpuWall[index];
        }
        __syncthreads();
        if (i == iteration-1)
            break;
        if (computed)	 //Assign the computation range
            prev[tx] = result[tx];
        __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    }

    // update the global memory
    // after the last iteration, only threads coordinated within the 
    // small block perform the calculation and switch on ``computed''
    if (computed){
        gpuResults[xidx]=result[tx];		
    }
}

__global__ void dynproc_kernel_pc(
                int iteration, 
                array_d_t<TYPE> *d_gpuWall,
                array_d_t<TYPE> *d_gpuSrc,
                array_d_t<TYPE> *d_gpuResults,
                uint64_t cols, 
                uint64_t rows,
                uint64_t startStep,
                int border
)
{
    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result[BLOCK_SIZE];

	uint64_t bx = blockIdx.x;
	uint64_t tx = threadIdx.x;
	
    //if (threadIdx.x + blockIdx.x*blockDim.x == 0)printf("iteration %d, startStep %lu\n", iteration, startStep);
    // each block finally computes result for a small block
    // after N iterations. 
    // it is the non-overlapping small blocks that cover 
    // all the input data

    // calculate the small block size
	int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

    // calculate the boundary for the block according to 
    // the boundary of its small block
    int blkX = small_block_cols*bx-border;
    int blkXmax = blkX+BLOCK_SIZE-1;

    // calculate the global thread coordination
	int xidx = blkX+tx;
    
    if (blkX >= cols) return;
    // effective range within this block that falls within 
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validXmin = (blkX < 0) ? -blkX : 0;
    //int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;
    int validXmax = (blkXmax > cols-1) ? (blkXmax-cols+1 > BLOCK_SIZE-1) ?  : BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

    int W = tx-1;
    int E = tx+1;
        
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool isValid = IN_RANGE(tx, validXmin, validXmax);

    //if (blockIdx.x*blockDim.x+threadIdx.x == 0) printf("read src start\n");

	if(IN_RANGE(xidx, 0, cols-1)) {
        //prev[tx] = gpuSrc[xidx];
        prev[tx] = d_gpuSrc->seq_read(xidx);
	}
	__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    
    //if (blockIdx.x*blockDim.x+threadIdx.x == 0) printf("read src done\n");
    bool computed;
    for (int i = 0; i < iteration ; i++) { 
        computed = false;
        if (IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) && isValid) {
            computed = true;
            TYPE left = prev[W];
            TYPE up = prev[tx];
            TYPE right = prev[E];
            TYPE shortest = MIN(left, up);
            shortest = MIN(shortest, right);
            uint64_t index = cols*(startStep+i)+xidx;
            //result[tx] = shortest + gpuWall[index];
            result[tx] = shortest + d_gpuWall->seq_read(index);
        }
        __syncthreads();
        if (i == iteration-1)
            break;
        if (computed)	 //Assign the computation range
            prev[tx] = result[tx];
        __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    }

    //if (blockIdx.x*blockDim.x+threadIdx.x == 0) printf("iteration done\n");
    // update the global memory
    // after the last iteration, only threads coordinated within the 
    // small block perform the calculation and switch on ``computed''
    if (computed){
        //gpuResults[xidx] = result[tx];		
        d_gpuResults->seq_write(xidx, result[tx]);		
    }

    //__syncthreads();
    //if (blockIdx.x*blockDim.x+threadIdx.x == 0) printf("write results done\n");
}


/*
   compute N time steps
*/
int calc_path(TYPE *gpuWall, TYPE *gpuResult[2], uint64_t rows, uint64_t cols, uint64_t pyramid_height, int blockCols, int borderCols)
{
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);  
    

    auto start = std::chrono::high_resolution_clock::now();

    int src = 1, dst = 0;
    for (int t = 0; t < rows-1; t += pyramid_height) {
        int temp = src;
        src = dst;
        dst = temp;

        if (settings.memalloc != BAFS_DIRECT) {
            dynproc_kernel<<<dimGrid, dimBlock>>>(MIN(pyramid_height, rows-t-1), gpuWall, gpuResult[src], gpuResult[dst], cols, rows, t, borderCols);
        }
        else {
        #if USE_HOST_CACHE && ENABLE_TMM
            if ((t+pyramid_height) >= (rows-1)) { // last iteartion
                lastIteration(&(stream_mngr->kernel_stream));
            }
            dynproc_kernel_pc<<<dimGrid, dimBlock, 0, stream_mngr->kernel_stream>>>(MIN(pyramid_height, rows-t-1), h_Wall_array->d_array_ptr, h_Results_array[src]->d_array_ptr, h_Results_array[dst]->d_array_ptr, cols, rows, t, borderCols);
            cudaStreamSynchronize(stream_mngr->kernel_stream);
        #else
            dynproc_kernel_pc<<<dimGrid, dimBlock>>>(MIN(pyramid_height, rows-t-1), h_Wall_array->d_array_ptr, h_Results_array[src]->d_array_ptr, h_Results_array[dst]->d_array_ptr, cols, rows, t, borderCols);
            //cudaDeviceSynchronize();
        #endif
        }

        // for the measurement fairness
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
        printf("iter %d done..\n", t);
    }

    #if USE_HOST_CACHE
    printf("compute done!\n");
    h_pc->flush_cache(); 
    flushHostCache();
    cuda_err_chk(cudaDeviceSynchronize());
    #else
    printf("compute done!\n");
    h_pc->flush_cache(); 
    cuda_err_chk(cudaDeviceSynchronize());
    #endif

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << std::dec << "Elapsed Time: " << elapsed.count() << " ms"<< std::endl;

    return dst;
}

int main(int argc, char** argv)
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) cudaSetDevice(DEVICE);

    try
    {
        settings.parseArguments(argc, argv);
    }
    catch (const string& e)
    {
        fprintf(stderr, "%s\n", e.c_str());
        fprintf(stderr, "%s\n", Settings::usageString(argv[0]).c_str());
        return 1;
    }
    
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, settings.cudaDevice) != cudaSuccess)
    {
        fprintf(stderr, "Failed to get CUDA device properties\n");
        return 1;
    }


    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    if (settings.memalloc != BAFS_DIRECT) {
        init(argc, argv);
    }
    else {
        init_page_cache();
    }

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
	
    //TYPE *gpuWall, *gpuResult[2];
    size_t size = rows*cols;

    if (settings.memalloc != BAFS_DIRECT) {
        cudaMalloc((void**)&gpuResult[0], sizeof(TYPE)*cols);
        cudaMalloc((void**)&gpuResult[1], sizeof(TYPE)*cols);
        cudaMemcpy(gpuResult[0], data, sizeof(TYPE)*cols, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&gpuWall, sizeof(TYPE)*(size-cols));
        cudaMemcpy(gpuWall, data+cols, sizeof(TYPE)*(size-cols), cudaMemcpyHostToDevice);
    }

#ifdef  TIMING
    gettimeofday(&tv_kernel_start, NULL);
#endif

    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height, blockCols, borderCols);

#ifdef  TIMING
    gettimeofday(&tv_kernel_end, NULL);
    tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
    kernel_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    if (settings.memalloc != BAFS_DIRECT) {
        cudaMemcpy(result, gpuResult[final_ret], sizeof(int)*cols, cudaMemcpyDeviceToHost);
    }

#ifdef BENCH_PRINT
    for (int i = 0; i < cols; i++)
        printf("%d ",data[i]);
    printf("\n") ;
    for (int i = 0; i < cols; i++)
        printf("%d ",result[i]);
    printf("\n");
#endif
    
    //revokeHostRuntime();
    print_stats();


    if (settings.memalloc != BAFS_DIRECT) {
        cudaFree(gpuWall);
        cudaFree(gpuResult[0]);
        cudaFree(gpuResult[1]);

        delete [] data;
        delete [] wall;
        delete [] result;
    }
    else {
        post_free();
    }

#ifdef  TIMING
    printf("Exec: %f\n", kernel_time);
#endif
}

