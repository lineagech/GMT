// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "srad.h"

// includes, project
#include <cuda.h>

#include "settings.h"
#include "page_cache.h"

using TYPE = double;

Settings settings; 
std::vector<Controller*> ctrls;
page_cache_t* h_pc;

range_t<TYPE>* J_cuda_range;
range_t<TYPE>* J_tmp_range[2];
range_t<TYPE>* J_tmp_square_range[2];
range_t<TYPE>* J_out_cuda_range;
range_t<TYPE>* C_cuda_range;
range_t<TYPE>* E_C_range;
range_t<TYPE>* W_C_range;
range_t<TYPE>* N_C_range;
range_t<TYPE>* S_C_range;
std::vector<range_t<TYPE>*> vec_J_cuda;
std::vector<range_t<TYPE>*> vec_J_tmp[2];
std::vector<range_t<TYPE>*> vec_J_tmp_square[2];
std::vector<range_t<TYPE>*> vec_J_out_cuda;
std::vector<range_t<TYPE>*> vec_C_cuda;
std::vector<range_t<TYPE>*> vec_E_C;
std::vector<range_t<TYPE>*> vec_W_C;
std::vector<range_t<TYPE>*> vec_N_C;
std::vector<range_t<TYPE>*> vec_S_C;
array_t<TYPE>* h_J_cuda_array;
array_t<TYPE>* h_J_tmp_array[2];
array_t<TYPE>* h_J_tmp_square_array[2];
array_t<TYPE>* h_J_out_cuda_array;
array_t<TYPE>* h_C_cuda_array;
array_t<TYPE>* h_E_C_array;
array_t<TYPE>* h_W_C_array;
array_t<TYPE>* h_N_C_array;
array_t<TYPE>* h_S_C_array;

void* sync_mem_h = NULL;
void* sync_mem_d = NULL;
TYPE* C_ptr = NULL;

#if USE_HOST_CACHE
HostCache* hc;
#endif
// includes, kernels
#include "srad_kernel.cu"

typedef enum {
    GPUMEM = 0,
    UVM_READONLY = 1,
    UVM_DIRECT = 2,
    UVM_READONLY_NVLINK = 3,
    UVM_DIRECT_NVLINK = 4,
    DRAGON_MAP = 5,
    BAFS_DIRECT = 6,
} mem_type;

const char* const sam_ctrls_paths[] = {"/dev/libnvm_vmalloc0"};

__global__ 
void copy(array_d_t<TYPE>* arr, array_d_t<TYPE>* arr_out, array_d_t<TYPE>* arr_out2, size_t cols)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    arr_out->seq_write(i*cols+j, arr->seq_read(i*cols+j));
    arr_out2->seq_write(i*cols+j, arr->seq_read(i*cols+j)*arr->seq_read(i*cols+j));
}

__global__ 
void cal_J_kernel(array_d_t<TYPE>* arr, array_d_t<TYPE>* arr_sum, TYPE* res)
{
    __shared__ TYPE sdata[1024];
    uint64_t tid = threadIdx.x;
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    //uint64_t i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
    
    sdata[tid] = arr->seq_read(i);
    //sdata[tid] = arr->seq_read(i) + arr->seq_read(i+blockDim.x);
    __syncthreads();

    for (uint64_t s = blockDim.x>>1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        arr_sum->seq_write(blockIdx.x, sdata[0]);
        *res = sdata[0];
    }
}

void init_page_cache(size_t size)
{
    size_t total_pages = 0;
    int cnt = 0;
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

    uint64_t n_pages = (size*sizeof(TYPE)+pc_page_size-1) / pc_page_size;
    
    h_pc = new page_cache_t(pc_page_size, pc_pages, settings.cudaDevice, ctrls[0][0], (uint64_t)64, ctrls);
    
    J_cuda_range = new range_t<TYPE>(0, size, total_pages, n_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_J_cuda.resize(1);
    vec_J_cuda[0] = J_cuda_range;
    h_J_cuda_array = new array_t<TYPE>(size, total_pages*pc_page_size, vec_J_cuda, settings.cudaDevice, cnt++);
    total_pages += n_pages;
    
    /*
    J_out_cuda_range = new range_t<TYPE>(0, size, total_pages, n_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_J_out_cuda.resize(1);
    vec_J_out_cuda[0] = J_out_cuda_range;
    h_J_out_cuda_array = new array_t<TYPE>(size, total_pages*pc_page_size, vec_J_out_cuda, settings.cudaDevice, cnt++);
    total_pages += n_pages;
    */

    C_cuda_range = new range_t<TYPE>(0, size, total_pages, n_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_C_cuda.resize(1);
    vec_C_cuda[0] = C_cuda_range;
    h_C_cuda_array = new array_t<TYPE>(size, total_pages*pc_page_size, vec_C_cuda, settings.cudaDevice, cnt++);
    total_pages += n_pages;

    E_C_range = new range_t<TYPE>(0, size, total_pages, n_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_E_C.resize(1);
    vec_E_C[0] = E_C_range;
    h_E_C_array = new array_t<TYPE>(size, total_pages*pc_page_size, vec_E_C, settings.cudaDevice, cnt++);
    total_pages += n_pages;

    W_C_range = new range_t<TYPE>(0, size, total_pages, n_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_W_C.resize(1);
    vec_W_C[0] = W_C_range;
    h_W_C_array = new array_t<TYPE>(size, total_pages*pc_page_size, vec_W_C, settings.cudaDevice, cnt++);
    total_pages += n_pages;

    N_C_range = new range_t<TYPE>(0, size, total_pages, n_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_N_C.resize(1);
    vec_N_C[0] = N_C_range;
    h_N_C_array = new array_t<TYPE>(size, total_pages*pc_page_size, vec_N_C, settings.cudaDevice, cnt++);
    total_pages += n_pages;

    S_C_range = new range_t<TYPE>(0, size, total_pages, n_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_S_C.resize(1);
    vec_S_C[0] = S_C_range;
    h_S_C_array = new array_t<TYPE>(size, total_pages*pc_page_size, vec_S_C, settings.cudaDevice, cnt++);
    total_pages += n_pages;

    J_tmp_range[0] = new range_t<TYPE>(0, size, total_pages, n_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_J_tmp[0].resize(1);
    vec_J_tmp[0][0] = J_tmp_range[0];
    h_J_tmp_array[0] = new array_t<TYPE>(size, total_pages*pc_page_size, vec_J_tmp[0], settings.cudaDevice, cnt++);
    total_pages += n_pages;

    J_tmp_range[1] = new range_t<TYPE>(0, size, total_pages, n_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_J_tmp[1].resize(1);
    vec_J_tmp[1][0] = J_tmp_range[1];
    h_J_tmp_array[1] = new array_t<TYPE>(size, total_pages*pc_page_size, vec_J_tmp[1], settings.cudaDevice, cnt++);
    total_pages += n_pages;

    J_tmp_square_range[0] = new range_t<TYPE>(0, size, total_pages, n_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_J_tmp_square[0].resize(1);
    vec_J_tmp_square[0][0] = J_tmp_square_range[0];
    h_J_tmp_square_array[0] = new array_t<TYPE>(size, total_pages*pc_page_size, vec_J_tmp_square[0], settings.cudaDevice, cnt++);
    total_pages += n_pages;

    J_tmp_square_range[1] = new range_t<TYPE>(0, size, total_pages, n_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_J_tmp_square[1].resize(1);
    vec_J_tmp_square[1][0] = J_tmp_square_range[1];
    h_J_tmp_square_array[1] = new array_t<TYPE>(size, total_pages*pc_page_size, vec_J_tmp_square[1], settings.cudaDevice, cnt++);
    total_pages += n_pages;

    printf("Page Cache Initialized\n");
    printf("Total pages %lu (%lu Mbytes)\n", total_pages, total_pages*pc_page_size/1024/1024);
    fflush(stdout);

#if USE_HOST_CACHE
    std::cerr << "creating Host Cache...\n";
    hc = createHostCache(ctrls[0], settings.maxPageCacheSize);
    
    uint64_t starting_lba = 0;
    for (int i = 0; i < 10; i++) {
        hc->registerRangesLBA(starting_lba);
        starting_lba += n_pages * pc_page_size / 512;
    }
#endif

}

void print_stats() 
{
    h_J_cuda_array->print_reset_stats();
    h_J_tmp_array[0]->print_reset_stats();
    h_J_tmp_array[1]->print_reset_stats();
    h_J_tmp_square_array[0]->print_reset_stats();
    h_J_tmp_square_array[1]->print_reset_stats();
    //h_J_out_cuda_array->print_reset_stats();
    h_C_cuda_array->print_reset_stats();
    h_E_C_array->print_reset_stats();
    h_W_C_array->print_reset_stats();
    h_N_C_array->print_reset_stats();
    h_S_C_array->print_reset_stats();

    ctrls[0]->print_reset_stats();
#if USE_HOST_CACHE
    revokeHostRuntime();
#endif
}


void random_matrix(TYPE *I, int rows, int cols);
void runTest( int argc, char** argv);
void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>\n", argv[0]);
	fprintf(stderr, "\t<rows>   - number of rows\n");
	fprintf(stderr, "\t<cols>    - number of cols\n");
	fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
	fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
	fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
	fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
	fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
	fprintf(stderr, "\t<no. of iter>   - number of iterations\n");
	
	exit(1);
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
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


    runTest( argc, argv);

    return EXIT_SUCCESS;
}


void
runTest( int argc, char** argv) 
{
    size_t rows, cols, size_I, size_R, niter = settings.total_iterations, iter;
    TYPE *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;
    TYPE *sum_d = NULL, *sum2_d = NULL;

#ifdef CPU
	TYPE Jc, G2, L, num, den, qsqr;
	size_t *iN,*iS,*jE,*jW, k;
	TYPE *dN,*dS,*dW,*dE;
	TYPE cN,cS,cW,cE,D;
#endif

#ifdef GPU
	
	TYPE *J_cuda;
    TYPE *C_cuda;
	TYPE *E_C, *W_C, *N_C, *S_C;

#endif

	size_t r1 = settings.r1, r2 = settings.r2, c1 = settings.c1, c2 = settings.c2;
	TYPE *c;
    
    if (settings.rows % 16 != 0 || settings.cols % 16 != 0) {   
        fprintf(stderr, "rows and cols must be multiples of 16\n");
        exit(1);
    }
    
    cols = settings.cols;
    rows = settings.rows;

	size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);   

    if (settings.memalloc == GPUMEM) {
	    I = (TYPE *)malloc(size_I * sizeof(TYPE));
        J = (TYPE *)malloc(size_I * sizeof(TYPE));
	    c  = (TYPE *)malloc(sizeof(TYPE)* size_I) ;
    }
    else if (settings.memalloc == BAFS_DIRECT) {
        cudaMalloc(&sum_d, sizeof(TYPE));
        cudaMalloc(&sum2_d, sizeof(TYPE));
        cudaMemset(sum_d, 0, sizeof(TYPE));
        cudaMemset(sum2_d, 0, sizeof(TYPE));
    }

#ifdef CPU

    iN = (size_t *)malloc(sizeof(size_t*) * rows) ;
    iS = (size_t *)malloc(sizeof(size_t*) * rows) ;
    jW = (size_t *)malloc(sizeof(size_t*) * cols) ;
    jE = (size_t *)malloc(sizeof(size_t*) * cols) ;    


	dN = (TYPE *)malloc(sizeof(TYPE)* size_I) ;
    dS = (TYPE *)malloc(sizeof(TYPE)* size_I) ;
    dW = (TYPE *)malloc(sizeof(TYPE)* size_I) ;
    dE = (TYPE *)malloc(sizeof(TYPE)* size_I) ;    
    

    for (int i=0; i< rows; i++) {
        iN[i] = i-1;
        iS[i] = i+1;
    }    
    for (int j=0; j< cols; j++) {
        jW[j] = j-1;
        jE[j] = j+1;
    }
    iN[0]    = 0;
    iS[rows-1] = rows-1;
    jW[0]    = 0;
    jE[cols-1] = cols-1;

#endif

#ifdef GPU

	//Allocate device memory
    if (settings.memalloc == GPUMEM) {
        cudaMalloc((void**)& J_cuda, sizeof(TYPE)* size_I);
        cudaMalloc((void**)& C_cuda, sizeof(TYPE)* size_I);
        cudaMalloc((void**)& E_C, sizeof(TYPE)* size_I);
        cudaMalloc((void**)& W_C, sizeof(TYPE)* size_I);
        cudaMalloc((void**)& S_C, sizeof(TYPE)* size_I);
        cudaMalloc((void**)& N_C, sizeof(TYPE)* size_I);
    }
    else if (settings.memalloc == BAFS_DIRECT) {
        init_page_cache(size_I);
        
        cudaMalloc(&sync_mem_d, 40*1024*1024);
        cudaMalloc((void**)&C_ptr, size_I*sizeof(TYPE));
        sync_mem_h = malloc(40*1024*1024);

    }
	
#endif 

	printf("Randomizing the input matrix\n");
	//Generate a random matrix
    if (settings.memalloc == GPUMEM) {
	    random_matrix(I, rows, cols);

        for (size_t k = 0;  k < size_I; k++ ) {
     	    J[k] = (TYPE)exp(I[k]) ;
        }
    }
	printf("Start the SRAD main loop\n");
    
    float time_elapsed;
    cudaEvent_t start, end;
    cuda_err_chk(cudaEventCreate(&start));
    cuda_err_chk(cudaEventCreate(&end));
    cuda_err_chk(cudaEventRecord(start, 0));

    for (iter=0; iter< niter; iter++) {
        fprintf(stderr, "iter %ld\n", iter);
        if (settings.memalloc == GPUMEM) {
            sum=0; sum2=0;
            for (size_t i=r1; i<=r2; i++) {
                for (size_t  j=c1; j<=c2; j++) {
                    tmp   = J[i * cols + j];
                    sum  += tmp ;
                    sum2 += tmp*tmp;
                }
            }
            meanROI = sum / size_R;
            varROI  = (sum2 / size_R) - meanROI*meanROI;
            q0sqr   = varROI / (meanROI*meanROI);
        }
        else if (settings.memalloc == BAFS_DIRECT) {
            dim3 blockDims(16, 16);
            dim3 gridDims((r2-r1+1)/16, (c2-c1+1)/16);

        #if USE_HOST_CACHE
            copy<<<gridDims, blockDims, 0, stream_mngr->kernel_stream>>>(h_J_cuda_array->d_array_ptr, h_J_tmp_array[0]->d_array_ptr, h_J_tmp_square_array[0]->d_array_ptr, cols);
        #else  
            copy<<<gridDims, blockDims>>>(h_J_cuda_array->d_array_ptr, h_J_tmp_array[0]->d_array_ptr, h_J_tmp_square_array[0]->d_array_ptr, cols);
        #endif
            printf("Copy to J_sum and J_sum2 done.\n");
            size_t s = (r2-r1+1) * (c2-c1+1);
            for (size_t i = s; s > 1; s >>= 10) {
        #if USE_HOST_CACHE
                cal_J_kernel<<<s/CalBlockSize, CalBlockSize, 0, stream_mngr->kernel_stream>>>(h_J_tmp_array[0]->d_array_ptr, h_J_tmp_array[1]->d_array_ptr, sum_d);
                cal_J_kernel<<<s/CalBlockSize, CalBlockSize, 0, stream_mngr->kernel_stream>>>(h_J_tmp_square_array[0]->d_array_ptr, h_J_tmp_square_array[1]->d_array_ptr, sum2_d);
                cudaStreamSynchronize(stream_mngr->kernel_stream);
        #else
                cal_J_kernel<<<s/CalBlockSize, CalBlockSize>>>(h_J_tmp_array[0]->d_array_ptr, h_J_tmp_array[1]->d_array_ptr, sum_d);
                cal_J_kernel<<<s/CalBlockSize, CalBlockSize>>>(h_J_tmp_square_array[0]->d_array_ptr, h_J_tmp_square_array[1]->d_array_ptr, sum2_d);
                cudaDeviceSynchronize();
        #endif
                printf("Update sum and sum2 (remaining %ld).\n", i);
                std::swap(h_J_tmp_array[0], h_J_tmp_array[1]);
                std::swap(h_J_tmp_square_array[0], h_J_tmp_square_array[1]);
            }
            printf("Update sum and sum2 done.\n");

            cudaMemcpy(&sum, sum_d, sizeof(TYPE), cudaMemcpyDeviceToHost);
            cudaMemcpy(&sum2, sum2_d, sizeof(TYPE), cudaMemcpyDeviceToHost);

            meanROI = sum / size_R;
            varROI  = (sum2 / size_R) - meanROI*meanROI;
            q0sqr   = varROI / (meanROI*meanROI);
        }

#ifdef CPU
        for (int i = 0 ; i < rows ; i++) {
            for (int j = 0; j < cols; j++) { 

                k = i * cols + j;
                Jc = J[k];

                // directional derivates
                dN[k] = J[iN[i] * cols + j] - Jc;
                dS[k] = J[iS[i] * cols + j] - Jc;
                dW[k] = J[i * cols + jW[j]] - Jc;
                dE[k] = J[i * cols + jE[j]] - Jc;

                G2 = (dN[k]*dN[k] + dS[k]*dS[k] 
                        + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

                L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

                num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
                den  = 1 + (.25*L);
                qsqr = num/(den*den);

                // diffusion coefficent (equ 33)
                den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
                c[k] = 1.0 / (1.0+den) ;

                // saturate diffusion coefficent
                if (c[k] < 0) {c[k] = 0;}
                else if (c[k] > 1) {c[k] = 1;}
            }
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {        

                // current index
                k = i * cols + j;

                // diffusion coefficent
                cN = c[k];
                cS = c[iS[i] * cols + j];
                cW = c[k];
                cE = c[i * cols + jE[j]];

                // divergence (equ 58)
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];

                // image update (equ 61)
                J[k] = J[k] + 0.25*lambda*D;
            }
        }

#endif // CPU


#ifdef GPU

        //Currently the input size must be divided by 16 - the block size
        int block_x = cols/BLOCK_SIZE ;
        int block_y = rows/BLOCK_SIZE ;

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(block_x , block_y);


        if (settings.memalloc == GPUMEM) {
            //Copy data from main memory to device memory
            cudaMemcpy(J_cuda, J, sizeof(TYPE) * size_I, cudaMemcpyHostToDevice);
        
            //Run kernels
            srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr); 
            srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr); 

            //Copy data from device memory to main memory
            cudaMemcpy(J, J_cuda, sizeof(TYPE) * size_I, cudaMemcpyDeviceToHost);
        }
        else if (settings.memalloc == BAFS_DIRECT) {
            //Run kernels
        #if USE_HOST_CACHE
            /*
            srad_cuda_1<<<dimGrid, dimBlock, 0, stream_mngr->kernel_stream>>>(h_E_C_array->d_array_ptr, 
                h_W_C_array->d_array_ptr, 
                h_N_C_array->d_array_ptr, 
                h_S_C_array->d_array_ptr, 
                h_J_cuda_array->d_array_ptr, 
                h_C_cuda_array->d_array_ptr, 
                cols, rows, q0sqr); 
            srad_cuda_2<<<dimGrid, dimBlock, 0, stream_mngr->kernel_stream>>>(h_E_C_array->d_array_ptr, 
                h_W_C_array->d_array_ptr, 
                h_N_C_array->d_array_ptr, 
                h_S_C_array->d_array_ptr, 
                h_J_cuda_array->d_array_ptr, 
                h_C_cuda_array->d_array_ptr, 
                cols, rows, lambda, q0sqr); 
            */        
            size_t divs = settings.divs;
            for (size_t d = 0; d < divs; d++) {
                srad_cuda_mod_1<<<cols*rows/settings.warp_size/divs, settings.warp_size, 0, stream_mngr->kernel_stream>>>(h_E_C_array->d_array_ptr, 
                        h_W_C_array->d_array_ptr, 
                        h_N_C_array->d_array_ptr, 
                        h_S_C_array->d_array_ptr, 
                        h_J_cuda_array->d_array_ptr, 
                        h_C_cuda_array->d_array_ptr, 
                        cols, rows, q0sqr, C_ptr, cols*rows*d/divs); 

                cudaStreamSynchronize(stream_mngr->kernel_stream);
                cudaMemcpy(sync_mem_d, sync_mem_h, 40*1024*1024, cudaMemcpyHostToDevice);
                fprintf(stderr, "cuda-1 %ld\n", d);
            }
            fprintf(stderr, "srad_cuda_1 done.\n");
            
            for (size_t d = 0; d < 64; d++) {
                if (d == 63) {
                    lastIteration(&(stream_mngr->kernel_stream));
                }
                srad_cuda_mod_2<<<cols*rows/settings.warp_size/64, settings.warp_size, 0, stream_mngr->kernel_stream>>>(h_E_C_array->d_array_ptr, 
                        h_W_C_array->d_array_ptr, 
                        h_N_C_array->d_array_ptr, 
                        h_S_C_array->d_array_ptr, 
                        h_J_cuda_array->d_array_ptr, 
                        h_C_cuda_array->d_array_ptr, 
                        cols, rows, lambda, q0sqr, C_ptr, cols*rows*d/64); 

                cudaStreamSynchronize(stream_mngr->kernel_stream);
                cudaMemcpy(sync_mem_d, sync_mem_h, 40*1024*1024, cudaMemcpyHostToDevice);
                fprintf(stderr, "cuda-2 %ld\n", d);
            }

            cudaStreamSynchronize(stream_mngr->kernel_stream);

        #else
        /*
            srad_cuda_1<<<dimGrid, dimBlock>>>(h_E_C_array->d_array_ptr, 
                h_W_C_array->d_array_ptr, 
                h_N_C_array->d_array_ptr, 
                h_S_C_array->d_array_ptr, 
                h_J_cuda_array->d_array_ptr, 
                h_C_cuda_array->d_array_ptr, 
                cols, rows, q0sqr); 
            */
            size_t divs = settings.divs;
            for (size_t d = 0; d < divs; d++) {
                srad_cuda_mod_1<<<cols*rows/settings.warp_size/divs, settings.warp_size>>>(h_E_C_array->d_array_ptr, 
                    h_W_C_array->d_array_ptr, 
                    h_N_C_array->d_array_ptr, 
                    h_S_C_array->d_array_ptr, 
                    h_J_cuda_array->d_array_ptr, 
                    h_C_cuda_array->d_array_ptr, 
                    cols, rows, q0sqr, C_ptr, cols*rows*d/divs); 
                
                cudaMemcpy(sync_mem_d, sync_mem_h, 40*1024*1024, cudaMemcpyHostToDevice);
                cudaDeviceSynchronize();
                fprintf(stderr, "cuda-1 %ld\n", d);
            }
            fprintf(stderr, "srad_cuda_1 done.\n");
            /*
            srad_cuda_2<<<dimGrid, dimBlock>>>(h_E_C_array->d_array_ptr, 
                h_W_C_array->d_array_ptr, 
                h_N_C_array->d_array_ptr, 
                h_S_C_array->d_array_ptr, 
                h_J_cuda_array->d_array_ptr, 
                h_C_cuda_array->d_array_ptr, 
                cols, rows, lambda, q0sqr); 
            */
            ///*
            for (size_t d = 0; d < 64; d++) {
                srad_cuda_mod_2<<<cols*rows/settings.warp_size/64, settings.warp_size>>>(h_E_C_array->d_array_ptr, 
                        h_W_C_array->d_array_ptr, 
                        h_N_C_array->d_array_ptr, 
                        h_S_C_array->d_array_ptr, 
                        h_J_cuda_array->d_array_ptr, 
                        h_C_cuda_array->d_array_ptr, 
                        cols, rows, lambda, q0sqr, C_ptr, cols*rows*d/64); 
            
                cudaMemcpy(sync_mem_d, sync_mem_h, 40*1024*1024, cudaMemcpyHostToDevice);
                cudaDeviceSynchronize();
                fprintf(stderr, "cuda-2 %ld\n", d);
            }
            //*/
        #endif
        }
#endif   
    } // endfor

    h_pc->flush_cache();
#if USE_HOST_CACHE
    flushHostCache();
#endif
    cudaDeviceSynchronize();
    cuda_err_chk(cudaEventRecord(end, 0));
    cuda_err_chk(cudaEventSynchronize(end));
    cuda_err_chk(cudaEventElapsedTime(&time_elapsed, start, end));
    printf("Elapsed Time %f ms\n", time_elapsed);


#ifdef OUTPUT
    //Printing output	
		printf("Printing Output:\n"); 
    for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
         printf("%.5f ", J[i * cols + j]); 
		}	
     printf("\n"); 
   }
#endif 

	printf("Computation Done\n");

    if (settings.memalloc == GPUMEM) {
        free(I);
        free(J);
#ifdef CPU
        free(iN); free(iS); free(jW); free(jE);
        free(dN); free(dS); free(dW); free(dE);
#endif
#ifdef GPU
        cudaFree(C_cuda);
        cudaFree(J_cuda);
        cudaFree(E_C);
        cudaFree(W_C);
        cudaFree(N_C);
        cudaFree(S_C);
     
#endif 
        free(c);
    }
    else if (settings.memalloc == BAFS_DIRECT) {
        print_stats();
        printf("freeing...");
        cudaFree(sum_d);
        cudaFree(sum2_d);

        cudaFree(sync_mem_d);
        free(sync_mem_h);
#if USE_HOST_CACHE
        //revokeHostRuntime();
#endif
    }
}


void random_matrix(TYPE *I, int rows, int cols){
    
	srand(7);
	
	for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
		 I[i * cols + j] = rand()/(TYPE)RAND_MAX ;
		}
	}

}

