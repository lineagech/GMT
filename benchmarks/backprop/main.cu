// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <omp.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <util.h>
#include <vector>
#include <unistd.h>
#include <sys/mman.h>
#include <chrono>
using namespace std::chrono;

// includes, kernels
//#include "backprop_cuda_kernel.cu"
//#include "backprop.h"
#include "settings.h"
#include "page_cache.h"
#include "buffer.h"

#define BIGRND 0x7fffffff

#define GPU
#define THREADS 256
#define WIDTH 16  // shared memory width  
#define HEIGHT 16 // shared memory height

#define ETA 0.3       //eta value
#define MOMENTUM 0.3  //momentum value
#define NUM_THREAD 4  //OpenMP threads

//#define OPEN

#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))

#define fastcopy(to,from,len)\
{\
    register char *_to,*_from;\
    register int _i,_l;\
    _to = (char *)(to);\
    _from = (char *)(from);\
    _l = (len);\
    for (_i = 0; _i < _l; _i++) *_to++ = *_from++;\
}

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


typedef double TYPE;

typedef struct {
    uint64_t input_n;                  /* number of input units */
    uint64_t hidden_n;                 /* number of hidden units */
    uint64_t output_n;                 /* number of output units */

    TYPE *input_units;          /* the input units */
    TYPE *hidden_units;         /* the hidden units */
    TYPE *output_units;         /* the output units */

    TYPE *hidden_delta;         /* storage for hidden unit error */
    TYPE *output_delta;         /* storage for output unit error */

    TYPE *target;               /* storage for target vector */

    TYPE **input_weights;       /* weights from input to hidden layer */
    TYPE **hidden_weights;      /* weights from hidden to output layer */

    /*** The next two are for momentum ***/
    TYPE **input_prev_weights;  /* previous change on input to hidden wgt */
    TYPE **hidden_prev_weights; /* previous change on hidden to output wgt */

    // For BaM
    uint64_t n_input_units_pages;
    uint64_t n_input_weights_pages;
    uint64_t n_output_hidden_pages;
    uint64_t n_hidden_partial_sum_pages;
    uint64_t n_input_prev_weights_pages;
    uint64_t n_hidden_prev_weights_pages;
    uint64_t n_buffer_pages;
    uint64_t n_output_units_pages;
    uint64_t n_hidden_weights_pages;
    uint64_t n_output_delta_pages;
    uint64_t n_target_pages;
    uint64_t n_hidden_delta_pages;
    
    #define NUM_INPUTS (32)
    range_t<TYPE>* h_range_input_units[NUM_INPUTS];              // input_cuda
    range_t<TYPE>* h_range_input_weights;
    range_t<TYPE>* h_range_output_hidden_units;      // hidden units
    range_t<TYPE>* h_range_hidden_partial_sum;
    range_t<TYPE>* h_range_input_prev_weights;
    range_t<TYPE>* h_range_hidden_prev_weights;      // hidden prev weights
    range_t<TYPE>* h_range_buffer[2];
    range_t<TYPE>* h_range_output_units;
    range_t<TYPE>* h_range_hidden_weights;           // hidden weights
    range_t<TYPE>* h_range_output_delta;            // output delta
    range_t<TYPE>* h_range_target;                  // target
    range_t<TYPE>* h_range_hidden_delta;           // hidden delta
    
    std::vector<range_t<TYPE>*> vec_range_input_units[NUM_INPUTS];
    std::vector<range_t<TYPE>*> vec_range_input_weights;
    std::vector<range_t<TYPE>*> vec_range_output_hidden_units;
    std::vector<range_t<TYPE>*> vec_range_hidden_partial_sum;
    std::vector<range_t<TYPE>*> vec_range_input_prev_weights;
    std::vector<range_t<TYPE>*> vec_range_hidden_prev_weights;
    std::vector<range_t<TYPE>*> vec_range_buffer[2];
    std::vector<range_t<TYPE>*> vec_range_output_units;
    std::vector<range_t<TYPE>*> vec_range_hidden_weights;
    std::vector<range_t<TYPE>*> vec_range_output_delta;
    std::vector<range_t<TYPE>*> vec_range_target;
    std::vector<range_t<TYPE>*> vec_range_hidden_delta;
    
    array_t<TYPE>* h_input_units_array[NUM_INPUTS];
    array_t<TYPE>* h_input_weights_array;
    array_t<TYPE>* h_output_hidden_units_array;
    array_t<TYPE>* h_hidden_partial_sum_array;
    array_t<TYPE>* h_input_prev_weights_array;
    array_t<TYPE>* h_hidden_prev_weights_array;
    array_t<TYPE>* h_buffer_array[2];
    array_t<TYPE>* h_output_units_array;
    array_t<TYPE>* h_hidden_weights_array;
    array_t<TYPE>* h_output_delta_array;
    array_t<TYPE>* h_target_array;
    array_t<TYPE>* h_hidden_delta_array;

} BPNN;



void print_stats(BPNN* bpnn)
{
    //for (int i = 0; i < settings.iter; i++)
    for (int i = 0; i < NUM_INPUTS; i++)
        bpnn->h_input_units_array[i]->print_reset_stats();
    bpnn->h_input_weights_array->print_reset_stats();
    bpnn->h_output_hidden_units_array->print_reset_stats();
    bpnn->h_hidden_partial_sum_array->print_reset_stats();
    bpnn->h_input_prev_weights_array->print_reset_stats();
    bpnn->h_buffer_array[0]->print_reset_stats();
    bpnn->h_buffer_array[1]->print_reset_stats();
    bpnn->h_output_units_array->print_reset_stats();
    bpnn->h_hidden_weights_array->print_reset_stats();
    bpnn->h_output_delta_array->print_reset_stats();
    bpnn->h_hidden_delta_array->print_reset_stats();

    revokeHostRuntime();
}

/*** User-level functions ***/

void bpnn_initialize();

BPNN *bpnn_create();
void bpnn_free();

void bpnn_train();
void bpnn_feedforward();

void bpnn_save();
BPNN *bpnn_read();

//const char* const sam_ctrls_paths[] = {"/dev/libnvm0"};
const char* const sam_ctrls_paths[] = {"/dev/libnvm_vmalloc0"};
Settings settings;
std::vector<Controller*> ctrls;
page_cache_t* h_pc;
HostCache* hc;

unsigned int num_threads = 0;
uint64_t num_blocks = 0;

void flush_for_hc(BPNN* bpnn)
{
    for (int i = 0; i < NUM_INPUTS; i++)
        h_pc->fetch_and_flush(bpnn->h_range_input_units[i]);              // input_cuda
    h_pc->fetch_and_flush(bpnn->h_range_input_weights);
    h_pc->fetch_and_flush(bpnn->h_range_output_hidden_units);      // hidden units
    h_pc->fetch_and_flush(bpnn->h_range_hidden_partial_sum);
    h_pc->fetch_and_flush(bpnn->h_range_input_prev_weights);
    h_pc->fetch_and_flush(bpnn->h_range_hidden_prev_weights);      // hidden prev weights
    h_pc->fetch_and_flush(bpnn->h_range_buffer[0]);
    h_pc->fetch_and_flush(bpnn->h_range_buffer[1]);
    h_pc->fetch_and_flush(bpnn->h_range_output_units);
    h_pc->fetch_and_flush(bpnn->h_range_hidden_weights);           // hidden weights
    h_pc->fetch_and_flush(bpnn->h_range_output_delta);            // output delta
    h_pc->fetch_and_flush(bpnn->h_range_target);                  // target
    h_pc->fetch_and_flush(bpnn->h_range_hidden_delta);           // hidden delta
    

}
//////////////////////////////////////
// CUDA Kernel ///////////////////////
//////////////////////////////////////

#include "math.h"
#include "cuda.h"

#include "host_cache.h"


__global__ void
bpnn_layerforward_CUDA(TYPE *input_cuda,
        TYPE *output_hidden_cuda,
        TYPE *input_hidden_cuda,
        TYPE *hidden_partial_sum,
        int in,
        int hid) 
{
    //int by = blockIdx.y;
    int by = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

    int index_in = HEIGHT * by + ty + 1;

    __shared__ TYPE input_node[HEIGHT];
    __shared__ TYPE weight_matrix[HEIGHT][WIDTH];


    if ( tx == 0 )
        input_node[ty] = input_cuda[index_in] ;

    __syncthreads();

    weight_matrix[ty][tx] = input_hidden_cuda[index];

    __syncthreads();

    weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

    __syncthreads();   

    for ( int i = 1 ; i <= __log2f(HEIGHT) ; i++){

        int power_two = __powf(2, i);

        if( ty % power_two == 0 )
            weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];

        __syncthreads();

    }

    //__syncthreads();

    input_hidden_cuda[index] = weight_matrix[ty][tx];

    /*
       for ( unsigned int i = 2 ; i <= HEIGHT ; i *= 2){

       unsigned int power_two = i - 1;

       if( (ty & power_two) == 0 ) {
       weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];
       }

       }
     */

    __syncthreads();

    if ( tx == 0 ) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
    }

}

__global__ void
bpnn_layerforward_CUDA(array_d_t<TYPE> *input_cuda,
        array_d_t<TYPE> *output_hidden_cuda,
        array_d_t<TYPE> *input_hidden_cuda,
        array_d_t<TYPE> *hidden_partial_sum,
        int64_t in,
        int64_t hid) 
{
    //int by = blockIdx.y;
    int64_t by = blockIdx.x;
    int64_t tx = threadIdx.x;
    int64_t ty = threadIdx.y;

    int64_t index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

    int64_t index_in = HEIGHT * by + ty + 1;

    __shared__ TYPE input_node[HEIGHT];
    __shared__ TYPE weight_matrix[HEIGHT][WIDTH];


    if ( tx == 0 ) {
        //printf("index_in %ld\n", index_in);
        input_node[ty] = input_cuda->seq_read(index_in) ;
    }

    __syncthreads();

    //weight_matrix[ty][tx] = input_hidden_cuda[index];
    weight_matrix[ty][tx] = input_hidden_cuda->seq_read(index);

    __syncthreads();

    weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

    __syncthreads();   

    for ( int i = 1 ; i <= __log2f(HEIGHT) ; i++){

        int power_two = __powf(2, i);

        if( ty % power_two == 0 )
            weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];

        __syncthreads();

    }

    //__syncthreads();

    input_hidden_cuda->seq_write(index, weight_matrix[ty][tx]);

    /*
       for ( unsigned int i = 2 ; i <= HEIGHT ; i *= 2){

       unsigned int power_two = i - 1;

       if( (ty & power_two) == 0 ) {
       weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];
       }

       }
     */

    __syncthreads();

    if ( tx == 0 ) {
        //hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
        hidden_partial_sum->seq_write((uint64_t)by * hid + ty, weight_matrix[tx][ty]);
    }

}


__global__
void reduce_sum(int hid, int num_blocks, TYPE* partial_sum, TYPE* out)
{
    //extern __shared__ TYPE sdata[];
    __shared__ TYPE sdata[512];
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t sid = tid % num_blocks;
    uint64_t j = tid / num_blocks;
    sdata[threadIdx.x] = partial_sum[sid*hid + j];

    __syncthreads();

    // one block works on this array
    TYPE sum = 0;
    
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x+s];
            //partial_sum[tid] += partial_sum[tid+s];
            //partial_sum[sid*hid+j] += partial_sum[(sid+s)*hid+j];
        }
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t num_in_col = num_blocks / blockDim.x;
        uint64_t bid = blockIdx.x % num_in_col;
        uint64_t b_j = blockIdx.x / num_in_col;
        out[b_j*num_in_col+bid] = sdata[0];
    }
    //return partial_sum[j*num_blocks];
}

__global__
void reduce_sum(int64_t hid, int64_t num_blocks, array_d_t<TYPE>* partial_sum, array_d_t<TYPE>* out)
{
    //extern __shared__ TYPE sdata[];
    __shared__ TYPE sdata[512];
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t sid = tid % num_blocks;
    uint64_t j = tid / num_blocks;
    sdata[threadIdx.x] = partial_sum->seq_read(sid*hid + j);

    __syncthreads();

    // one block works on this array
    TYPE sum = 0;
    
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x+s];
        }
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t num_in_col = num_blocks / blockDim.x;
        uint64_t bid = blockIdx.x % num_in_col;
        uint64_t b_j = blockIdx.x / num_in_col;
        out->seq_write(b_j*num_in_col+bid, sdata[0]);
    }
}


__global__ 
void activate_kernel(array_d_t<TYPE>* hidden_units, array_d_t<TYPE>* buffer, array_d_t<TYPE>* input_weights, int64_t hid)
{
    //cudaMemcpy((void*)((TYPE*)(net->hidden_units)+1), buffers[0], (hid) * sizeof(TYPE), cudaMemcpyDeviceToHost);
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    //cudaMemcpy((void*)((TYPE*)(net->hidden_units)+1), buffers[0], (hid) * sizeof(TYPE), cudaMemcpyDeviceToHost);
    //for (uint32_t j = 1; j <= hid; j++) {
    //    net->hidden_units[j] = TYPE(1.0/(1.0+exp(-(net->hidden_units[j]+net->input_weights[0][j]))));
    //}
    if (tid >= 1 && tid <= hid) {
        hidden_units->seq_write(tid, TYPE(1.0/(1.0+exp(-(buffer->seq_read(tid-1)+input_weights->seq_read(tid))))));
    }
}

__global__ void 
bpnn_aggregate(int hid, int num_blocks, TYPE* partial_sum, TYPE* input_weights, TYPE* hidden_units)
{
    /*
    for (int j = 1; j <= hid; j++) {
        sum = 0.0;
        for (int k = 0; k < num_blocks; k++) {	
            sum += partial_sum[k * hid + j-1] ;
        }
        sum += net->input_weights[0][j];
        net-> hidden_units[j] = TYPE(1.0 / (1.0 + exp(-sum)));
    }
    */
    // blockDim.x should be eqaul to num_blocks
    //int j = blockIdx.x + 1;
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t sid = tid % num_blocks;
    uint64_t j = tid / num_blocks + 1;

    TYPE sum = 0.0;
    //for (int k = 0; k < num_blocks; k++) {	
    //    sum += partial_sum[k * hid + j-1] ;
    //}
    //sum += reduce_sum(hid, num_blocks, partial_sum);
    sum += input_weights[j];

    //hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
    if (sid == 0) {
        hidden_units[j] = TYPE(1.0/(1.0+exp(-sum)));
    }
}

__device__ TYPE reduce_sum_product(TYPE* l1, TYPE* conn, int n1, int n2, TYPE* out) 
{
    TYPE sum = 0.0;
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t cid = tid / (n1+1);
    uint64_t rid = tid % (n1+1);

    out[rid*(n2+1)+cid] = conn[rid*(n2+1)+cid] * l1[rid];
    for (uint32_t i = (n1+1)/2; i > 0; i >>= 1) {
        if (rid <= i && rid+i <= n1) {
            sum += out[rid*(n2+1)+cid];
        }
    }
    __syncthreads();
    return sum;
}

__global__ void bpnn_layer_forward(TYPE* l1, TYPE* l2, TYPE* conn, int n1, int n2, TYPE* out)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t cid = tid / (n1+1);
    uint64_t rid = tid % (n1+1);
    TYPE sum = 0.0;

    l1[0] = 1.0;
    /*** For each unit in second layer ***/
    //for (j = 1; j <= n2; j++) {
        /*** Compute weighted sum of its inputs ***/
    //    sum = 0.0;
        //for (k = 0; k <= n1; k++) {	
        //    sum += conn[k][j] * l1[k]; 
        //}
    sum += reduce_sum_product(l1, conn, n1, n2, out);
    
    //l2[j] = squash(sum);
    if (rid == 0) {
        l2[cid] = (1.0 / (1.0 + exp(-sum)));
    }
    //}
}

__global__ void bpnn_output_error_d(TYPE* delta, TYPE* target, TYPE* output, int nj, TYPE* err) 
{
    int j;
    TYPE o, t, errsum;
    errsum = 0.0;
    for (j = 1; j <= nj; j++) {
        o = output[j];
        t = target[j];
        delta[j] = o * (1.0 - o) * (t - o);
        errsum += ABS(delta[j]);
    }
    *err = errsum;
}

__global__ void bpnn_adjust_weights_cuda(TYPE * delta,   
        int64_t hid,         
        TYPE * ly,      
        int64_t in,          
        TYPE * w,       
        TYPE * oldw)  									
{
    //int by = blockIdx.y;
    int64_t by = blockIdx.x;
    int64_t tx = threadIdx.x;
    int64_t ty = threadIdx.y;

    int64_t index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
    int64_t index_y = HEIGHT * by + ty + 1;
    int64_t index_x = tx + 1;
    //eta = 0.3;
    //momentum = 0.3;

    w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
    oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

    __syncthreads();

    if (ty == 0 && by ==0) {
        w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
        oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
    }
}

__global__ void bpnn_adjust_weights_cuda(array_d_t<TYPE>* delta,   
        int64_t hid,         
        array_d_t<TYPE>* ly,      
        int64_t in,          
        array_d_t<TYPE>* w,       
        array_d_t<TYPE>* oldw)  									
{
    //int by = blockIdx.y;
    int64_t by = blockIdx.x;
    int64_t tx = threadIdx.y;
    int64_t ty = threadIdx.x;

    int64_t index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
    int64_t index_y = HEIGHT * by + ty + 1;
    int64_t index_x = tx + 1;
    //eta = 0.3;
    //momentum = 0.3;

    //w[index] += ((ETA * delta[index_x] * ly->seq_read(index_y)) + (MOMENTUM * oldw[index]));
    w->seq_write(index, w->seq_read(index)+((ETA * delta->seq_read(index_x) * ly->seq_read(index_y)) + (MOMENTUM * oldw->seq_read(index))));
    //oldw[index] = ((ETA * delta[index_x] * ly->seq_read(index_y)) + (MOMENTUM * oldw[index]));
    oldw->seq_write(index, ((ETA * delta->seq_read(index_x) * ly->seq_read(index_y)) + (MOMENTUM * oldw->seq_read(index))));

    __syncthreads();

    if (ty == 0 && by ==0) {
        //w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
        w->seq_write(index_x, w->seq_read(index_x)+((ETA * delta->seq_read(index_x)) + (MOMENTUM * oldw->seq_read(index_x))));
        //oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
        oldw->seq_write(index_x, ((ETA * delta->seq_read(index_x)) + (MOMENTUM * oldw->seq_read(index_x))));
    }
}



////////////////////////////////////////////////////////////////////////////////

void bpnn_layerforward(TYPE *l1, TYPE *l2, TYPE **conn, int n1, int n2);

void bpnn_output_error(TYPE *delta, TYPE *target, TYPE *output, int nj, TYPE *err);

void bpnn_hidden_error(TYPE *delta_h, int nh, TYPE *delta_o, int no, TYPE **who, TYPE *hidden, TYPE *err);

void bpnn_adjust_weights(TYPE *delta, int ndelta, TYPE *ly, int nly, TYPE **w, TYPE **oldw);

int setup(int argc, char** argv);

TYPE **alloc_2d_dbl(int m, int n);

TYPE squash(TYPE x);

//extern char *strcpy();
//extern void exit();

uint64_t layer_size = 0;

void
load(BPNN* net)
//    BPNN *net
{
    TYPE *units;
    int nr, nc, imgsize, i, j, k;

    nr = layer_size;

    imgsize = nr * nc;
    units = net->input_units;

    k = 1;
    for (i = 0; i < nr; i++) {
        units[k] = (TYPE) rand()/RAND_MAX ;
        k++;
    }
}

/*** Return random number between 0.0 and 1.0 ***/
TYPE drnd()
{
    return ((TYPE) rand() / (TYPE) BIGRND);
}

/*** Return random number between -1.0 and 1.0 ***/
TYPE dpn1()
{
    return ((drnd() * 2.0) - 1.0);
}

/*** The squashing function.  Currently, it's a sigmoid. ***/

__host__ __device__
TYPE squash(int x)
{
    TYPE m;
    //x = -x;
    //m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
    //return(1.0 / (1.0 + m));
    return (1.0 / (1.0 + exp(-x)));
}

__host__ __device__
TYPE squash(TYPE x)
{
    TYPE m;
    //x = -x;
    //m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
    //return(1.0 / (1.0 + m));
    return (1.0 / (1.0 + exp(-x)));
}



/*** Allocate 1d array of TYPEs ***/

extern "C"
TYPE *alloc_1d_dbl(int n)
//    int n;
{
    TYPE *_new;

    _new = (TYPE *) malloc ((unsigned) (n * sizeof (TYPE)));
    if (_new == NULL) {
        printf("ALLOC_1D_DBL: Couldn't allocate array of TYPEs\n");
        return (NULL);
    }
    return (_new);
}


/*** Allocate 2d array of TYPEs ***/

TYPE **alloc_2d_dbl(int m, int n)
//    int m, n;
{
    int i;
    TYPE **_new;

    _new = (TYPE **) malloc ((unsigned) (m * sizeof (TYPE *)));
    if (_new == NULL) {
        printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
        return (NULL);
    }

    for (i = 0; i < m; i++) {
        _new[i] = alloc_1d_dbl(n);
    }

    return (_new);
}

void
bpnn_randomize_weights(TYPE** w, int m, int n)
//    TYPE **w;
//    int m, n;
{
    int i, j;

    for (i = 0; i <= m; i++) {
        for (j = 0; j <= n; j++) {
            w[i][j] = (TYPE) rand()/RAND_MAX;
            //  w[i][j] = dpn1();
        }
    }
}

void
bpnn_randomize_row(TYPE* w, int m)
//    TYPE *w;
//    int m;
{
    int i;
    for (i = 0; i <= m; i++) {
        //w[i] = (TYPE) rand()/RAND_MAX;
        w[i] = 0.1;
    }
}

void
bpnn_zero_weights(TYPE** w, int m, int n)
//    TYPE **w;
//    int m, n;
{
    int i, j;

    for (i = 0; i <= m; i++) {
        for (j = 0; j <= n; j++) {
            w[i][j] = 0.0;
        }
    }
}


void bpnn_initialize(int seed)
{
    printf("Random number generator seed: %d\n", seed);
    srand(seed);
}


BPNN *bpnn_internal_create(uint64_t n_in, uint64_t n_hidden, uint64_t n_out)
//    int n_in, n_hidden, n_out;
{
    BPNN *newnet;
    uint64_t cnt = 0;
    uint64_t total_pages = 0;

    //newnet = (BPNN *) malloc (sizeof (BPNN));
    newnet = new BPNN;
    if (newnet == NULL) {
        printf("BPNN_CREATE: Couldn't allocate neural network\n");
        return (NULL);
    }

    newnet->input_n = n_in;
    num_blocks = n_in / 16;
    newnet->hidden_n = n_hidden;
    newnet->output_n = n_out;
    
    if (settings.memalloc == BAFS_DIRECT) {
        newnet->n_input_units_pages = (NUM_INPUTS * (n_in+1) * sizeof(TYPE) + settings.pageSize-1) / settings.pageSize;
        uint64_t offset = 0;
        for (uint64_t i = 0; i < NUM_INPUTS; i++) {
            newnet->vec_range_input_units[i].resize(1);
            newnet->h_range_input_units[i] = new range_t<TYPE>(0, (n_in+1), total_pages, newnet->n_input_units_pages/NUM_INPUTS, 0, settings.pageSize, h_pc, settings.cudaDevice);
            newnet->vec_range_input_units[i][0] = newnet->h_range_input_units[i];
            newnet->h_input_units_array[i] = new array_t<TYPE>(n_in+1, total_pages*settings.pageSize/*disk start offset*/, newnet->vec_range_input_units[i], settings.cudaDevice, cnt++);       
            
            offset += ((n_in+1) * sizeof(TYPE) + settings.pageSize-1) / settings.pageSize /512;
            
            total_pages += newnet->n_input_units_pages/NUM_INPUTS;
        }

        //total_pages += newnet->n_input_units_pages;
    }
    newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
    newnet->output_units = alloc_1d_dbl(n_out + 1);

    newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
    newnet->output_delta = alloc_1d_dbl(n_out + 1);
    newnet->target = alloc_1d_dbl(n_out + 1);

    if (settings.memalloc == BAFS_DIRECT) {
        newnet->vec_range_input_weights.resize(1);
        newnet->n_input_weights_pages = ((n_in+1) * (n_hidden+1) * sizeof(TYPE) + settings.pageSize-1) / settings.pageSize;
        newnet->h_range_input_weights = new range_t<TYPE>(0, (n_in+1) * (n_hidden+1), 4194304/*total_pages*/, newnet->n_input_weights_pages, 0, settings.pageSize, h_pc, settings.cudaDevice);
        newnet->vec_range_input_weights[0] = newnet->h_range_input_weights;
        newnet->h_input_weights_array = new array_t<TYPE>((n_in+1) * (n_hidden+1), total_pages*settings.pageSize, newnet->vec_range_input_weights, settings.cudaDevice, cnt++);

        total_pages += newnet->n_input_weights_pages;
    }


    newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

    if (settings.memalloc == BAFS_DIRECT) {
        newnet->vec_range_input_prev_weights.resize(1);
        newnet->n_input_prev_weights_pages = ((n_in+1) * (n_hidden+1) * sizeof(TYPE) + settings.pageSize-1)/ settings.pageSize;
        newnet->h_range_input_prev_weights = new range_t<TYPE>(0, (n_in+1) * (n_hidden+1), 8388604/*total_pages*/, newnet->n_input_prev_weights_pages, 0, settings.pageSize, h_pc, settings.cudaDevice);
        newnet->vec_range_input_prev_weights[0] = newnet->h_range_input_prev_weights;
        newnet->h_input_prev_weights_array = new array_t<TYPE>((n_in+1) * (n_hidden+1), total_pages*settings.pageSize, newnet->vec_range_input_prev_weights, settings.cudaDevice, cnt++);

        total_pages += newnet->n_input_prev_weights_pages;
    }


    if (settings.memalloc == BAFS_DIRECT) {
        //cudaMalloc((void**) &output_hidden_cuda, (hid + 1) * sizeof(TYPE));
        newnet->vec_range_output_hidden_units.resize(1);
        newnet->n_output_hidden_pages = ((n_hidden + 1)*sizeof(TYPE) + settings.pageSize-1) / settings.pageSize;
        newnet->h_range_output_hidden_units = new range_t<TYPE>(0, (n_hidden+1), total_pages, newnet->n_output_hidden_pages, 0, settings.pageSize, h_pc, settings.cudaDevice);
        newnet->vec_range_output_hidden_units[0] = newnet->h_range_output_hidden_units;
        newnet->h_output_hidden_units_array = new array_t<TYPE>((n_hidden+1), total_pages*settings.pageSize, newnet->vec_range_output_hidden_units, settings.cudaDevice, cnt++);

        total_pages += newnet->n_output_hidden_pages;

        //cudaMalloc((void**) &hidden_partial_sum, num_blocks * WIDTH * sizeof(TYPE));
        newnet->vec_range_hidden_partial_sum.resize(1);
        newnet->n_hidden_partial_sum_pages = ((uint64_t)num_blocks*WIDTH*sizeof(TYPE) + settings.pageSize-1) / settings.pageSize;
        newnet->h_range_hidden_partial_sum = new range_t<TYPE>(0, (uint64_t)num_blocks * WIDTH, total_pages, newnet->n_hidden_partial_sum_pages, 0, settings.pageSize, h_pc, settings.cudaDevice);
        newnet->vec_range_hidden_partial_sum[0] = newnet->h_range_hidden_partial_sum;
        newnet->h_hidden_partial_sum_array = new array_t<TYPE>((uint64_t)num_blocks * WIDTH, total_pages*settings.pageSize, newnet->vec_range_hidden_partial_sum, settings.cudaDevice, cnt++);

        total_pages += newnet->n_hidden_partial_sum_pages;

        //cudaMalloc((void**) &buffers[0], hid*num_blocks/1024*sizeof(TYPE));
        uint64_t bsize= 512;
        newnet->n_buffer_pages = (n_hidden * num_blocks / bsize * sizeof(TYPE) + settings.pageSize-1) / settings.pageSize;
        newnet->vec_range_buffer[0].resize(1);
        newnet->h_range_buffer[0] = new range_t<TYPE>(0, (uint64_t)n_hidden * num_blocks / bsize, total_pages, newnet->n_buffer_pages, 0, settings.pageSize, h_pc, settings.cudaDevice);
        newnet->vec_range_buffer[0][0] = newnet->h_range_buffer[0];
        newnet->h_buffer_array[0] = new array_t<TYPE>((uint64_t)n_hidden * num_blocks / bsize, total_pages*settings.pageSize, newnet->vec_range_buffer[0], settings.cudaDevice, cnt++);
        total_pages += newnet->n_buffer_pages;

        newnet->vec_range_buffer[1].resize(1);
        newnet->h_range_buffer[1] = new range_t<TYPE>(0, (uint64_t)n_hidden * num_blocks / bsize, total_pages, newnet->n_buffer_pages, 0, settings.pageSize, h_pc, settings.cudaDevice);
        newnet->vec_range_buffer[1][0] = newnet->h_range_buffer[1];
        newnet->h_buffer_array[1] = new array_t<TYPE>((uint64_t)n_hidden * num_blocks / bsize, total_pages*settings.pageSize, newnet->vec_range_buffer[1], settings.cudaDevice, cnt++);
        total_pages += newnet->n_buffer_pages;

        // output units
        newnet->vec_range_output_units.resize(1);
        newnet->n_output_units_pages = ((uint64_t)(n_out+1)*sizeof(TYPE) + settings.pageSize-1) / settings.pageSize;
        newnet->h_range_output_units = new range_t<TYPE>(0, (uint64_t)n_out+1, total_pages, newnet->n_output_units_pages, 0, settings.pageSize, h_pc, settings.cudaDevice);
        newnet->vec_range_output_units[0] = newnet->h_range_output_units;
        //newnet->h_output_units_array = new array_t<TYPE>((uint64_t)num_blocks * WIDTH, total_pages*settings.pageSize, newnet->vec_range_output_units, settings.cudaDevice, cnt++);
        newnet->h_output_units_array = new array_t<TYPE>((uint64_t)n_out+1, total_pages*settings.pageSize, newnet->vec_range_output_units, settings.cudaDevice, cnt++);
        total_pages += newnet->n_output_units_pages;

        // hidden weights
        //newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
        newnet->vec_range_hidden_weights.resize(1);
        newnet->n_hidden_weights_pages = ((uint64_t)(n_hidden+1)*(n_out+1)*sizeof(TYPE) + settings.pageSize-1) / settings.pageSize;
        newnet->h_range_hidden_weights = new range_t<TYPE>(0, (uint64_t)(n_hidden+1)*(n_out+1), total_pages, newnet->n_hidden_weights_pages, 0, settings.pageSize, h_pc, settings.cudaDevice);
        newnet->vec_range_hidden_weights[0] = newnet->h_range_hidden_weights;
        newnet->h_hidden_weights_array = new array_t<TYPE>((uint64_t)(n_hidden+1)*(n_out+1), total_pages*settings.pageSize, newnet->vec_range_hidden_weights, settings.cudaDevice, cnt++);
        total_pages += newnet->n_hidden_weights_pages;

        // output delta
        //newnet->output_delta = alloc_1d_dbl(n_out + 1);
        newnet->vec_range_output_delta.resize(1);
        newnet->n_output_delta_pages = ((uint64_t)(n_out+1)*sizeof(TYPE) + settings.pageSize-1) / settings.pageSize;
        newnet->h_range_output_delta = new range_t<TYPE>(0, (uint64_t)(n_out+1), total_pages, newnet->n_output_delta_pages, 0, settings.pageSize, h_pc, settings.cudaDevice);
        newnet->vec_range_output_delta[0] = newnet->h_range_output_delta;
        newnet->h_output_delta_array = new array_t<TYPE>((uint64_t)(n_out+1), total_pages*settings.pageSize, newnet->vec_range_output_delta, settings.cudaDevice, cnt++);
        total_pages += newnet->n_output_delta_pages;

        // target
        //newnet->target = alloc_1d_dbl(n_out + 1);
        newnet->vec_range_target.resize(1);
        newnet->n_target_pages = ((uint64_t)(n_out+1)*sizeof(TYPE) + settings.pageSize-1) / settings.pageSize;
        newnet->h_range_target = new range_t<TYPE>(0, (uint64_t)(n_out+1), total_pages, newnet->n_target_pages, 0, settings.pageSize, h_pc, settings.cudaDevice);
        newnet->vec_range_target[0] = newnet->h_range_target;
        newnet->h_target_array = new array_t<TYPE>((uint64_t)(n_out+1), total_pages*settings.pageSize, newnet->vec_range_target, settings.cudaDevice, cnt++);
        total_pages += newnet->n_target_pages;

        // hidden delta
        //newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
        newnet->vec_range_hidden_delta.resize(1);
        newnet->n_hidden_delta_pages = ((uint64_t)(n_hidden+1)*sizeof(TYPE) + settings.pageSize-1) / settings.pageSize;
        newnet->h_range_hidden_delta = new range_t<TYPE>(0, (uint64_t)(n_hidden+1), total_pages, newnet->n_hidden_delta_pages, 0, settings.pageSize, h_pc, settings.cudaDevice);
        newnet->vec_range_hidden_delta[0] = newnet->h_range_hidden_delta;
        newnet->h_hidden_delta_array = new array_t<TYPE>((uint64_t)(n_hidden+1), total_pages*settings.pageSize, newnet->vec_range_hidden_delta, settings.cudaDevice, cnt++);
        total_pages += newnet->n_hidden_delta_pages;

        // hidden prev weights  
        //newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
        newnet->vec_range_hidden_prev_weights.resize(1);
        newnet->n_hidden_prev_weights_pages = ((n_hidden+1) * (n_out+1) * sizeof(TYPE) + settings.pageSize-1)/ settings.pageSize;
        newnet->h_range_hidden_prev_weights = new range_t<TYPE>(0, (n_hidden+1) * (n_out+1), total_pages, newnet->n_hidden_prev_weights_pages, 0, settings.pageSize, h_pc, settings.cudaDevice);
        newnet->vec_range_hidden_prev_weights[0] = newnet->h_range_hidden_prev_weights;
        newnet->h_hidden_prev_weights_array = new array_t<TYPE>((n_hidden+1) * (n_out+1), total_pages*settings.pageSize, newnet->vec_range_hidden_prev_weights, settings.cudaDevice, cnt++);
        total_pages += newnet->n_hidden_prev_weights_pages;
    }

    printf("total pages %lu (%lu bytes)\n", total_pages, total_pages*settings.pageSize);
    return (newnet);
}


void bpnn_free(BPNN* net)
//    BPNN *net;
{
    int n1, n2, i;

    n1 = net->input_n;
    n2 = net->hidden_n;

    free((char *) net->input_units);
    free((char *) net->hidden_units);
    free((char *) net->output_units);

    free((char *) net->hidden_delta);
    free((char *) net->output_delta);
    free((char *) net->target);

    for (i = 0; i <= n1; i++) {
        free((char *) net->input_weights[i]);
        free((char *) net->input_prev_weights[i]);
    }
    free((char *) net->input_weights);
    free((char *) net->input_prev_weights);

    for (i = 0; i <= n2; i++) {
        free((char *) net->hidden_weights[i]);
        free((char *) net->hidden_prev_weights[i]);
    }
    free((char *) net->hidden_weights);
    free((char *) net->hidden_prev_weights);

    free((char *) net);
}


/*** Creates a new fully-connected network from scratch,
  with the given numbers of input, hidden, and output units.
  Threshold units are automatically included.  All weights are
  randomly initialized.

  Space is also allocated for temporary storage (momentum weights,
  error computations, etc).
 ***/

BPNN *bpnn_create(uint64_t n_in, uint64_t n_hidden, uint64_t n_out)
    //int n_in, n_hidden, n_out;
{
    // n_in: n_layers
    BPNN *newnet;
    newnet = bpnn_internal_create(n_in, n_hidden, n_out);

    if (settings.memalloc == BAFS_DIRECT){

    }
    //bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
    //bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
    
    if (settings.memalloc == BAFS_DIRECT){

    }

    bpnn_randomize_row(newnet->target, n_out);
    return (newnet);
}


void bpnn_layerforward(TYPE* l1, TYPE* l2, TYPE** conn, int n1, int n2)
    //TYPE *l1, *l2, **conn;
    //int n1, n2;
{
    TYPE sum;
    int j, k;

    /*** Set up thresholding unit ***/
    l1[0] = 1.0;
#ifdef OPEN
    omp_set_num_threads(NUM_THREAD);
#pragma omp parallel for shared(conn, n1, n2, l1) private(k, j) reduction(+: sum) schedule(static)
#endif 
    /*** For each unit in second layer ***/
    for (j = 1; j <= n2; j++) {
        /*** Compute weighted sum of its inputs ***/
        sum = 0.0;
        for (k = 0; k <= n1; k++) {	
            sum += conn[k][j] * l1[k]; 
        }
        l2[j] = squash(sum);
    }
}

__global__ 
void bpnn_layerforward_kernel(array_d_t<TYPE>* l1, array_d_t<TYPE>* l2, array_d_t<TYPE>* conn, int n1, int n2)
{
    TYPE sum;
    int j, k;
    uint64_t tid = blockIdx.x*blockDim.x+threadIdx.x;

    l1->seq_write(0, 1.0);
    /*** For each unit in second layer ***/
    for (j = 1; j <= n2; j++) {
    //if (tid >= 1 && tid <= n2) {
        /*** Compute weighted sum of its inputs ***/
        sum = 0.0;
        for (k = 0; k <= n1; k++) {	
            sum += conn->seq_read(k*(n2+1)+tid) * l1->seq_read(k); 
        }
        l2->seq_write(tid, squash(sum));
    }
    __syncthreads();
}

//extern "C"
void bpnn_output_error(TYPE* delta, TYPE* target, TYPE* output, int nj, TYPE* err)  
    //TYPE *delta, *target, *output, *err;
    //int nj;
{
    int j;
    TYPE o, t, errsum;
    errsum = 0.0;
    for (j = 1; j <= nj; j++) {
        o = output[j];
        t = target[j];
        delta[j] = o * (1.0 - o) * (t - o);
        errsum += ABS(delta[j]);
    }
    *err = errsum;
}

__global__
void bpnn_output_error_kernel(array_d_t<TYPE>* delta, array_d_t<TYPE>* target, array_d_t<TYPE>* output, int nj)  
{
    int j;
    TYPE o, t, errsum;
    uint64_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    
    //if (tid >= 1 && tid <= nj) {
        o = output->seq_read(tid);
        t = target->seq_read(tid);
        //delta->seq_write(j, o * (1.0 - o) * (t - o));
        delta->seq_write(tid, o * (1.0 - o) * (t - o));
    //}
}



//extern "C"
void bpnn_hidden_error(TYPE* delta_h,   
        int nh, 
        TYPE* delta_o, 
        int no, 
        TYPE** who, 
        TYPE* hidden, 
        TYPE* err)
    //TYPE *delta_h, *delta_o, *hidden, **who, *err;
    //int nh, no;
{
    int j, k;
    TYPE h, sum, errsum;

    errsum = 0.0;
    for (j = 1; j <= nh; j++) {
        h = hidden[j];
        sum = 0.0;
        for (k = 1; k <= no; k++) {
            sum += delta_o[k] * who[j][k];
        }
        delta_h[j] = h * (1.0 - h) * sum;
        errsum += ABS(delta_h[j]);
    }
    *err = errsum;
}

__global__
void bpnn_hidden_error_kernel(array_d_t<TYPE>* delta_h,   
        int nh, 
        array_d_t<TYPE>* delta_o, 
        int no, 
        array_d_t<TYPE>* who, 
        array_d_t<TYPE>* hidden)
{
    int j, k;
    TYPE h, sum;
    uint32_t tid = blockIdx.x*blockDim.x+threadIdx.x;

    //for (j = 1; j <= nh; j++) {
    if (tid >= 1 && tid <= nh) {
        h = hidden->seq_read(tid);
        sum = 0.0;
        for (k = 1; k <= no; k++) {
            sum += delta_o->seq_read(k) * who->seq_read(tid*no+k);
        }
        delta_h->seq_write(tid, h * (1.0 - h) * sum);
    }
}




//extern "C"
void bpnn_adjust_weights(TYPE* delta, int ndelta, TYPE* ly, int nly, TYPE** w, TYPE** oldw)
    //TYPE *delta, *ly, **w, **oldw;
{
    TYPE new_dw;
    int k, j;
    ly[0] = 1.0;
    //eta = 0.3;
    //momentum = 0.3;

#ifdef OPEN
    omp_set_num_threads(NUM_THREAD);
#pragma omp parallel for  \
    shared(oldw, w, delta) \
    private(j, k, new_dw) \
    firstprivate(ndelta, nly, momentum) 
#endif 
    for (j = 1; j <= ndelta; j++) {
        for (k = 0; k <= nly; k++) {
            new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]));
            w[k][j] += new_dw;
            oldw[k][j] = new_dw;
        }
    }
}

__global__
void bpnn_adjust_weights_kernel(array_d_t<TYPE>* delta, int ndelta/*out*/, array_d_t<TYPE>* ly, int nly/*hid*/, array_d_t<TYPE>* w, array_d_t<TYPE>* oldw)
    //TYPE *delta, *ly, **w, **oldw;
{
    TYPE new_dw;
    int k, j;
    //ly->seq_write(0, 1.0);
    uint32_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid == 0) {
        ly->seq_write(0, 1.0);
    }

    //for (j = 1; j <= ndelta; j++) {
    if (tid >= 1 && tid <= ndelta) {
        for (k = 0; k <= nly; k++) {
            new_dw = ((ETA * delta->seq_read(tid) * ly->seq_read(k)) + (MOMENTUM * oldw->seq_read(k*(ndelta+1)+tid)));
            //w[k][j] += new_dw;
            w->seq_write(k*(ndelta+1)+j, w->seq_read(k*(ndelta+1)+j)+new_dw);
        }
    }
}

extern "C"
void bpnn_feedforward(BPNN* net)
    //BPNN *net;
{
    int in, hid, out;

    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;

    /*** Feed forward input activations. ***/
    bpnn_layerforward(net->input_units, net->hidden_units,
            net->input_weights, in, hid);
    bpnn_layerforward(net->hidden_units, net->output_units,
            net->hidden_weights, hid, out);

}


extern "C"
void bpnn_train(BPNN* net, TYPE* eo, TYPE* eh)
    //BPNN *net;
    //TYPE *eo, *eh;
{
    int in, hid, out;
    TYPE out_err, hid_err;

    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;

    /*** Feed forward input activations. ***/
    bpnn_layerforward(net->input_units, net->hidden_units,
            net->input_weights, in, hid);
    bpnn_layerforward(net->hidden_units, net->output_units,
            net->hidden_weights, hid, out);

    /*** Compute error on output and hidden units. ***/
    bpnn_output_error(net->output_delta, net->target, net->output_units,
            out, &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
            net->hidden_weights, net->hidden_units, &hid_err);
    *eo = out_err;
    *eh = hid_err;

    /*** Adjust input and hidden weights. ***/
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
            net->hidden_weights, net->hidden_prev_weights);
    bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
            net->input_weights, net->input_prev_weights);

}


extern "C"
void bpnn_save(BPNN* net, char* filename)
    //BPNN *net;
    //char *filename;
{
    int n1, n2, n3, i, j, memcnt;
    TYPE dvalue, **w;
    char *mem;
    ///add//
    FILE *pFile;
    pFile = fopen( filename, "w+" );
    ///////
    /*
       if ((fd = creat(filename, 0644)) == -1) {
       printf("BPNN_SAVE: Cannot create '%s'\n", filename);
       return;
       }
     */

    n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
    printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
    //fflush(stdout);

    //write(fd, (char *) &n1, sizeof(int));
    //write(fd, (char *) &n2, sizeof(int));
    //write(fd, (char *) &n3, sizeof(int));

    fwrite( (char *) &n1 , sizeof(char), sizeof(char), pFile);
    fwrite( (char *) &n2 , sizeof(char), sizeof(char), pFile);
    fwrite( (char *) &n3 , sizeof(char), sizeof(char), pFile);



    memcnt = 0;
    w = net->input_weights;
    mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(TYPE)));
    for (i = 0; i <= n1; i++) {
        for (j = 0; j <= n2; j++) {
            dvalue = w[i][j];
            fastcopy(&mem[memcnt], &dvalue, sizeof(TYPE));
            memcnt += sizeof(TYPE);
        }
    }
    //write(fd, mem, (n1+1) * (n2+1) * sizeof(TYPE));
    fwrite( mem , (unsigned)(sizeof(TYPE)), (unsigned) ((n1+1) * (n2+1) * sizeof(TYPE)) , pFile);
    free(mem);

    memcnt = 0;
    w = net->hidden_weights;
    mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(TYPE)));
    for (i = 0; i <= n2; i++) {
        for (j = 0; j <= n3; j++) {
            dvalue = w[i][j];
            fastcopy(&mem[memcnt], &dvalue, sizeof(TYPE));
            memcnt += sizeof(TYPE);
        }
    }
    //write(fd, mem, (n2+1) * (n3+1) * sizeof(TYPE));
    fwrite( mem , sizeof(TYPE), (unsigned) ((n2+1) * (n3+1) * sizeof(TYPE)) , pFile);
    free(mem);

    fclose(pFile);
    return;
}


extern "C"
BPNN *bpnn_read(char* filename)
    //char *filename;
{
    char *mem;
    BPNN *_new;
    int fd, n1, n2, n3, i, j, memcnt;

    if ((fd = open(filename, 0, 0644)) == -1) {
        return (NULL);
    }

    printf("Reading '%s'\n", filename);  //fflush(stdout);

    read(fd, (char *) &n1, sizeof(int));
    read(fd, (char *) &n2, sizeof(int));
    read(fd, (char *) &n3, sizeof(int));
    _new = bpnn_internal_create(n1, n2, n3);

    printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
    printf("Reading input weights...");  //fflush(stdout);

    memcnt = 0;
    mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(TYPE)));
    read(fd, mem, (n1+1) * (n2+1) * sizeof(TYPE));
    for (i = 0; i <= n1; i++) {
        for (j = 0; j <= n2; j++) {
            fastcopy(&(_new->input_weights[i][j]), &mem[memcnt], sizeof(TYPE));
            memcnt += sizeof(TYPE);
        }
    }
    free(mem);

    printf("Done\nReading hidden weights...");  //fflush(stdout);

    memcnt = 0;
    mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(TYPE)));
    read(fd, mem, (n2+1) * (n3+1) * sizeof(TYPE));
    for (i = 0; i <= n2; i++) {
        for (j = 0; j <= n3; j++) {
            fastcopy(&(_new->hidden_weights[i][j]), &mem[memcnt], sizeof(TYPE));
            memcnt += sizeof(TYPE);
        }
    }
    free(mem);
    close(fd);

    printf("Done\n");  //fflush(stdout);

    bpnn_zero_weights(_new->input_prev_weights, n1, n2);
    bpnn_zero_weights(_new->hidden_prev_weights, n2, n3);

    return (_new);
}


double gettime() {
    struct timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
    int
main( int argc, char** argv) 
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) cudaSetDevice(0);

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

    setup(argc, argv);
}


    extern "C"
void bpnn_train_cuda(BPNN *net, TYPE *eo, TYPE *eh)
{
    int in, hid, out;
    TYPE out_err, hid_err;

    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;   

#ifdef GPU  
    int m = 0;
    int m_h = 0;
    TYPE *input_hidden_cuda;
    TYPE *input_cuda;
    TYPE *output_hidden_cuda;
    TYPE *partial_sum;
    TYPE *hidden_partial_sum;
    TYPE *hidden_delta_cuda;
    TYPE *input_prev_weights_cuda;
    TYPE sum;
    TYPE *input_weights_one_dim;
    TYPE *hidden_weights_one_dim;
    TYPE *input_weights_prev_one_dim;
    num_blocks = in / 16;  
    //dim3  grid( 1 , num_blocks);
    dim3  grid(num_blocks, 1);
    dim3  threads(16 , 16);
    dim3  threads2(4, 4);
    
    void* hidden_units_d, *input_weights_one_dim_d, *output_units_d, *hidden_weights_one_dim_d;
    void* buffers[2];

    float time_elapsed;
    //printf("num blocks %u\n", num_blocks);
    
#endif

#ifdef CPU

    printf("Performing CPU computation\n");
    bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid);

#endif

#ifdef GPU

    printf("Performing GPU computation\n");
    printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);
        //printf("input-0 %p\n", net->h_input_units_array[0]->d_array_ptr);

    cudaEvent_t start, end;
    cudaEvent_t i_start, i_end;
    cuda_err_chk(cudaEventCreate(&start));
    cuda_err_chk(cudaEventCreate(&end));

    cuda_err_chk(cudaEventRecord(start, 0));

// 
//int iter = 0;
for (int iter = 0; iter < settings.iter; iter++) 
//for (int i = 0; i < settings.iter; i++) 
{
    auto t1 = high_resolution_clock::now();
    
    // Chia-Hao: 1119
#if USE_HOST_CACHE
    //if (iter == settings.iter-1) lastIteration(&(stream_mngr->kernel_stream));
#endif

    if (settings.memalloc == BAFS_DIRECT) {
#if USE_HOST_CACHE
        bpnn_layerforward_CUDA<<< grid, threads, 0, stream_mngr->kernel_stream >>>(
                net->h_input_units_array[iter]->d_array_ptr,
                net->h_output_hidden_units_array->d_array_ptr,
                net->h_input_weights_array->d_array_ptr,
                net->h_hidden_partial_sum_array->d_array_ptr,
                in,
                hid);

        cuda_err_chk(cudaStreamSynchronize(stream_mngr->kernel_stream));
        cudaError_t error = cudaGetLastError();
#else
        printf("input %p\n", net->h_input_units_array[iter]->d_array_ptr);
        bpnn_layerforward_CUDA<<< grid, threads >>>(
                net->h_input_units_array[iter]->d_array_ptr,
                net->h_output_hidden_units_array->d_array_ptr,
                net->h_input_weights_array->d_array_ptr,
                net->h_hidden_partial_sum_array->d_array_ptr,
                in,
                hid);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
        cudaDeviceSynchronize();
        error = cudaGetLastError();
#endif
        if (error != cudaSuccess) {
            printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    }

#endif

#if 0
    ///*
    cudaMemcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(TYPE), cudaMemcpyDeviceToHost);
    for (int j = 1; j <= hid; j++) {
        sum = 0.0;
        for (int k = 0; k < num_blocks; k++) {	
            sum += partial_sum[k * hid + j-1] ;
        }
        sum += net->input_weights[0][j];
        net->hidden_units[j] = TYPE(1.0 / (1.0 + exp(-sum)));
    }
    //*/
#else
    ///*
    int bsize = 512;
    int threads_in_block = (num_blocks > bsize) ? bsize : num_blocks;
    int gpu_blocks = hid*num_blocks/threads_in_block;

    int num_in_col = num_blocks;
    int cnt = 0;
    while (num_in_col > 1) {
        int threads_in_block = (num_in_col > bsize) ? bsize : num_in_col;
        int gpu_blocks = hid*num_in_col/threads_in_block;
        if (cnt == 0) {
            if (settings.memalloc == BAFS_DIRECT) {
                cudaError_t error;
#if USE_HOST_CACHE
                reduce_sum<<<gpu_blocks, threads_in_block, 0, stream_mngr->kernel_stream>>>(hid, num_in_col, net->h_hidden_partial_sum_array->d_array_ptr, net->h_buffer_array[1]->d_array_ptr);
                error = cudaGetLastError();
                if (error != cudaSuccess) {
                    printf("bpnn reduce sum1 kernel error: %s\n", cudaGetErrorString(error));
                    exit(EXIT_FAILURE);
                }
                cuda_err_chk(cudaStreamSynchronize(stream_mngr->kernel_stream));
#else
                reduce_sum<<<gpu_blocks, threads_in_block>>>(hid, num_in_col, net->h_hidden_partial_sum_array->d_array_ptr, net->h_buffer_array[1]->d_array_ptr);
                cudaDeviceSynchronize();
#endif

                std::swap(net->h_buffer_array[0], net->h_buffer_array[1]);
            }
        }
        else {
            if (settings.memalloc == BAFS_DIRECT) {
                cudaError_t error;
#if USE_HOST_CACHE
                reduce_sum<<<gpu_blocks, threads_in_block, 0, stream_mngr->kernel_stream>>>(hid, num_in_col, net->h_buffer_array[0]->d_array_ptr, net->h_buffer_array[1]->d_array_ptr);
                error = cudaGetLastError();
                if (error != cudaSuccess) {
                    printf("bpnn reduce sum2 kernel error: %s\n", cudaGetErrorString(error));
                    exit(EXIT_FAILURE);
                }
                cuda_err_chk(cudaStreamSynchronize(stream_mngr->kernel_stream));

#else
                reduce_sum<<<gpu_blocks, threads_in_block>>>(hid, num_in_col, net->h_buffer_array[0]->d_array_ptr, net->h_buffer_array[1]->d_array_ptr);
                cudaDeviceSynchronize();
#endif
                std::swap(net->h_buffer_array[0], net->h_buffer_array[1]);
            }
        }
        num_in_col /= threads_in_block;
        cnt++;
    }

    if (settings.memalloc == BAFS_DIRECT) {
        cudaError_t error;
#if USE_HOST_CACHE
        activate_kernel<<<1, hid, 0, stream_mngr->kernel_stream>>>(net->h_output_hidden_units_array->d_array_ptr, net->h_buffer_array[0]->d_array_ptr, net->h_input_weights_array->d_array_ptr, hid);
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("bpnn activate kernel error: %s\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }       
        cuda_err_chk(cudaStreamSynchronize(stream_mngr->kernel_stream));
#else
        activate_kernel<<<1, hid>>>(net->h_output_hidden_units_array->d_array_ptr, net->h_buffer_array[0]->d_array_ptr, net->h_input_weights_array->d_array_ptr, hid);
        cudaDeviceSynchronize();
#endif 
    }

    //cudaMemcpy(net->hidden_units, hidden_units_d, (hid+1) * sizeof(TYPE), cudaMemcpyDeviceToHost);
    //cudaMemcpy(net->output_units, output_units_d, (out+1) * sizeof(TYPE), cudaMemcpyDeviceToHost);
    //*/
#endif

    if (settings.memalloc == BAFS_DIRECT) {
        cudaError_t error;
#if USE_HOST_CACHE
        bpnn_layerforward_kernel<<<1, out, 0, stream_mngr->kernel_stream>>>(net->h_output_hidden_units_array->d_array_ptr, net->h_output_units_array->d_array_ptr, 
                                             net->h_hidden_weights_array->d_array_ptr, hid, out);
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "bpnn layerforward kernel error: %s\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
        else printf("bpnn layerforward kernel \n");
        cuda_err_chk(cudaStreamSynchronize(stream_mngr->kernel_stream));
///*
        bpnn_output_error_kernel<<<1, out, 0, stream_mngr->kernel_stream>>>(net->h_output_delta_array->d_array_ptr, net->h_target_array->d_array_ptr, net->h_output_units_array->d_array_ptr, out);
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "bpnn output error kernel error: %s\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
        else printf("bpnn output error kernel \n");
        cuda_err_chk(cudaStreamSynchronize(stream_mngr->kernel_stream));
        bpnn_hidden_error_kernel<<<1, hid, 0, stream_mngr->kernel_stream>>>(net->h_hidden_delta_array->d_array_ptr, hid, net->h_output_delta_array->d_array_ptr, out, 
                                          net->h_hidden_weights_array->d_array_ptr, net->h_output_hidden_units_array->d_array_ptr);  
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "bpnn hidden error kernel error: %s\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
        else printf("bpnn hidden error kernel \n");
        cuda_err_chk(cudaStreamSynchronize(stream_mngr->kernel_stream));
//*/
        bpnn_adjust_weights_kernel<<<1, out, 0, stream_mngr->kernel_stream>>>(net->h_output_delta_array->d_array_ptr, out, net->h_output_hidden_units_array->d_array_ptr, hid, 
                                            net->h_hidden_weights_array->d_array_ptr, net->h_hidden_prev_weights_array->d_array_ptr);
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "bpnn adjust weights kernel error: %s\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
        else printf("bpnn adjust weights kernel \n");
//*/
        cuda_err_chk(cudaStreamSynchronize(stream_mngr->kernel_stream));
#else
        bpnn_layerforward_kernel<<<1, out>>>(net->h_output_hidden_units_array->d_array_ptr, net->h_output_units_array->d_array_ptr, 
                                             net->h_hidden_weights_array->d_array_ptr, hid, out);
        bpnn_output_error_kernel<<<1, out>>>(net->h_output_delta_array->d_array_ptr, net->h_target_array->d_array_ptr, net->h_output_units_array->d_array_ptr, out);
        bpnn_hidden_error_kernel<<<1, hid>>>(net->h_hidden_delta_array->d_array_ptr, hid, net->h_output_delta_array->d_array_ptr, out, 
                                          net->h_hidden_weights_array->d_array_ptr, net->h_output_hidden_units_array->d_array_ptr);  
        bpnn_adjust_weights_kernel<<<1, out>>>(net->h_output_delta_array->d_array_ptr, out, net->h_output_hidden_units_array->d_array_ptr, hid, 
                                            net->h_hidden_weights_array->d_array_ptr, net->h_hidden_prev_weights_array->d_array_ptr);
        cudaDeviceSynchronize();
#endif    
    }

#ifdef CPU

    bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

#endif  


#ifdef GPU

    //cudaMalloc((void**) &hidden_delta_cuda, (hid + 1) * sizeof(TYPE));
    //cudaMalloc((void**) &input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(TYPE));

    if (settings.memalloc == BAFS_DIRECT) {
#if USE_HOST_CACHE
        //lastIteration(&(stream_mngr->kernel_stream));
        bpnn_adjust_weights_cuda<<< grid, threads, 0, stream_mngr->kernel_stream >>>(net->h_hidden_delta_array->d_array_ptr, hid, 
                                                      net->h_input_units_array[iter]->d_array_ptr, in,
                                                      net->h_input_weights_array->d_array_ptr, 
                                                      net->h_input_prev_weights_array->d_array_ptr);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "bpnn adjust weights cuda kernel error: %s\n", cudaGetErrorString(error));
            //exit(EXIT_FAILURE);
        }
        else printf("bpnn adjust weights cuda kernel \n");
        cuda_err_chk(cudaStreamSynchronize(stream_mngr->kernel_stream));
        
        //h_pc->flush_cache();
        //flushHostCache();
        //cudaDeviceSynchronize();

#else
        bpnn_adjust_weights_cuda<<< grid, threads >>>(net->h_hidden_delta_array->d_array_ptr, hid, 
                                                      net->h_input_units_array[iter]->d_array_ptr, in,
                                                      net->h_input_weights_array->d_array_ptr, 
                                                      net->h_input_prev_weights_array->d_array_ptr);
        cudaDeviceSynchronize();
        
        //h_pc->flush_cache();
        //cudadevicesynchronize();
#endif
    }

    auto t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "One iteration took " << time_span.count() << " seconds." << std::endl;   
    printf("iter %d\n", iter);

}
#if USE_HOST_CACHE
    h_pc->flush_cache();
    flushHostCache();
    cudaDeviceSynchronize();

#else
    h_pc->flush_cache();
    cudaDeviceSynchronize();
#endif
    cuda_err_chk(cudaEventRecord(end, 0));
    cuda_err_chk(cudaEventSynchronize(end));
    cuda_err_chk(cudaEventElapsedTime(&time_elapsed, start, end));
    printf("Elapsed Time %f ms\n", time_elapsed);
    
    if (settings.memalloc == BAFS_DIRECT) {
        print_stats(net);
        ctrls[0]->print_reset_stats();
    }

#endif   




}

void
backprop_face()
{
    BPNN *net;
    int i;
    TYPE out_err, hid_err;

        net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)

    printf("Input layer size : %d\n", layer_size);
    
    //entering the training kernel, only one iteration
#if USE_HOST_CACHE
    hc = createHostCache(ctrls[0], settings.maxPageCacheSize);

    uint64_t acc_pages = 0;

    for (int i = 0; i < NUM_INPUTS; i++)
        hc->registerRangesLBA(acc_pages*settings.pageSize/512); acc_pages += net->n_input_units_pages/NUM_INPUTS;
    
    //hc->registerRangesLBA(acc_pages*settings.pageSize/512); acc_pages += net->n_input_weights_pages;
    hc->registerRangesLBA(274877906944/512); acc_pages += net->n_input_weights_pages;
    //hc->registerRangesLBA(acc_pages*settings.pageSize/512); acc_pages += net->n_input_prev_weights_pages;
    hc->registerRangesLBA(549755813888/512); acc_pages += net->n_input_prev_weights_pages;
    hc->registerRangesLBA(acc_pages*settings.pageSize/512); acc_pages += net->n_output_hidden_pages;
    hc->registerRangesLBA(acc_pages*settings.pageSize/512); acc_pages += net->n_hidden_partial_sum_pages;
    hc->registerRangesLBA(acc_pages*settings.pageSize/512); acc_pages += net->n_buffer_pages;
    hc->registerRangesLBA(acc_pages*settings.pageSize/512); acc_pages += net->n_buffer_pages;
    hc->registerRangesLBA(acc_pages*settings.pageSize/512); acc_pages += net->n_output_units_pages;
    hc->registerRangesLBA(acc_pages*settings.pageSize/512); acc_pages += net->n_hidden_weights_pages;
    hc->registerRangesLBA(acc_pages*settings.pageSize/512); acc_pages += net->n_output_delta_pages;
    hc->registerRangesLBA(acc_pages*settings.pageSize/512); acc_pages += net->n_target_pages;
    hc->registerRangesLBA(acc_pages*settings.pageSize/512); acc_pages += net->n_hidden_delta_pages;
    hc->registerRangesLBA(acc_pages*settings.pageSize/512); acc_pages += net->n_hidden_prev_weights_pages;
#endif

    printf("Starting training kernel\n");
        bpnn_train_cuda(net, &out_err, &hid_err);
        //bpnn_free(net);

        //ctrls[0]->print_reset_stats();
        delete net;
    printf("Training done\n");
#if USE_HOST_CACHE
    //delete hc;
#endif
}

int setup(int argc, char* argv[])
    //int argc;
    //char *argv[];
{
    int seed;

    layer_size = settings.layer_size;

    if (layer_size % 16 != 0){
        fprintf(stderr, "The number of input points must be divided by 16\n");
        exit(0);
    }

    if (settings.memalloc == BAFS_DIRECT) {
        ctrls.resize(settings.n_ctrls);
        for (uint32_t i = 0; i < settings.n_ctrls; i++) {
            ctrls[i] = new Controller(sam_ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);
        }
        h_pc = new page_cache_t(settings.pageSize, settings.maxPageCacheSize/settings.pageSize, settings.cudaDevice, ctrls[0][0], 64, ctrls);
    }

    seed = 7;   
    bpnn_initialize(seed);
    backprop_face();

    exit(0);
}
