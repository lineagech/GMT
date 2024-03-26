#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "page_cache.h"
#include "settings.h"

#define BLOCK_SIZE 16
#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

typedef enum {
    GPUMEM = 0,
    UVM_READONLY = 1,
    UVM_DIRECT = 2,
    UVM_READONLY_NVLINK = 3,
    UVM_DIRECT_NVLINK = 4,
    DRAGON_MAP = 5,
    BAFS_DIRECT = 6,
} mem_type;



/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;
Settings settings; 

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

typedef double TYPE;

//const char* const sam_ctrls_paths[] = {"/dev/libnvm0"};
const char* const sam_ctrls_paths[] = {"/dev/libnvm_vmalloc0"};

void 
fatal(char *s)
{
    fprintf(stderr, "error: %s\n", s);
}

void writeoutput(TYPE *vect, size_t grid_rows, size_t grid_cols, char *file)
{
    int i,j, index=0;
    FILE *fp;
    char str[STR_SIZE];

    if( (fp = fopen(file, "w" )) == 0 )
        printf( "The file was not opened\n" );


    for (i=0; i < grid_rows; i++) 
        for (j=0; j < grid_cols; j++)
        {
            sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
            fputs(str,fp);
            index++;
        }

    fclose(fp);	
}


void readinput(TYPE *vect, size_t grid_rows, size_t grid_cols, char *file)
{
    int i,j;
    FILE *fp;
    char str[STR_SIZE];
    TYPE val;

    if( (fp  = fopen(file, "r" )) ==0 )
        printf( "The file was not opened\n" );


    for (i=0; i <= grid_rows-1; i++) 
        for (j=0; j <= grid_cols-1; j++)
        {
            fgets(str, STR_SIZE, fp);
            if (feof(fp))
                fatal("not enough lines in file");
            //if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
            if ((sscanf(str, "%f", &val) != 1))
                fatal("invalid file format");
            vect[i*grid_cols+j] = val;
        }

    fclose(fp);	

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void calculate_temp(size_t iteration,  //number of iteration
        TYPE *power,   //power input
        TYPE *temp_src,    //temperature input/output
        TYPE *temp_dst,    //temperature input/output
        size_t grid_cols,  //Col of grid
        size_t grid_rows,  //Row of grid
        size_t border_cols,  // border offset 
        size_t border_rows,  // border offset
        TYPE Cap,      //Capacitance
        TYPE Rx, 
        TYPE Ry, 
        TYPE Rz, 
        TYPE step, 
        TYPE time_elapsed){

    __shared__ TYPE temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ TYPE power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ TYPE temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

    TYPE amb_temp = 80.0;
    TYPE step_div_Cap;
    TYPE Rx_1,Ry_1,Rz_1;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    step_div_Cap=step/Cap;

    Rx_1=1/Rx;
    Ry_1=1/Ry;
    Rz_1=1/Rz;

    // each block finally computes result for a small block
    // after N iterations. 
    // it is the non-overlapping small blocks that cover 
    // all the input data

    // calculate the small block size
    int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
    int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE

    // calculate the boundary for the block according to 
    // the boundary of its small block
    int blkY = small_block_rows*by-border_rows;
    int blkX = small_block_cols*bx-border_cols;
    int blkYmax = blkY+BLOCK_SIZE-1;
    int blkXmax = blkX+BLOCK_SIZE-1;

    // calculate the global thread coordination
    int yidx = blkY+ty;
    int xidx = blkX+tx;

    // load data if it is within the valid input range
    int loadYidx=yidx, loadXidx=xidx;
    int index = grid_cols*loadYidx+loadXidx;

    if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
        temp_on_cuda[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
        power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
    }
    __syncthreads();

    // effective range within this block that falls within 
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validYmin = (blkY < 0) ? -blkY : 0;
    int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;

    int N = ty-1;
    int S = ty+1;
    int W = tx-1;
    int E = tx+1;

    N = (N < validYmin) ? validYmin : N;
    S = (S > validYmax) ? validYmax : S;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool computed;
    for (int i=0; i<iteration ; i++){ 
        computed = false;
        if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
                IN_RANGE(tx, validXmin, validXmax) && \
                IN_RANGE(ty, validYmin, validYmax) ) {
            computed = true;
            temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] + 
                    (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 + 
                    (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 + 
                    (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);

        }
        __syncthreads();
        if(i==iteration-1)
            break;
        if(computed)	 //Assign the computation range
            temp_on_cuda[ty][tx]= temp_t[ty][tx];
        __syncthreads();
    }

    // update the global memory
    // after the last iteration, only threads coordinated within the 
    // small block perform the calculation and switch on ``computed''
    if (computed){
        temp_dst[index]= temp_t[ty][tx];		
    }
}

std::vector<Controller*> ctrls;
page_cache_t* h_pc;
range_t<TYPE>* h_power_range;
range_t<TYPE>* h_temp_range[2];
std::vector<range_t<TYPE>*> vec_power;
std::vector<range_t<TYPE>*> vec_temp[2];
array_t<TYPE>* h_power_array;
array_t<TYPE>* h_temp_array[2];
#if USE_HOST_CACHE
HostCache* hc = NULL;
#endif

__global__ void calculate_temp(
        size_t iteration,  //number of iteration
        array_d_t<TYPE> *power,   //power input
        array_d_t<TYPE> *temp_src,    //temperature input/output
        array_d_t<TYPE> *temp_dst,    //temperature input/output
        size_t grid_cols,  //Col of grid
        size_t grid_rows,  //Row of grid
        size_t border_cols,  // border offset 
        size_t border_rows,  // border offset
        TYPE Cap,      //Capacitance
        TYPE Rx, 
        TYPE Ry, 
        TYPE Rz, 
        TYPE step, 
        TYPE time_elapsed){

    __shared__ TYPE temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ TYPE power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ TYPE temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

    TYPE amb_temp = 80.0;
    TYPE step_div_Cap;
    TYPE Rx_1,Ry_1,Rz_1;

    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;

    size_t tx=threadIdx.x;
    size_t ty=threadIdx.y;

    step_div_Cap=step/Cap;

    Rx_1=1/Rx;
    Ry_1=1/Ry;
    Rz_1=1/Rz;

    // each block finally computes result for a small block
    // after N iterations. 
    // it is the non-overlapping small blocks that cover 
    // all the input data

    // calculate the small block size
    size_t small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
    size_t small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE

    // calculate the boundary for the block according to 
    // the boundary of its small block
    size_t blkY = small_block_rows*by-border_rows;
    size_t blkX = small_block_cols*bx-border_cols;
    size_t blkYmax = blkY+BLOCK_SIZE-1;
    size_t blkXmax = blkX+BLOCK_SIZE-1;

    // calculate the global thread coordination
    size_t yidx = blkY+ty;
    size_t xidx = blkX+tx;

    // load data if it is within the valid input range
    size_t loadYidx=yidx, loadXidx=xidx;
    size_t index = grid_cols*loadYidx+loadXidx;

    if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
        temp_on_cuda[ty][tx] = temp_src->seq_read(index);  // Load the temperature data from global memory to shared memory
        power_on_cuda[ty][tx] = power->seq_read(index);// Load the power data from global memory to shared memory
    }
    __syncthreads();

    // effective range within this block that falls within 
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    size_t validYmin = (blkY < 0) ? -blkY : 0;
    size_t validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
    size_t validXmin = (blkX < 0) ? -blkX : 0;
    size_t validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;

    size_t N = ty-1;
    size_t S = ty+1;
    size_t W = tx-1;
    size_t E = tx+1;

    N = (N < validYmin) ? validYmin : N;
    S = (S > validYmax) ? validYmax : S;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool computed;
    for (int i=0; i<iteration ; i++){ 
        computed = false;
        if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
                IN_RANGE(tx, validXmin, validXmax) && \
                IN_RANGE(ty, validYmin, validYmax) ) {
            computed = true;
            temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] + 
                    (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 + 
                    (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 + 
                    (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);

            // Chia-Hao: for verification
            temp_t[ty][tx] = temp_on_cuda[ty][tx]+1;

        }
        __syncthreads();
        if(i==iteration-1)
            break;
        if(computed)	 //Assign the computation range
            temp_on_cuda[ty][tx]= temp_t[ty][tx];
        __syncthreads();
    }

    // update the global memory
    // after the last iteration, only threads coordinated within the 
    // small block perform the calculation and switch on ``computed''
    if (computed){
        temp_dst->seq_write(index, temp_t[ty][tx]);
    }
}


/*
   compute N time steps
 */

int compute_tran_temp(TYPE *MatrixPower,TYPE *MatrixTemp[2], size_t col, size_t row, \
        size_t total_iterations, size_t num_iterations, size_t blockCols, size_t blockRows, size_t borderCols, size_t borderRows) 
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(blockCols, blockRows);  

    TYPE grid_height = chip_height / row;
    TYPE grid_width = chip_width / col;

    TYPE Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    TYPE Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    TYPE Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    TYPE Rz = t_chip / (K_SI * grid_height * grid_width);

    TYPE max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    TYPE step = PRECISION / max_slope;
    size_t t;
    TYPE time_elapsed;
    time_elapsed=0.001;

    int src = 1, dst = 0;

    for (t = 0; t < total_iterations; t += num_iterations) {
        int temp = src;
        src = dst;
        dst = temp;
        calculate_temp<<<dimGrid, dimBlock>>>(MIN(num_iterations, total_iterations-t), 
                                              MatrixPower, MatrixTemp[src], MatrixTemp[dst],
                                              col, row, borderCols, borderRows, Cap, 
                                              Rx, Ry, Rz, step, time_elapsed);
    }
    return dst;
}

int compute_tran_temp(size_t col, size_t row, size_t total_iterations, size_t num_iterations, size_t blockCols, size_t blockRows, size_t borderCols, size_t borderRows)
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(blockCols, blockRows);  

    auto start = std::chrono::high_resolution_clock::now();

    TYPE grid_height = chip_height / row;
    TYPE grid_width = chip_width / col;

    TYPE Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    TYPE Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    TYPE Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    TYPE Rz = t_chip / (K_SI * grid_height * grid_width);

    TYPE max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    TYPE step = PRECISION / max_slope;
    size_t t;
    TYPE time_elapsed;
    time_elapsed=0.001;

    int src = 1, dst = 0;

    for (t = 0; t < total_iterations; t += num_iterations) {
        auto t1 = std::chrono::high_resolution_clock::now();
        int temp = src;
        src = dst;
        dst = temp;
        #if USE_HOST_CACHE
        calculate_temp<<<dimGrid, dimBlock, 0, stream_mngr->kernel_stream>>>(MIN(num_iterations, total_iterations-t), 
                                              h_power_array->d_array_ptr, h_temp_array[src]->d_array_ptr, h_temp_array[src]->d_array_ptr,
                                              col, row, borderCols, borderRows, Cap, 
                                              Rx, Ry, Rz, step, time_elapsed);
        cudaStreamSynchronize(stream_mngr->kernel_stream);
        #else 
        calculate_temp<<<dimGrid, dimBlock>>>(MIN(num_iterations, total_iterations-t), 
                                              h_power_array->d_array_ptr, h_temp_array[src]->d_array_ptr, h_temp_array[src]->d_array_ptr,
                                              col, row, borderCols, borderRows, Cap, 
                                              Rx, Ry, Rz, step, time_elapsed);
        cudaDeviceSynchronize();
        #endif
        auto t2 = std::chrono::high_resolution_clock::now();
        printf("iter %ld (%ld ms)\n", t, std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count());
    }
    
    printf("flush gpu cache...\n");
    h_pc->flush_cache();
#if USE_HOST_CACHE
    flushHostCache();
#endif
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << std::dec << "Elapsed Time: " << elapsed.count() << " ms"<< std::endl;

    return dst;
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
    fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
    fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
    fprintf(stderr, "\t<sim_time>   - number of iterations\n");
    fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
    fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
    fprintf(stderr, "\t<output_file> - name of the output file\n");
    exit(1);
}

int main(int argc, char** argv)
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

    run(argc,argv);

    return EXIT_SUCCESS;
}

void init_page_cache(size_t size)
{
    size_t total_pages = (size_t)512*(size_t)1024*(size_t)1024*(size_t)1024/settings.pageSize;
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
    
    h_power_range = new range_t<TYPE>(0, size, total_pages, n_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_power.resize(1);
    vec_power[0] = h_power_range;
    h_power_array = new array_t<TYPE>(size, total_pages*pc_page_size, vec_power, settings.cudaDevice, cnt++);
    total_pages += n_pages;

    h_temp_range[0] = new range_t<TYPE>(0, size, total_pages, n_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_temp[0].resize(1);
    vec_temp[0][0] = h_temp_range[0];
    h_temp_array[0] = new array_t<TYPE>(size, total_pages*pc_page_size, vec_temp[0], settings.cudaDevice, cnt++);
    total_pages += n_pages;
    
    h_temp_range[1] = new range_t<TYPE>(0, size, total_pages, n_pages, 0, pc_page_size, h_pc, settings.cudaDevice);
    vec_temp[1].resize(1);
    vec_temp[1][0] = h_temp_range[1];
    h_temp_array[1] = new array_t<TYPE>(size, total_pages*pc_page_size, vec_temp[1], settings.cudaDevice, cnt++);
    total_pages += n_pages;

    printf("Page Cache Initialized\n");
    printf("Total pages %lu (%lu Mbytes)\n", total_pages, total_pages*pc_page_size/1024/1024);
    fflush(stdout);

#if USE_HOST_CACHE
    std::cerr << "creating Host Cache...\n";
    hc = createHostCache(ctrls[0], settings.maxPageCacheSize);
    
    size_t offset = (size_t)512*(size_t)1024*(size_t)1024*(size_t)1024/settings.pageSize;
    hc->registerRangesLBA(offset*pc_page_size/512); offset += n_pages;
    hc->registerRangesLBA(offset*pc_page_size/512); offset += n_pages;
    hc->registerRangesLBA(offset*pc_page_size/512); offset += n_pages;
#endif

}

void print_stats()
{
    h_power_array->print_reset_stats();
    h_temp_array[0]->print_reset_stats();
    h_temp_array[1]->print_reset_stats();

    ctrls[0]->print_reset_stats();
#if USE_HOST_CACHE
    revokeHostRuntime();
#endif
}


void run(int argc, char** argv)
{
    size_t size;
    //int grid_rows,grid_cols;
    TYPE *FilesavingTemp, *FilesavingPower, *MatrixOut; 
    char *tfile, *pfile, *ofile;

    size_t total_iterations = settings.total_iterations;
    size_t pyramid_height = settings.pyramid_height; // number of iterations

    //if (argc != 7)
    //    usage(argc, argv);
    //if((grid_rows = atoi(argv[1]))<=0||
    //        (grid_cols = atoi(argv[1]))<=0||
    //        (pyramid_height = atoi(argv[2]))<=0||
    //        (total_iterations = atoi(argv[3]))<=0)
    //    usage(argc, argv);

    //tfile=argv[4];
    //pfile=argv[5];
    //ofile=argv[6];
    
    size_t grid_rows = settings.grid_rows;
    size_t grid_cols = settings.grid_cols;

    size = grid_rows * grid_cols;
   
    /* --------------- pyramid parameters --------------- */
# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
    size_t borderCols = (pyramid_height)*EXPAND_RATE/2;
    size_t borderRows = (pyramid_height)*EXPAND_RATE/2;
    size_t smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    size_t smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    size_t blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    size_t blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    if (settings.memalloc == GPUMEM) {
        FilesavingTemp = (TYPE *) malloc(size*sizeof(TYPE));
        FilesavingPower = (TYPE *) malloc(size*sizeof(TYPE));
        MatrixOut = (TYPE *) calloc (size, sizeof(TYPE));

        if( !FilesavingPower || !FilesavingTemp || !MatrixOut)
            fatal("unable to allocate memory");

        printf("pyramidHeight: %ld\ngridSize: [%ld, %ld]\nborder:[%ld, %ld]\nblockGrid:[%ld, %ld]\ntargetBlock:[%ld, %ld]\n",\
                pyramid_height, grid_cols, grid_rows, borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow);

        readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
        readinput(FilesavingPower, grid_rows, grid_cols, pfile);

        TYPE *MatrixTemp[2], *MatrixPower;
        cudaMalloc((void**)&MatrixTemp[0], sizeof(TYPE)*size);
        cudaMalloc((void**)&MatrixTemp[1], sizeof(TYPE)*size);
        cudaMemcpy(MatrixTemp[0], FilesavingTemp, sizeof(TYPE)*size, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&MatrixPower, sizeof(TYPE)*size);
        cudaMemcpy(MatrixPower, FilesavingPower, sizeof(TYPE)*size, cudaMemcpyHostToDevice);

        printf("Start computing the transient temperature\n");
        int ret = compute_tran_temp(MatrixPower, MatrixTemp, grid_cols, grid_rows, total_iterations, pyramid_height, blockCols, blockRows, borderCols, borderRows);
        printf("Ending simulation\n");
        cudaMemcpy(MatrixOut, MatrixTemp[ret], sizeof(TYPE)*size, cudaMemcpyDeviceToHost);

        writeoutput(MatrixOut, grid_rows, grid_cols, ofile);

        cudaFree(MatrixPower);
        cudaFree(MatrixTemp[0]);
        cudaFree(MatrixTemp[1]);
        free(MatrixOut);
    }
    else if (settings.memalloc == BAFS_DIRECT) {
        init_page_cache(size);

        int ret = compute_tran_temp(grid_cols, grid_rows, total_iterations, pyramid_height, blockCols, blockRows, borderCols, borderRows);
    
        print_stats();
    }
}
