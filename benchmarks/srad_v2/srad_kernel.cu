#include "srad.h"
#include <stdio.h>

__global__ void
srad_cuda_1(
		  TYPE *E_C, 
		  TYPE *W_C, 
		  TYPE *N_C, 
		  TYPE *S_C,
		  TYPE * J_cuda, 
		  TYPE * C_cuda, 
		  size_t cols, 
		  size_t rows, 
		  TYPE q0sqr
) 
{

  //block id
  size_t bx = blockIdx.x;
  size_t by = blockIdx.y;

  //thread id
  size_t tx = threadIdx.x;
  size_t ty = threadIdx.y;
  
  //indices
  size_t index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
  size_t index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
  size_t index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
  size_t index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
  size_t index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

  TYPE n, w, e, s, jc, g2, l, num, den, qsqr, c;

  //shared memory allocation
  __shared__ TYPE temp[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ TYPE temp_result[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ TYPE north[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ TYPE south[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ TYPE  east[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ TYPE  west[BLOCK_SIZE][BLOCK_SIZE];

  //load data to shared memory
  north[ty][tx] = J_cuda[index_n]; 
  south[ty][tx] = J_cuda[index_s];
  if ( by == 0 ){
  north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx]; 
  }
  else if ( by == gridDim.y - 1 ){
  south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
  }
   __syncthreads();
 
  west[ty][tx] = J_cuda[index_w];
  east[ty][tx] = J_cuda[index_e];

  if ( bx == 0 ){
  west[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + cols * ty]; 
  }
  else if ( bx == gridDim.x - 1 ){
  east[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1];
  }
 
  __syncthreads();
  
 

  temp[ty][tx]      = J_cuda[index];

  __syncthreads();

   jc = temp[ty][tx];

   if ( ty == 0 && tx == 0 ){ //nw
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
   }	    
   else if ( ty == 0 && tx == BLOCK_SIZE-1 ){ //ne
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
   }
   else if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx]  - jc;
   }
   else if ( ty == BLOCK_SIZE -1 && tx == 0 ){//sw
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
   }

   else if ( ty == 0 ){ //n
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else if ( tx == BLOCK_SIZE -1 ){ //e
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
   }
   else if ( ty == BLOCK_SIZE -1){ //s
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else if ( tx == 0 ){ //w
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else{  //the data elements which are not on the borders 
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }


    g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);

    l = ( n + s + w + e ) / jc;

	num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
	den  = 1 + (.25*l);
	qsqr = num/(den*den);

	// diffusion coefficent (equ 33)
	den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
	c = 1.0 / (1.0+den) ;

    // saturate diffusion coefficent
	if (c < 0){temp_result[ty][tx] = 0;}
	else if (c > 1) {temp_result[ty][tx] = 1;}
	else {temp_result[ty][tx] = c;}

    __syncthreads();

    C_cuda[index] = temp_result[ty][tx];
	E_C[index] = e;
	W_C[index] = w;
	S_C[index] = s;
	N_C[index] = n;

}

__global__ void
srad_cuda_2(
		  TYPE *E_C, 
		  TYPE *W_C, 
		  TYPE *N_C, 
		  TYPE *S_C,	
		  TYPE * J_cuda, 
		  TYPE * C_cuda, 
		  size_t cols, 
		  size_t rows, 
		  TYPE lambda,
		  TYPE q0sqr
) 
{
    //block id
	size_t bx = blockIdx.x;
    size_t by = blockIdx.y;

	//thread id
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

	//indices
    size_t index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
	size_t index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    size_t index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
	TYPE cc, cn, cs, ce, cw, d_sum;

	//shared memory allocation
	__shared__ TYPE south_c[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ TYPE  east_c[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ TYPE c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ TYPE c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ TYPE temp[BLOCK_SIZE][BLOCK_SIZE];

    //load data to shared memory
	temp[ty][tx]      = J_cuda[index];

    __syncthreads();
	 
	south_c[ty][tx] = C_cuda[index_s];

	if ( by == gridDim.y - 1 ){
	south_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
	}
	__syncthreads();
	 
	 
	east_c[ty][tx] = C_cuda[index_e];
	
	if ( bx == gridDim.x - 1 ){
	east_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1];
	}
	 
    __syncthreads();
  
    c_cuda_temp[ty][tx]      = C_cuda[index];

    __syncthreads();

	cc = c_cuda_temp[ty][tx];

   if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc; 
    ce  = east_c[ty][tx];
   } 
   else if ( tx == BLOCK_SIZE -1 ){ //e
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc; 
    ce  = east_c[ty][tx];
   }
   else if ( ty == BLOCK_SIZE -1){ //s
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc; 
    ce  = c_cuda_temp[ty][tx+1];
   }
   else{ //the data elements which are not on the borders 
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc; 
    ce  = c_cuda_temp[ty][tx+1];
   }

   // divergence (equ 58)
   d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

   // image update (equ 61)
   c_cuda_result[ty][tx] = temp[ty][tx] + 0.25 * lambda * d_sum;

   __syncthreads();
              
   J_cuda[index] = c_cuda_result[ty][tx];
    
}

__global__ void
srad_cuda_1(
		  array_d_t<TYPE> *E_C, 
		  array_d_t<TYPE> *W_C, 
		  array_d_t<TYPE> *N_C, 
		  array_d_t<TYPE> *S_C,
		  array_d_t<TYPE> * J_cuda, 
		  array_d_t<TYPE> * C_cuda, 
		  size_t cols, 
		  size_t rows, 
		  TYPE q0sqr
) 
{

    //block id
    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;

    //thread id
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    //indices
    size_t index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    size_t index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
    size_t index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    size_t index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
    size_t index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

    TYPE n, w, e, s, jc, g2, l, num, den, qsqr, c;

    //shared memory allocation
    __shared__ TYPE temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ TYPE temp_result[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ TYPE north[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ TYPE south[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ TYPE  east[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ TYPE  west[BLOCK_SIZE][BLOCK_SIZE];

    //load data to shared memory
    north[ty][tx] = J_cuda->seq_read(index_n); 
    __syncthreads();

    south[ty][tx] = J_cuda->seq_read(index_s);
    __syncthreads();
    
    
    
    if ( by == 0 ){
        north[ty][tx] = J_cuda->seq_read(BLOCK_SIZE * bx + tx); 
    }
    else if ( by == gridDim.y - 1 ){
        south[ty][tx] = J_cuda->seq_read(cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx);
    }
   
    __syncthreads();

    
    west[ty][tx] = J_cuda->seq_read(index_w);
    __syncthreads();

    east[ty][tx] = J_cuda->seq_read(index_e);

    if ( bx == 0 ){
        west[ty][tx] = J_cuda->seq_read(cols * BLOCK_SIZE * by + cols * ty); 
    }
    else if ( bx == gridDim.x - 1 ){
        east[ty][tx] = J_cuda->seq_read(cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1);
    }
    __syncthreads();

    ///*
    temp[ty][tx]      = J_cuda->seq_read(index);

    __syncthreads();

    //printf("checkpoint 1\n");
    
    jc = temp[ty][tx];

    if ( ty == 0 && tx == 0 ){ //nw
        n  = north[ty][tx] - jc;
        s  = temp[ty+1][tx] - jc;
        w  = west[ty][tx]  - jc; 
        e  = temp[ty][tx+1] - jc;
    }	    
    else if ( ty == 0 && tx == BLOCK_SIZE-1 ){ //ne
        n  = north[ty][tx] - jc;
        s  = temp[ty+1][tx] - jc;
        w  = temp[ty][tx-1] - jc; 
        e  = east[ty][tx] - jc;
    }
    else if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
        n  = temp[ty-1][tx] - jc;
        s  = south[ty][tx] - jc;
        w  = temp[ty][tx-1] - jc; 
        e  = east[ty][tx]  - jc;
    }
    else if ( ty == BLOCK_SIZE -1 && tx == 0 ){//sw
        n  = temp[ty-1][tx] - jc;
        s  = south[ty][tx] - jc;
        w  = west[ty][tx]  - jc; 
        e  = temp[ty][tx+1] - jc;
    }

    else if ( ty == 0 ){ //n
        n  = north[ty][tx] - jc;
        s  = temp[ty+1][tx] - jc;
        w  = temp[ty][tx-1] - jc; 
        e  = temp[ty][tx+1] - jc;
    }
    else if ( tx == BLOCK_SIZE -1 ){ //e
        n  = temp[ty-1][tx] - jc;
        s  = temp[ty+1][tx] - jc;
        w  = temp[ty][tx-1] - jc; 
        e  = east[ty][tx] - jc;
    }
    else if ( ty == BLOCK_SIZE -1){ //s
        n  = temp[ty-1][tx] - jc;
        s  = south[ty][tx] - jc;
        w  = temp[ty][tx-1] - jc; 
        e  = temp[ty][tx+1] - jc;
    }
    else if ( tx == 0 ){ //w
        n  = temp[ty-1][tx] - jc;
        s  = temp[ty+1][tx] - jc;
        w  = west[ty][tx] - jc; 
        e  = temp[ty][tx+1] - jc;
    }
    else{  //the data elements which are not on the borders 
        n  = temp[ty-1][tx] - jc;
        s  = temp[ty+1][tx] - jc;
        w  = temp[ty][tx-1] - jc; 
        e  = temp[ty][tx+1] - jc;
    }


    g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);

    l = ( n + s + w + e ) / jc;

    num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
    den  = 1 + (.25*l);
    qsqr = num/(den*den);

    // diffusion coefficent (equ 33)
    den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
    c = 1.0 / (1.0+den) ;

    // saturate diffusion coefficent
    if (c < 0){temp_result[ty][tx] = 0;}
    else if (c > 1) {temp_result[ty][tx] = 1;}
    else {temp_result[ty][tx] = c;}

    __syncthreads();
    
    //printf("checkpoint 2\n");
    
    C_cuda->seq_write(index, temp_result[ty][tx]);
    __syncthreads();
    /*
    E_C->seq_write(index, e);
    __syncthreads();
    W_C->seq_write(index, w);
    __syncthreads();
    S_C->seq_write(index, s);
    __syncthreads();
    N_C->seq_write(index, n);
    */
    //printf("checkpoint 3\n");
    if (by % 8 == 0 && ty == 0) printf("[%ld][%ld]index w %ld, index e %ld\n", by*BLOCK_SIZE+ty, bx*BLOCK_SIZE+tx, index_w, index_e);

}


__global__ void
srad_cuda_2(
		  array_d_t<TYPE> *E_C, 
		  array_d_t<TYPE> *W_C, 
		  array_d_t<TYPE> *N_C, 
		  array_d_t<TYPE> *S_C,	
		  array_d_t<TYPE> * J_cuda, 
		  array_d_t<TYPE> * C_cuda, 
		  size_t cols, 
		  size_t rows, 
		  TYPE lambda,
		  TYPE q0sqr
) 
{
    //block id
    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;

    //thread id
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    //indices
    size_t index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    size_t index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    size_t index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
    TYPE cc, cn, cs, ce, cw, d_sum;

    //shared memory allocation
    __shared__ TYPE south_c[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ TYPE  east_c[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ TYPE c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ TYPE c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ TYPE temp[BLOCK_SIZE][BLOCK_SIZE];

    //load data to shared memory
    temp[ty][tx]      = J_cuda->seq_read(index);
    //printf("checkpoint 2-0\n");

    __syncthreads();

    south_c[ty][tx] = C_cuda->seq_read(index_s);
    
    //printf("checkpoint 2-1\n");

    if ( by == gridDim.y - 1 ){
        south_c[ty][tx] = C_cuda->seq_read(cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx);
    }
    __syncthreads();


    east_c[ty][tx] = C_cuda->seq_read(index_e);

    if ( bx == gridDim.x - 1 ){
        east_c[ty][tx] = C_cuda->seq_read(cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1);
    }

    __syncthreads();

    c_cuda_temp[ty][tx]      = C_cuda->seq_read(index);

    __syncthreads();

    //printf("checkpoint 4\n");

    cc = c_cuda_temp[ty][tx];

    if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
        cn  = cc;
        cs  = south_c[ty][tx];
        cw  = cc; 
        ce  = east_c[ty][tx];
    } 
    else if ( tx == BLOCK_SIZE -1 ){ //e
        cn  = cc;
        cs  = c_cuda_temp[ty+1][tx];
        cw  = cc; 
        ce  = east_c[ty][tx];
    }
    else if ( ty == BLOCK_SIZE -1){ //s
        cn  = cc;
        cs  = south_c[ty][tx];
        cw  = cc; 
        ce  = c_cuda_temp[ty][tx+1];
    }
    else{ //the data elements which are not on the borders 
        cn  = cc;
        cs  = c_cuda_temp[ty+1][tx];
        cw  = cc; 
        ce  = c_cuda_temp[ty][tx+1];
    }

    // divergence (equ 58)
    d_sum = cn * N_C->seq_read(index) + cs * S_C->seq_read(index) + cw * W_C->seq_read(index) + ce * E_C->seq_read(index);

    // image update (equ 61)
    c_cuda_result[ty][tx] = temp[ty][tx] + 0.25 * lambda * d_sum;

    __syncthreads();
    //printf("checkpoint 5\n");

    J_cuda->seq_write(index, c_cuda_result[ty][tx]);

}


__global__ void
srad_cuda_mod_1(
		  array_d_t<TYPE> *E_C, 
		  array_d_t<TYPE> *W_C, 
		  array_d_t<TYPE> *N_C, 
		  array_d_t<TYPE> *S_C,
		  array_d_t<TYPE> * J_cuda, 
		  array_d_t<TYPE> * C_cuda, 
		  size_t cols, 
		  size_t rows, 
		  TYPE q0sqr,
          TYPE* Cptr,
          size_t offset = 0
) 
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x + offset;
    size_t tidy = tid / cols;
    size_t tidx = tid % cols;
    
    //block id
    size_t bx = tidx / BLOCK_SIZE; // blockIdx.x;
    size_t by = tidy / BLOCK_SIZE; // blockIdx.y;

    //thread id
    size_t tx = tidx % BLOCK_SIZE; // threadIdx.x;
    size_t ty = tidy % BLOCK_SIZE; // threadIdx.y;
    
    size_t gdim_y = rows/BLOCK_SIZE;
    size_t gdim_x = cols/BLOCK_SIZE;

    //indices
    size_t index   = tid;                           // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    size_t index_n = tid - cols * ty - cols;        // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
    size_t index_s = tid + cols * (BLOCK_SIZE-ty);  // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    size_t index_w = tid - tx - 1;                  // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
    size_t index_e = tid + (BLOCK_SIZE-tx);         // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

    TYPE n, w, e, s, jc, g2, l, num, den, qsqr, c;

    //shared memory allocation
    TYPE temp_ty_tx;
    TYPE temp_result_ty_tx;

    TYPE north_ty_tx;
    TYPE south_ty_tx;
    TYPE  east_ty_tx;
    TYPE  west_ty_tx;
    
    size_t index_r = tid + 1;                           // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx+1;
    size_t index_l = tid - 1;                           // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx-1;
    size_t index_d = tid + cols;                        // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * (ty+1) + tx;
    size_t index_u = tid - cols;                        // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * (ty-1) + tx;

    if (by == 0) index_n = tid - cols * ty;
    else if (by == gdim_y-1) index_s = tid + cols * (BLOCK_SIZE-1-ty);

    if (bx == 0) index_w = tid - tx;
    else if (bx == gdim_x-1) index_e = tid - tx + BLOCK_SIZE-1;
    
    //load data to shared memory
    north_ty_tx = J_cuda->seq_read(index_n);
    __syncthreads();
    south_ty_tx = J_cuda->seq_read(index_s);
    __syncthreads();
    west_ty_tx = J_cuda->seq_read(index_w);
    __syncthreads();
    east_ty_tx = J_cuda->seq_read(index_e);
    __syncthreads();
    temp_ty_tx = J_cuda->seq_read(index);
    __syncthreads();

    
    jc = temp_ty_tx;

    n  = J_cuda->seq_read(index_u) - jc;
    __syncthreads();
    s  = J_cuda->seq_read(index_d) - jc;
    __syncthreads();
    w  = J_cuda->seq_read(index_l) - jc; 
    __syncthreads();
    e  = J_cuda->seq_read(index_r) - jc;
    __syncthreads();


    if ( ty == 0 && tx == 0 ){ //nw
        n  = north_ty_tx - jc;
        s  = J_cuda->seq_read(index_d) - jc;
        w  = west_ty_tx - jc; 
        e  = J_cuda->seq_read(index_r) - jc;
    }	    
    else if ( ty == 0 && tx == BLOCK_SIZE-1 ){ //ne
        n  = north_ty_tx - jc;
        s  = J_cuda->seq_read(index_d) - jc;
        w  = J_cuda->seq_read(index_l) - jc; 
        e  = east_ty_tx - jc;
    }
    else if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
        n  = J_cuda->seq_read(index_u) - jc;
        s  = south_ty_tx - jc;
        w  = J_cuda->seq_read(index_l) - jc; 
        e  = east_ty_tx  - jc;
    }
    else if ( ty == BLOCK_SIZE -1 && tx == 0 ){//sw
        n  = J_cuda->seq_read(index_u) - jc;
        s  = south_ty_tx - jc;
        w  = west_ty_tx  - jc; 
        e  = J_cuda->seq_read(index_r) - jc;
    }

    else if ( ty == 0 ){ //n
        n  = north_ty_tx - jc;
        s  = J_cuda->seq_read(index_d) - jc;
        w  = J_cuda->seq_read(index_l) - jc; 
        e  = J_cuda->seq_read(index_r) - jc;
    }
    else if ( tx == BLOCK_SIZE -1 ){ //e
        n  = J_cuda->seq_read(index_u) - jc;
        s  = J_cuda->seq_read(index_d) - jc;
        w  = J_cuda->seq_read(index_l) - jc; 
        e  = east_ty_tx - jc;
    }
    else if ( ty == BLOCK_SIZE -1){ //s
        n  = J_cuda->seq_read(index_u) - jc;
        s  = south_ty_tx - jc;
        w  = J_cuda->seq_read(index_l) - jc; 
        e  = J_cuda->seq_read(index_r) - jc;
    }
    else if ( tx == 0 ){ //w
        n  = J_cuda->seq_read(index_u) - jc;
        s  = J_cuda->seq_read(index_d) - jc;
        w  = west_ty_tx - jc; 
        e  = J_cuda->seq_read(index_r) - jc;
    }
    else{  //the data elements which are not on the borders 
    
    }


    g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);

    l = ( n + s + w + e ) / jc;

    num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
    den  = 1 + (.25*l);
    qsqr = num/(den*den);

    // diffusion coefficent (equ 33)
    den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
    c = 1.0 / (1.0+den) ;

    // saturate diffusion coefficent
    if (c < 0){temp_result_ty_tx = 0;}
    else if (c > 1) {temp_result_ty_tx = 1;}
    else {temp_result_ty_tx = c;}

    __syncthreads();
    
    //printf("checkpoint 2\n");
    
    C_cuda->seq_write(index, temp_result_ty_tx);
    //Cptr[index] = temp_result_ty_tx;
    __syncthreads();
    E_C->seq_write(index, e);
    __syncthreads();
    W_C->seq_write(index, w);
    __syncthreads();
    S_C->seq_write(index, s);
    __syncthreads();
    N_C->seq_write(index, n);
    __syncthreads();
    
    //printf("checkpoint 3\n");
    //if (blockIdx.x % 128 == 0 && ty == 0) printf("srad_cuda_mod_1: [%ld]index w %ld, index e %ld\n", tid/32, index_w, index_e);

}

__global__ void
srad_cuda_mod_2(
		  array_d_t<TYPE> *E_C, 
		  array_d_t<TYPE> *W_C, 
		  array_d_t<TYPE> *N_C, 
		  array_d_t<TYPE> *S_C,	
		  array_d_t<TYPE> * J_cuda, 
		  array_d_t<TYPE> * C_cuda, 
		  size_t cols, 
		  size_t rows, 
		  TYPE lambda,
		  TYPE q0sqr,
          TYPE* C_ptr,
          size_t offset = 0
) 
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x + offset;
    size_t tidy = tid / cols;
    size_t tidx = tid % cols;
    
    //block id
    size_t bx = tidx / BLOCK_SIZE; // blockIdx.x;
    size_t by = tidy / BLOCK_SIZE; // blockIdx.y;

    //thread id
    size_t tx = tidx % BLOCK_SIZE; // threadIdx.x;
    size_t ty = tidy % BLOCK_SIZE; // threadIdx.y;
    
    size_t gdim_y = rows/BLOCK_SIZE;
    size_t gdim_x = cols/BLOCK_SIZE;
    
    //indices
    size_t index   = tid;                           // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    size_t index_s = tid + cols * (BLOCK_SIZE-ty);  // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    size_t index_e = tid + (BLOCK_SIZE-tx);         // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

    size_t index_r = tid + 1;                           // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx+1;
    size_t index_l = tid - 1;                           // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx-1;
    size_t index_d = tid + cols;                        // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * (ty+1) + tx;
    size_t index_u = tid - cols;                        // cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * (ty-1) + tx;

    TYPE cc, cn, cs, ce, cw, d_sum;

    //shared memory allocation
    TYPE south_c_ty_tx;
    TYPE east_c_ty_tx;

    TYPE c_cuda_temp_ty_tx;
    TYPE c_cuda_result;
    TYPE temp_ty_tx;
    
    if ( by == gridDim.y - 1 ) {
        index_s = tid + cols * (BLOCK_SIZE-1-ty);
    }
    if ( bx == gridDim.x - 1 ){
        index_e = tid - tx + BLOCK_SIZE-1;
    }

    //load data to shared memory
    temp_ty_tx = J_cuda->seq_read(index);
    __syncthreads();

    

    south_c_ty_tx = C_cuda->seq_read(index_s);
    __syncthreads();
    
    east_c_ty_tx = C_cuda->seq_read(index_e);

    __syncthreads();

    c_cuda_temp_ty_tx      = C_cuda->seq_read(index);
    //c_cuda_temp_ty_tx      = C_ptr[index];


    __syncthreads();


    cc = c_cuda_temp_ty_tx;

    cn  = cc;
    cs  = C_cuda->seq_read(index_d); //c_cuda_temp[ty+1][tx];
    //cs  = C_ptr[index_d]; //c_cuda_temp[ty+1][tx];
    __syncthreads();
    
    cw  = cc; 
    ce  = C_cuda->seq_read(index_r); //c_cuda_temp[ty][tx+1];
    //ce  = C_ptr[index_r]; //c_cuda_temp[ty][tx+1];
    __syncthreads();


    if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
        cn  = cc;
        cs  = south_c_ty_tx;
        cw  = cc; 
        ce  = east_c_ty_tx;
    } 
    else if ( tx == BLOCK_SIZE -1 ){ //e
        cn  = cc;
        cs  = C_cuda->seq_read(index_d); //c_cuda_temp[ty+1][tx];
        //cs = C_ptr[index_d];
        cw  = cc; 
        ce  = east_c_ty_tx;
    }
    else if ( ty == BLOCK_SIZE -1){ //s
        cn  = cc;
        cs  = south_c_ty_tx;
        cw  = cc; 
        ce  = C_cuda->seq_read(index_r); //c_cuda_temp[ty][tx+1];
        //ce = C_ptr[index_r];
    }
    else{ //the data elements which are not on the borders 
    
    }
    __syncthreads();

    // divergence (equ 58)
    d_sum = cn * N_C->seq_read(index);
    __syncthreads();
    d_sum += cs * S_C->seq_read(index);
    __syncthreads();
    d_sum += cw * W_C->seq_read(index);
    __syncthreads();
    d_sum += ce * E_C->seq_read(index);

    // image update (equ 61)
    c_cuda_result = temp_ty_tx + 0.25 * lambda * d_sum;
    __syncthreads();
    //printf("checkpoint 5\n");

    J_cuda->seq_write(index, c_cuda_result);
    __syncthreads();
    //if (blockIdx.x % 128 == 0 && ty == 0) printf("srad_cuda_mod_2: [%ld]index e %ld, index s %ld\n", tid/32, index_e, index_s);

}

