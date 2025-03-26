#include<stdio.h>
#include<sys/time.h>
#include<cuda.h>

template<int BLOCK_DIM>
__global__ void reduce(float *A, int n)
{
    __shared__ float shareMem[BLOCK_DIM];
    float tmp = 0;
    // printf("now run on GPU tread:%d\n",threadIdx.x);
    // __syncthreads();
    // if(threadIdx.x < BLOCK_DIM)
    // {
        // printf("now run on GPU tread:%d\n",threadIdx.x);
        // __syncthreads();
        for(int id = threadIdx.x; id < n;id += BLOCK_DIM)
        {
            tmp = tmp + A[id];
        }
        shareMem[threadIdx.x] = tmp;
        __syncthreads();
        for(int rad = BLOCK_DIM >> 1; rad; rad >>= 1)
        {
            if(threadIdx.x < rad) {
                shareMem[threadIdx.x] = shareMem[threadIdx.x] + shareMem[threadIdx.x + rad]; 
            }
            __syncthreads();
        }
        if(blockIdx.x == 0)
        A[threadIdx.x] = shareMem[threadIdx.x];
    // }
    // else {
        // printf("now not run on GPU tread:%d\n",threadIdx.x);
        // __syncthreads();
    // }
}

template<int BLOCK_DIM>
__global__ void shfl_reduce(float *A, int n)
{
    __shared__ float shareMem[BLOCK_DIM];
    float tmp = 0;
    for(int id = threadIdx.x; id < n;id += BLOCK_DIM)
    {
        tmp = tmp + A[id];
    }
    shareMem[threadIdx.x] = tmp;
    __syncthreads();
    __shared__ float val[32];
    tmp = 0;
    tmp += __shfl_down_sync(0xffffffff, tmp, 16);
    tmp += __shfl_down_sync(0xffffffff, tmp, 8);
    tmp += __shfl_down_sync(0xffffffff, tmp, 4);
    tmp += __shfl_down_sync(0xffffffff, tmp, 2);
    tmp += __shfl_down_sync(0xffffffff, tmp, 1);
    if(threadIdx.x >> 5 & 1){
        val[threadIdx.x >> 5] = tmp;
    }
    __syncthreads();
    if(threadIdx.x < 32)
    {
        tmp = val[threadIdx.x];
        tmp += __shfl_down_sync(0xffffffff, tmp, 16);
        tmp += __shfl_down_sync(0xffffffff, tmp, 8);
        tmp += __shfl_down_sync(0xffffffff, tmp, 4);
        tmp += __shfl_down_sync(0xffffffff, tmp, 2);
        tmp += __shfl_down_sync(0xffffffff, tmp, 1);
    }
    __syncthreads();
    if(threadIdx.x == 0){
        A[0] = tmp;
    }
}

int main()
{
    float *A;
    const int n = 102400;
    A = (float*) malloc(n * sizeof(float));
    // ans = (float *) malloc(sizeof(float));
    for(int i = 0;i < n; ++i){
        A[i] = (n - i + 1) * 1e-2;
    }
    float *dA, *ans;
    ans = (float *) malloc(sizeof(float));
    cudaMalloc((void **)&dA, n * sizeof(float));
    // printf("%f\n",A[0]);
    cudaMemcpy(dA, A, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_dim(100, 1, 1);
    dim3 block_dim(1024, 1, 1);
    shfl_reduce<1024><<<grid_dim, block_dim>>>(dA, n);
    cudaDeviceSynchronize();

    cudaMemcpy(ans, dA, sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 1;i < n;++i) A[0] += A[i];
    printf("%f\n",*ans);
    printf("%f\n",*A);
}