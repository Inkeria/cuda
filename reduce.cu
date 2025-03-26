#include<stdio.h>
#include<sys/time.h>
#include<cuda.h>

template<int BLOCK_DIM>
__global__ void reduce(float *A, int n)
{
    __shared__ float shareMem[BLOCK_DIM];
    float tmp = 0;
    if(threadIdx.x < BLOCK_DIM)
    {
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
        // if(threadIdx.x == 0) ans = shareMem[threadIdx.x];
    }
}

int main()
{
    float *A, *ans;
    const int n = 102400;
    A = (float*) malloc(n * sizeof(float));
    ans = (float *) malloc(sizeof(float));
    for(int i = 0;i < n; ++i){
        A[i] = (n - i + 1) * 1e-1; 
    }
    float *dA;
    cudaMalloc((void **)&dA, n * sizeof(float));
    cudaMemcpy(A, dA, n * sizeof(float), cudaMemcpyHostToDevice);

    // dim3 grid_dim(1, 1, 1);
    // dim3 block_dim(1024, 1, 1);
    // reduce<1024><<<grid_dim, block_dim>>>(dA, n);

    // cudaMemcpy(dA, ans, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n",dA[0]);
}