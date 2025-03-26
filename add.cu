#include<stdio.h>
#include<cuda.h>
#include<sys/time.h>

__global__ void addKernel(float *deviceA, float *deviceB, float *deviceC, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < n)
    {
        deviceC[index] = deviceA[index] + deviceB[index] + deviceC[index];
    }
}

int main()
{
    float *hostA, *hostB, *hostC;
    const int n = 102400;
    const int N = n * sizeof(float);
    hostA = (float *) malloc(N);
    hostB = (float *) malloc(N);
    hostC = (float *) malloc(N);
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc((void **)&deviceA, N);
    cudaMalloc((void **)&deviceB, N);
    cudaMalloc((void **)&deviceC, N);

    cudaMemcpy(deviceA, hostA, N, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, N, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float kernel_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int BLOCK_DIM = 1024;
    int num_block_x = n / BLOCK_DIM;
    int num_block_y = 1;

    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(BLOCK_DIM, 1, 1);

    addKernel<<<grid_dim, block_dim>>>(deviceA, deviceB, deviceC, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&kernel_time, start, stop);
    cudaMemcpy(hostC, deviceC, N, cudaMemcpyDeviceToHost);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    printf("kernel_time:%f \n", kernel_time);
    return 0;
}