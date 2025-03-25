#include<stdio.h>
#include<cuda.h>

__global__ void printHello()
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello GPU by thread:%d",index);
}
int main()
{
    dim3 grid_dim = {1, 1, 1};
    dim3 block_dim = {4, 1, 1};
    printHello<<<grid_dim, block_dim>>>();
    cudaDeviceSynchronize();
    return 0;
}
