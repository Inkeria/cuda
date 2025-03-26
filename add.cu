#include<stdio.h>
#include<cuda.h>
#include<sys/time.h>

double get_time()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

__global__ void addKernel(float *A, float *B, float *C, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < n)
    {
        C[index] = A[index] + B[index];
    }
}
void initCpu(float *hostA, float *hostB, int n)
{
    for (int i = 0; i < n; i++)
    {
        hostA[i] = 1;
        hostB[i] = 1;
    }
}
int main()
{
    float *hostA, *hostB, *hostC, *ddvvC;
    int n = 102400;
    // const int N = n * sizeof(float);
    hostA = (float *) malloc(n * sizeof(float));
    hostB = (float *) malloc(n * sizeof(float));
    hostC = (float *) malloc(n * sizeof(float));
    ddvvC = (float *) malloc(n * sizeof(float));
    initCpu(hostA, hostB, n);
    // memset(hostA, 0x3f, N);
    // memset(hostB, 0x3f, N);
    double st = get_time();
    for(int i = 0;i < n;++i){
        hostC[i] = hostA[i] + hostB[i] + hostC[i];
    }
    double cpu_time = get_time() - st;
    st = get_time();
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc((void **)&deviceA, n * sizeof(float));
    cudaMalloc((void **)&deviceB, n * sizeof(float));
    cudaMalloc((void **)&deviceC, n * sizeof(float));

    cudaMemcpy(deviceA, hostA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, n * sizeof(float), cudaMemcpyHostToDevice);

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
    cudaMemcpy(ddvvC, deviceC, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    double gpu_time = get_time() - st;
    printf("kernel_time:%.6f \n", kernel_time);
    printf("cpu_time:%.6f \n gpu_time:%.6f\n",cpu_time ,gpu_time);
    
    free(hostA);
    free(hostB);
    free(hostC);
    free(ddvvC);
    return 0;
}