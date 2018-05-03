#include <stdio.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#define N   10
#define M   5

__global__ void add(int *a, int *b, int *c)
{
	int tid = blockIdx.x * blockDim.x+threadIdx.x;
	c[tid] = a[tid] + b[tid];
}

__host__ int main(void)
{
	int a[N*M], b[N*M], c[N*M];

	for (int i = 0; i<N*M; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}

	int* devA;
	int* devB;
	int* devC;

	cudaMalloc((void**)&devA, sizeof(int) * N*M);
	cudaMalloc((void**)&devB, sizeof(int) * N*M);
	cudaMalloc((void**)&devC, sizeof(int) * N*M);

	cudaMemcpy(devA, a, sizeof(int) * N*M, cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, sizeof(int) * N*M, cudaMemcpyHostToDevice);

	//printf("%d + %d \n", (N + (M - 1) / M),M);
	add <<<(N+(M-1)/M),M>>> (devA, devB, devC);
	//add <<<(N, M >>>(devA, devB, devC);

	cudaEvent_t syncEvent;

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);

	cudaMemcpy(c, devC, sizeof(int) * N*M, cudaMemcpyDeviceToHost);

	for (int i = 0; i<N*M; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	cudaEventDestroy(syncEvent);

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	
	std::system("pause");

	return 0;
}