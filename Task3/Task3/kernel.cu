#include <stdio.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#define N   2
#define M   10

__global__ void summ(int *a, int *b, int *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = i * M + j;
	c[tid] = a[tid] + b[tid];

}

__host__ int main(void)
{
	int a[M*M], b[M*M], c[M*M];

	for (int i = 0; i<M*M; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}
	
	for (int i = 0; i<M; i++)
	{
		for (int j = 0; j < M; j++) {
			printf("%d ", a[i*M + j]);
		}
		printf("\n");
	}

	printf("-------------------------------- \n");

	for (int i = 0; i<M; i++)
	{
		for (int j = 0; j < M; j++) {
			printf("%d ", b[i*M + j]);
		}
		printf("\n");
	}

	printf("-------------------------------- \n");

	int* devA;
	int* devB;
	int* devC;

	cudaMalloc((void**)&devA, sizeof(int) * M*M);
	cudaMalloc((void**)&devB, sizeof(int) * M*M);
	cudaMalloc((void**)&devC, sizeof(int) * M*M);

	cudaMemcpy(devA, a, sizeof(int) * M*M, cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, sizeof(int) * M*M, cudaMemcpyHostToDevice);

	dim3 blocks(M / N, M / N);
	dim3 threads(N, N);

	summ << <blocks, threads>> > (devA, devB, devC);

	cudaEvent_t syncEvent;

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);

	cudaMemcpy(c, devC, sizeof(int) * M*M, cudaMemcpyDeviceToHost);

	for (int i = 0; i<M; i++)
	{
		for (int j = 0; j < M; j++) {
			printf("%d ", c[i*M+j]);
		}
		printf("\n");
	}

	cudaEventDestroy(syncEvent);

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);

	std::system("pause");

	return 0;
}