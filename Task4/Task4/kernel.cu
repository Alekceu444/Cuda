#include <stdio.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#define N   4
#define M   8

__global__ void composition(int *a, int *b, int *c)
{
	int aEnd = M * N * blockIdx.y + M - 1;
	int sum = 0;

	for (int aBegin = M * N * blockIdx.y, bBegin = N * blockIdx.x; aBegin <= aEnd; aBegin += N, bBegin += N * M) {
		__shared__ int as[N*N];
		__shared__ int bs[N*N];
		as[N*threadIdx.y+threadIdx.x] = a[aBegin + M * threadIdx.y + threadIdx.x];
		bs[N*threadIdx.y+threadIdx.x] = b[bBegin + M * threadIdx.y + threadIdx.x];
		__syncthreads(); 
		for (int k = 0; k < N; k++)
			sum += as[threadIdx.y*N+k] * bs[k*N+threadIdx.x];
		__syncthreads(); 
	}

	c[M * N * blockIdx.y + N * blockIdx.x + M * threadIdx.y + threadIdx.x] = sum;
}

__host__ int main(void)
{
	int a[M*M], b[M*M], c[M*M];

	for (int i = 0; i < M*M; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < M; j++) {
			printf("%d ", a[i*M + j]);
		}
		printf("\n");
	}

	printf("-------------------------------- \n");

	for (int i = 0; i < M; i++)
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

	composition << <blocks, threads >> > (devA, devB, devC);

	cudaEvent_t syncEvent;

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);

	cudaMemcpy(c, devC, sizeof(int) * M*M, cudaMemcpyDeviceToHost);

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < M; j++) {
			printf("%d ", c[i*M + j]);
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