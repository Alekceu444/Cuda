#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#define N   10

__global__ void add(int *a, int *b, int *c)
{
	int tid = threadIdx.x;
	while (tid < N)
	{
		c[tid] = a[tid] + b[tid];
		//printf("%d + %d = %d\n", a[tid], b[tid], c[tid]);
		tid += 1;

	}
}

__global__ void Threadadd(int *a, int *b, int *c)
{
	int tid = threadIdx.x;
	c[tid] = a[tid] + b[tid];
	//printf("%d + %d = %d\n", a[tid], b[tid], c[tid]);
}

__global__ void Blockadd(int *a, int *b, int *c)
{
	int tid = blockIdx.x;
	c[tid] = a[tid] + b[tid];
	//printf("%d + %d = %d\n", a[tid], b[tid], c[tid]);
}

__host__ int main(void) 
{
	int a[N], b[N], c[N];

	for (int i = 0; i<N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}

	int* devA;
	int* devB;
	int* devC;

	cudaMalloc((void**)&devA, sizeof(int) * N);
	cudaMalloc((void**)&devB, sizeof(int) * N);
	cudaMalloc((void**)&devC, sizeof(int) * N);

	cudaMemcpy(devA, a, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, sizeof(int) * N, cudaMemcpyHostToDevice);

	add <<<1, 1 >>> (devA, devB, devC);

	cudaEvent_t syncEvent;

	cudaEventCreate(&syncEvent);    
	cudaEventRecord(syncEvent, 0);  
	cudaEventSynchronize(syncEvent);  

	cudaMemcpy(c, devC, sizeof(int) * N, cudaMemcpyDeviceToHost);

	for (int i = 0; i<N; i++) 
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	cudaEventDestroy(syncEvent);

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	/////////////////////////////////////////
	int a1[N], b1[N], c1[N];

	for (int i = 0; i<N; i++)
	{
		a1[i] = i;
		b1[i] = i * i;
	}

	int* devA1;
	int* devB1;
	int* devC1;

	cudaMalloc((void**)&devA1, sizeof(int) * N);
	cudaMalloc((void**)&devB1, sizeof(int) * N);
	cudaMalloc((void**)&devC1, sizeof(int) * N);

	cudaMemcpy(devA1, a1, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(devB1, b1, sizeof(int) * N, cudaMemcpyHostToDevice);

	Blockadd << <N, 1 >> > (devA1, devB1, devC1);

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);

	cudaMemcpy(c1, devC1, sizeof(int) * N, cudaMemcpyDeviceToHost);

	for (int i = 0; i<N; i++)
	{
		printf("%d + %d = %d\n", a1[i], b1[i], c1[i]);
	}

	cudaEventDestroy(syncEvent);

	cudaFree(devA1);
	cudaFree(devB1);
	cudaFree(devC1);

	/////////////////////////////////////////
	int a2[N], b2[N], c2[N];

	for (int i = 0; i<N; i++)
	{
		a2[i] = i*i;
		b2[i] = i * i*i;	
	}

	int* devA2;
	int* devB2;
	int* devC2;

	cudaMalloc((void**)&devA2, sizeof(int) * N);
	cudaMalloc((void**)&devB2, sizeof(int) * N);
	cudaMalloc((void**)&devC2, sizeof(int) * N);

	cudaMemcpy(devA2, a2, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(devB2, b2, sizeof(int) * N, cudaMemcpyHostToDevice);

	Threadadd << <1, N >> > (devA2, devB2, devC2);

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);

	cudaMemcpy(c2, devC2, sizeof(int) * N, cudaMemcpyDeviceToHost);


	cudaEventDestroy(syncEvent);

	cudaFree(devA2);
	cudaFree(devB2);
	cudaFree(devC2);

	for (int i = 0; i<N; i++)
	{
		printf("%d + %d = %d\n", a2[i], b2[i], c2[i]);
	}

	std::system("pause");

	return 0;
}