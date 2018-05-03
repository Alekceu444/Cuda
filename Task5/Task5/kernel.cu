#include <stdio.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <ctime>
#include <cstdlib>

#define NUM_BINS	256
#define N			9192
#define NUM_THREADS 512


__global__ void histogram(int * histogramm, int * arrays)
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int num = arrays[tid];
	histogramm[num] += 1;
}


int main(void) {

	srand(time(NULL));

	int a[N], b[NUM_BINS];


	for (int i = 0; i < N; i++) {

		a[i] = rand()%256;
	
	}

	for (int i = 0; i < NUM_BINS; i++) {

		b[i] = 0;

	}

	int* devA;
	int* devB;

	cudaMalloc((void**)&devA, sizeof(int) * N);
	cudaMalloc((void**)&devB, sizeof(int) * NUM_BINS);

	cudaMemcpy(devA, a, sizeof(int) * N, cudaMemcpyHostToDevice);
	

	histogram <<< ( N / NUM_THREADS), NUM_THREADS >>>(devB, devA);


	cudaEvent_t syncEvent;

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);

	cudaMemcpy(b, devB, sizeof(int) * NUM_BINS, cudaMemcpyDeviceToHost);


	for (int i = 0; i < NUM_BINS; i++)
		printf("%d :: %d\n", i, b[i]);

	cudaEventDestroy(syncEvent);

	cudaFree(devA);
	cudaFree(devB);

	std::system("pause");

	return 0;

}