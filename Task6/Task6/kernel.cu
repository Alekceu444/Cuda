
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <ctime>
#include <iostream>

using namespace std;

#define RAD   0.5f
#define POINTS 10000
#define THREADS 200


__global__ void IsInsideCircle(float *a, float *b)
{
	int i = blockIdx.x * blockDim.x*2 + threadIdx.x*2;
	int tid = i/2;
	//printf("%d::::%d:::%d:::::%d::::%d\n", threadIdx.x, blockIdx.x, i, i+1,tid);
	if (sqrt(pow(RAD-a[i], 2) + pow(RAD- a[i+1], 2))<=RAD) {
		b[tid] = 1;
	}
	else {
		b[tid] = 0;
	}
}


int main(void)
{
	srand(time(NULL));
	
	float a[POINTS * 2];
	float b[POINTS];

	for (int i = 0; i<POINTS*2; i++)
	{
		a[i] = rand() / float(RAND_MAX);
		if (i < POINTS)
			b[i] = 0;
	}

	float in_circle = 0;

	setlocale(LC_ALL, "Russian");

	float* devA;
	float* devB;
	
	cudaMalloc((void**)&devA, sizeof(float) * 2*POINTS);
	cudaMalloc((void**)&devB, sizeof(float) * POINTS);

	cudaMemcpy(devA, a, sizeof(float) * POINTS*2, cudaMemcpyHostToDevice);

	dim3 blocks(POINTS / THREADS);
	dim3 threads(THREADS);

	IsInsideCircle<<< blocks, threads >>>(devA,devB);
	
	cudaEvent_t syncEvent;

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);

	cudaMemcpy(b, devB, sizeof(int) * POINTS, cudaMemcpyDeviceToHost);

	for (int i = 0; i<POINTS; i++)
	{
		if (b[i] == 1)
			in_circle += 1;
	}

	cout << "„исло пи = ";
	cout << 4 * (in_circle / POINTS) << endl;
	system("pause");
	return 0;
}