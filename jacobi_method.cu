#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define N         1024
#define MAX_ITERS 10000
#define TOL       1e-4f

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void generateMatrixAndSolution(float* A, float* x_true, int n)
{
    srand((unsigned int)time(NULL));

    for (int i = 0; i < n; ++i)
        x_true[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < n; ++i)
    {
        int aii_int = 1 + rand() % 10;
        float aii = (float)aii_int;
        A[i*n + i] = aii;

        for (int j = 0; j < n; ++j)
        {
            if (i == j) continue;
            int k = 1 + rand() % aii_int;
            A[i*n + j] = (float)k / (2.0f * aii * n);
        }
    }
}

float computeResidualCPU(const float* A, const float* x, const float* b, int n)
{
    float num = 0.0f, den = 0.0f;

    for (int i = 0; i < n; ++i)
    {
        float Ax_i = 0.0f;
        for (int j = 0; j < n; ++j)
            Ax_i += A[i*n + j] * x[j];

        float r = Ax_i - b[i];
        num += r*r;
        den += b[i]*b[i];
    }
    return (den == 0.0f) ? sqrtf(num) : sqrtf(num / den);
}

__global__ void matVecKernel(const float* A, const float* x, float* y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sum = 0;
    int row = i * n;
    for (int j = 0; j < n; ++j)
        sum += A[row + j] * x[j];

    y[i] = sum;
}

__global__ void jacobiKernel(const float* A, const float* b, const float* x_old, float* x_new, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sigma = 0;
    int row = i * n;

    for (int j = 0; j < n; ++j)
        if (j != i)
            sigma += A[row + j] * x_old[j];

    x_new[i] = (b[i] - sigma) / A[row + i];
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads dim: %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid size: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
 
    const int n = N;
    size_t matrixSize = n * n * sizeof(float);
    size_t vectorSize = n * sizeof(float);

    float *h_A = (float*)malloc(matrixSize);
    float *h_x_true = (float*)malloc(vectorSize);
    float *h_b = (float*)malloc(vectorSize);
    float *h_x = (float*)malloc(vectorSize);

    generateMatrixAndSolution(h_A, h_x_true, n);

    float *d_A, *d_x_true, *d_b, *d_x_old, *d_x_new;
    CUDA_CHECK(cudaMalloc(&d_A,      matrixSize));
    CUDA_CHECK(cudaMalloc(&d_x_true, vectorSize));
    CUDA_CHECK(cudaMalloc(&d_b,      vectorSize));
    CUDA_CHECK(cudaMalloc(&d_x_old, vectorSize));
    CUDA_CHECK(cudaMalloc(&d_x_new, vectorSize));

    CUDA_CHECK(cudaMemcpy(d_A,      h_A,      matrixSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_true, h_x_true, vectorSize, cudaMemcpyHostToDevice));

    int threadsPerBlock = 2048;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    matVecKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_x_true, d_b, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_b, d_b, vectorSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(d_x_old, 0, vectorSize));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    float residual = 0.0f;
    int it = 0;

    for (it = 0; it < MAX_ITERS; ++it)
    {
        jacobiKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_b, d_x_old, d_x_new, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        float* tmp = d_x_old;
        d_x_old = d_x_new;
        d_x_new = tmp;

        CUDA_CHECK(cudaMemcpy(h_x, d_x_old, vectorSize, cudaMemcpyDeviceToHost));
        residual = computeResidualCPU(h_A, h_x, h_b, n);

        if (residual < TOL) break;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    printf("Jacobi method finished\n");
    printf("N = %d\n", n);
    printf("Iterations = %d\n", it);
    printf("Residual = %e\n", residual);
    printf("Time = %f ms\n", time_ms);

    cudaFree(d_A);
    cudaFree(d_x_true);
    cudaFree(d_b);
    cudaFree(d_x_old);
    cudaFree(d_x_new);

    free(h_A);
    free(h_x_true);
    free(h_b);
    free(h_x);

    return 0;
}

