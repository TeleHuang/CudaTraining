/*
 * md4_simulation.cu
 * 
 * MD4 Molecular Dynamics Simulation Module - Implementation
 * Cell List optimized molecular dynamics simulation
 */

#include "md4_simulation.cuh"
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace std;

// Error checking macro
#define CUDA_CHECK_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUDA_CHECK_ERROR_BOOL(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

// ============== CUDA Kernels ==============

// Device helper: get cell index from position
__device__ int getCellIndex(double x, double y, double z, 
                            double cellSize, int numCellsX, int numCellsY, int numCellsZ, double L) {
    if (x < 0) x += L;
    if (x >= L) x -= L;
    if (y < 0) y += L;
    if (y >= L) y -= L;
    if (z < 0) z += L;
    if (z >= L) z -= L;
    
    int cx = min((int)(x / cellSize), numCellsX - 1);
    int cy = min((int)(y / cellSize), numCellsY - 1);
    int cz = min((int)(z / cellSize), numCellsZ - 1);
    
    cx = max(0, cx);
    cy = max(0, cy);
    cz = max(0, cz);
    
    return cx + cy * numCellsX + cz * numCellsX * numCellsY;
}

// Kernel: Build cell list
__global__ void buildCellListKernel(double* d_r, int* d_cellCount, int* d_cellList,
                                    int* d_cellOverflow,
                                    double cellSize, int numCellsX, int numCellsY, int numCellsZ,
                                    int maxParticlesPerCell, double L, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    double x = d_r[i * 3];
    double y = d_r[i * 3 + 1];
    double z = d_r[i * 3 + 2];
    
    int cellIndex = getCellIndex(x, y, z, cellSize, numCellsX, numCellsY, numCellsZ, L);
    int offset = atomicAdd(&d_cellCount[cellIndex], 1);
    
    if (offset < maxParticlesPerCell) {
        d_cellList[cellIndex * maxParticlesPerCell + offset] = i;
    } else {
        atomicAdd(d_cellOverflow, 1);
    }
}

// Kernel: Build neighbor list from cell list
__global__ void buildNeighborListKernel(double* d_r, int* d_cellCount, int* d_cellList,
                                        int* d_pairList, double* d_drPair, double* d_rSqdPair,
                                        int* d_nPairs, int nPairsMax,
                                        double cellSize, int numCellsX, int numCellsY, int numCellsZ,
                                        int maxParticlesPerCell, double rMax, double L, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    double xi = d_r[i * 3];
    double yi = d_r[i * 3 + 1];
    double zi = d_r[i * 3 + 2];
    
    int cxi = min((int)(xi / cellSize), numCellsX - 1);
    int cyi = min((int)(yi / cellSize), numCellsY - 1);
    int czi = min((int)(zi / cellSize), numCellsZ - 1);
    cxi = max(0, cxi);
    cyi = max(0, cyi);
    czi = max(0, czi);
    
    double rMaxSqd = rMax * rMax;
    
    for (int dcx = -1; dcx <= 1; dcx++) {
        for (int dcy = -1; dcy <= 1; dcy++) {
            for (int dcz = -1; dcz <= 1; dcz++) {
                int ncx = cxi + dcx;
                int ncy = cyi + dcy;
                int ncz = czi + dcz;
                
                if (ncx < 0) ncx += numCellsX;
                if (ncx >= numCellsX) ncx -= numCellsX;
                if (ncy < 0) ncy += numCellsY;
                if (ncy >= numCellsY) ncy -= numCellsY;
                if (ncz < 0) ncz += numCellsZ;
                if (ncz >= numCellsZ) ncz -= numCellsZ;
                
                int neighborCellIndex = ncx + ncy * numCellsX + ncz * numCellsX * numCellsY;
                int cellParticleCount = d_cellCount[neighborCellIndex];
                
                for (int k = 0; k < cellParticleCount && k < maxParticlesPerCell; k++) {
                    int j = d_cellList[neighborCellIndex * maxParticlesPerCell + k];
                    
                    if (j <= i) continue;
                    
                    double xj = d_r[j * 3];
                    double yj = d_r[j * 3 + 1];
                    double zj = d_r[j * 3 + 2];
                    
                    double dx = xi - xj;
                    double dy = yi - yj;
                    double dz = zi - zj;
                    
                    if (dx >= 0.5 * L) dx -= L;
                    if (dx < -0.5 * L) dx += L;
                    if (dy >= 0.5 * L) dy -= L;
                    if (dy < -0.5 * L) dy += L;
                    if (dz >= 0.5 * L) dz -= L;
                    if (dz < -0.5 * L) dz += L;
                    
                    double rSqd = dx * dx + dy * dy + dz * dz;
                    
                    if (rSqd < rMaxSqd) {
                        int pairIdx = atomicAdd(d_nPairs, 1);
                        
                        if (pairIdx < nPairsMax) {
                            d_pairList[pairIdx * 2] = i;
                            d_pairList[pairIdx * 2 + 1] = j;
                            d_drPair[pairIdx * 3] = dx;
                            d_drPair[pairIdx * 3 + 1] = dy;
                            d_drPair[pairIdx * 3 + 2] = dz;
                            d_rSqdPair[pairIdx] = rSqd;
                        }
                    }
                }
            }
        }
    }
}

// Kernel: Update pair separations
__global__ void computeSeparationKernel(double* d_r, int* d_pairList,
                                        double* d_drPair, double* d_rSqdPair,
                                        int* d_nPairs, double L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_nPairs[0]) return;

    int p1 = d_pairList[2 * i];
    int p2 = d_pairList[2 * i + 1];

    d_rSqdPair[i] = 0;
    for (int d = 0; d < 3; d++) {
        d_drPair[i * 3 + d] = d_r[p1 * 3 + d] - d_r[p2 * 3 + d];

        if (d_drPair[i * 3 + d] >= 0.5 * L)
            d_drPair[i * 3 + d] -= L;
        if (d_drPair[i * 3 + d] < -0.5 * L)
            d_drPair[i * 3 + d] += L;

        d_rSqdPair[i] += d_drPair[i * 3 + d] * d_drPair[i * 3 + d];
    }
}

// Kernel: Zero accelerations
__global__ void zeroAccelerationsKernel(double* d_a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_a[idx * 3] = 0.0;
        d_a[idx * 3 + 1] = 0.0;
        d_a[idx * 3 + 2] = 0.0;
    }
}

// Kernel: Compute accelerations
__global__ void computeAccelerationsKernel(double* d_r, double* d_a, int* d_pairList, 
                                          double* d_drPair, double* d_rSqdPair, 
                                          int nPairs, double rCutOff, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nPairs) return;
    
    int i = d_pairList[idx * 2];
    int j = d_pairList[idx * 2 + 1];
    double rSqd = d_rSqdPair[idx];

    double epsilon = 1e-9; 
    if (rSqd < epsilon) return;

    if (rSqd < rCutOff * rCutOff) {
        double r2Inv = 1.0 / rSqd;
        double r6Inv = r2Inv * r2Inv * r2Inv;
        double f = 24.0 * r2Inv * r6Inv * (2.0 * r6Inv - 1.0);

        for (int d = 0; d < 3; d++) {
            double force = f * d_drPair[idx * 3 + d];
            
            unsigned long long int* address_as_ull = (unsigned long long int*)&d_a[i * 3 + d];
            unsigned long long int old = *address_as_ull;
            unsigned long long int assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(force + __longlong_as_double(assumed)));
            } while (assumed != old);
            
            address_as_ull = (unsigned long long int*)&d_a[j * 3 + d];
            old = *address_as_ull;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(-force + __longlong_as_double(assumed)));
            } while (assumed != old);
        }
    }
}

// Kernel: Verlet step 1
__global__ void velocityVerletStep1Kernel(double* d_r, double* d_v, double* d_a, 
                                         double dt, double L, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    for (int d = 0; d < 3; d++) {
        d_r[idx * 3 + d] += d_v[idx * 3 + d] * dt + 0.5 * d_a[idx * 3 + d] * dt * dt;
        
        if (d_r[idx * 3 + d] < 0)
            d_r[idx * 3 + d] += L;
        if (d_r[idx * 3 + d] >= L)
            d_r[idx * 3 + d] -= L;
        
        d_v[idx * 3 + d] += 0.5 * d_a[idx * 3 + d] * dt;
    }
}

// Kernel: Verlet step 2
__global__ void velocityVerletStep2Kernel(double* d_v, double* d_a, double dt, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    for (int d = 0; d < 3; d++) {
        d_v[idx * 3 + d] += 0.5 * d_a[idx * 3 + d] * dt;
    }
}

// ============== MD4Simulation Class Implementation ==============

MD4Simulation::MD4Simulation(int numParticles, double density, double temperature) {
    params.N = numParticles;
    params.rho = density;
    params.T = temperature;
    params.L = 0.0;  // Computed in initPositions
    params.dt = 0.01;
    params.rCutOff = 2.5;
    params.rMax = 3.3;
    params.updateInterval = 10;
    
    currentStep = 0;
    nPairs = 0;
    nPairsMax = 0;
    totalCells = 0;
    
    // Initialize pointers
    h_r = h_v = h_a = nullptr;
    d_r = d_v = d_a = nullptr;
    d_nPairs = nullptr;
    d_pairList = nullptr;
    d_drPair = nullptr;
    d_rSqdPair = nullptr;
    d_cellCount = nullptr;
    d_cellList = nullptr;
    d_cellOverflow = nullptr;
}

MD4Simulation::~MD4Simulation() {
    freeMemory();
}

bool MD4Simulation::initialize() {
    // Check CUDA device
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        cerr << "No CUDA devices found!" << endl;
        return false;
    }
    
    CUDA_CHECK_ERROR_BOOL(cudaSetDevice(0));
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Using GPU: " << prop.name << endl;
    
    allocateMemory();
    initPositions();
    initVelocities();
    copyDataToDevice();
    
    // Initial neighbor list and acceleration
    updatePairList();
    
    if (nPairs > 0) {
        updatePairSeparations();
        
        int blockSize = 256;
        int numBlocks = (params.N + blockSize - 1) / blockSize;
        zeroAccelerationsKernel<<<numBlocks, blockSize>>>(d_a, params.N);
        cudaDeviceSynchronize();
        
        int numBlocksPairs = (nPairs + blockSize - 1) / blockSize;
        computeAccelerationsKernel<<<numBlocksPairs, blockSize>>>(
            d_r, d_a, d_pairList, d_drPair, d_rSqdPair, nPairs, params.rCutOff, params.N);
        cudaDeviceSynchronize();
    }
    
    cout << "Simulation initialized: N=" << params.N << ", L=" << params.L 
         << ", nPairs=" << nPairs << endl;
    
    return true;
}

void MD4Simulation::allocateMemory() {
    int N = params.N;
    
    // Host memory
    h_r = new double*[N];
    h_v = new double*[N];
    h_a = new double*[N];
    for (int i = 0; i < N; i++) {
        h_r[i] = new double[3];
        h_v[i] = new double[3];
        h_a[i] = new double[3];
        for (int d = 0; d < 3; d++) {
            h_a[i][d] = 0.0;
        }
    }

    // Device memory
    cudaMalloc((void**)&d_r, N * 3 * sizeof(double));
    cudaMalloc((void**)&d_v, N * 3 * sizeof(double));
    cudaMalloc((void**)&d_a, N * 3 * sizeof(double));

    // Neighbor list memory
    double avgNeighbors = (4.0 / 3.0) * 3.14159 * params.rMax * params.rMax * params.rMax * params.rho;
    nPairsMax = (int)(N * avgNeighbors / 2.0 * 1.5);
    if (nPairsMax < N * 50) nPairsMax = N * 50;
    
    cudaMalloc((void**)&d_pairList, nPairsMax * 2 * sizeof(int));
    cudaMalloc((void**)&d_drPair, nPairsMax * 3 * sizeof(double));
    cudaMalloc((void**)&d_rSqdPair, nPairsMax * sizeof(double));
    cudaMalloc((void**)&d_nPairs, sizeof(int));

    // Cell list (initial estimate)
    int estimatedCells = (int)(pow(N / params.rho, 1.0) / (params.rMax * params.rMax * params.rMax));
    if (estimatedCells < 1) estimatedCells = 1;
    if (estimatedCells < 27) estimatedCells = 27;
    
    cudaMalloc((void**)&d_cellCount, estimatedCells * sizeof(int));
    cudaMalloc((void**)&d_cellList, estimatedCells * MAX_PARTICLES_PER_CELL * sizeof(int));
    cudaMalloc((void**)&d_cellOverflow, sizeof(int));
}

void MD4Simulation::freeMemory() {
    if (h_r) {
        for (int i = 0; i < params.N; i++) {
            delete[] h_r[i];
            delete[] h_v[i];
            delete[] h_a[i];
        }
        delete[] h_r;
        delete[] h_v;
        delete[] h_a;
        h_r = h_v = h_a = nullptr;
    }

    if (d_r) cudaFree(d_r);
    if (d_v) cudaFree(d_v);
    if (d_a) cudaFree(d_a);
    if (d_pairList) cudaFree(d_pairList);
    if (d_drPair) cudaFree(d_drPair);
    if (d_rSqdPair) cudaFree(d_rSqdPair);
    if (d_nPairs) cudaFree(d_nPairs);
    if (d_cellCount) cudaFree(d_cellCount);
    if (d_cellList) cudaFree(d_cellList);
    if (d_cellOverflow) cudaFree(d_cellOverflow);
}

void MD4Simulation::initPositions() {
    int N = params.N;
    params.L = pow(N / params.rho, 1.0 / 3);

    int M = 1;
    while (4 * M * M * M < N)
        ++M;
    double a = params.L / M;

    double xCell[4] = { 0.25, 0.75, 0.75, 0.25 };
    double yCell[4] = { 0.25, 0.75, 0.25, 0.75 };
    double zCell[4] = { 0.25, 0.25, 0.75, 0.75 };

    int n = 0;
    for (int x = 0; x < M; x++)
        for (int y = 0; y < M; y++)
            for (int z = 0; z < M; z++)
                for (int k = 0; k < 4; k++)
                    if (n < N) {
                        h_r[n][0] = (x + xCell[k]) * a;
                        h_r[n][1] = (y + yCell[k]) * a;
                        h_r[n][2] = (z + zCell[k]) * a;
                        ++n;
                    }
    
    computeCellParameters();
    reallocateCellMemory();
}

void MD4Simulation::computeCellParameters() {
    cellSize = params.rMax;
    
    numCellsX = (int)floor(params.L / cellSize);
    numCellsY = (int)floor(params.L / cellSize);
    numCellsZ = (int)floor(params.L / cellSize);
    if (numCellsX < 1) numCellsX = 1;
    if (numCellsY < 1) numCellsY = 1;
    if (numCellsZ < 1) numCellsZ = 1;
    
    cellSize = params.L / numCellsX;
    totalCells = numCellsX * numCellsY * numCellsZ;
}

void MD4Simulation::reallocateCellMemory() {
    if (d_cellCount) cudaFree(d_cellCount);
    if (d_cellList) cudaFree(d_cellList);
    
    cudaMalloc((void**)&d_cellCount, totalCells * sizeof(int));
    cudaMalloc((void**)&d_cellList, totalCells * MAX_PARTICLES_PER_CELL * sizeof(int));
}

double MD4Simulation::gasdev() {
    static bool available = false;
    static double gset;
    double fac, rsq, v1, v2;
    if (!available) {
        do {
            v1 = 2.0 * rand() / double(RAND_MAX) - 1.0;
            v2 = 2.0 * rand() / double(RAND_MAX) - 1.0;
            rsq = v1 * v1 + v2 * v2;
        } while (rsq >= 1.0 || rsq == 0.0);
        fac = sqrt(-2.0 * log(rsq) / rsq);
        gset = v1 * fac;
        available = true;
        return v2 * fac;
    } else {
        available = false;
        return gset;
    }
}

void MD4Simulation::initVelocities() {
    int N = params.N;
    
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            h_v[n][i] = gasdev();
    
    double vCM[3] = { 0, 0, 0 };
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            vCM[i] += h_v[n][i];
    for (int i = 0; i < 3; i++)
        vCM[i] /= N;
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            h_v[n][i] -= vCM[i];

    double vSqdSum = 0;
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            vSqdSum += h_v[n][i] * h_v[n][i];
    double lambda = sqrt(3 * (N - 1) * params.T / vSqdSum);
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            h_v[n][i] *= lambda;
}

void MD4Simulation::copyDataToDevice() {
    int N = params.N;
    double* h_r_flat = new double[N * 3];
    double* h_v_flat = new double[N * 3];
    double* h_a_flat = new double[N * 3];

    for (int i = 0; i < N; i++) {
        for (int d = 0; d < 3; d++) {
            h_r_flat[i * 3 + d] = h_r[i][d];
            h_v_flat[i * 3 + d] = h_v[i][d];
            h_a_flat[i * 3 + d] = h_a[i][d];
        }
    }

    cudaMemcpy(d_r, h_r_flat, N * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v_flat, N * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a_flat, N * 3 * sizeof(double), cudaMemcpyHostToDevice);

    delete[] h_r_flat;
    delete[] h_v_flat;
    delete[] h_a_flat;
}

void MD4Simulation::copyDataFromDevice() {
    int N = params.N;
    double* h_r_flat = new double[N * 3];
    double* h_v_flat = new double[N * 3];
    double* h_a_flat = new double[N * 3];

    cudaMemcpy(h_r_flat, d_r, N * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v_flat, d_v, N * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a_flat, d_a, N * 3 * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        for (int d = 0; d < 3; d++) {
            h_r[i][d] = h_r_flat[i * 3 + d];
            h_v[i][d] = h_v_flat[i * 3 + d];
            h_a[i][d] = h_a_flat[i * 3 + d];
        }
    }

    delete[] h_r_flat;
    delete[] h_v_flat;
    delete[] h_a_flat;
}

void MD4Simulation::copyVelocityToDevice() {
    int N = params.N;
    double* h_v_flat = new double[N * 3];
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < 3; d++) {
            h_v_flat[i * 3 + d] = h_v[i][d];
        }
    }
    cudaMemcpy(d_v, h_v_flat, N * 3 * sizeof(double), cudaMemcpyHostToDevice);
    delete[] h_v_flat;
}

void MD4Simulation::updatePairList() {
    int blockSize = 256;
    int numBlocks = (params.N + blockSize - 1) / blockSize;
    
    cudaMemset(d_cellCount, 0, totalCells * sizeof(int));
    cudaMemset(d_nPairs, 0, sizeof(int));
    cudaMemset(d_cellOverflow, 0, sizeof(int));
    
    buildCellListKernel<<<numBlocks, blockSize>>>(
        d_r, d_cellCount, d_cellList, d_cellOverflow,
        cellSize, numCellsX, numCellsY, numCellsZ,
        MAX_PARTICLES_PER_CELL, params.L, params.N);
    cudaDeviceSynchronize();
    
    buildNeighborListKernel<<<numBlocks, blockSize>>>(
        d_r, d_cellCount, d_cellList,
        d_pairList, d_drPair, d_rSqdPair,
        d_nPairs, nPairsMax,
        cellSize, numCellsX, numCellsY, numCellsZ,
        MAX_PARTICLES_PER_CELL, params.rMax, params.L, params.N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&nPairs, d_nPairs, sizeof(int), cudaMemcpyDeviceToHost);
}

void MD4Simulation::updatePairSeparations() {
    cudaMemcpy(&nPairs, d_nPairs, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (nPairs > 0) {
        int blockSize = 256;
        int numBlocks = (nPairs + blockSize - 1) / blockSize;
        computeSeparationKernel<<<numBlocks, blockSize>>>(
            d_r, d_pairList, d_drPair, d_rSqdPair, d_nPairs, params.L);
        cudaDeviceSynchronize();
    }
}

void MD4Simulation::velocityVerlet() {
    int blockSize = 256;
    int numBlocks = (params.N + blockSize - 1) / blockSize;
    int numBlocksPairs = (nPairs + blockSize - 1) / blockSize;

    velocityVerletStep1Kernel<<<numBlocks, blockSize>>>(d_r, d_v, d_a, params.dt, params.L, params.N);
    cudaDeviceSynchronize();

    updatePairSeparations();

    zeroAccelerationsKernel<<<numBlocks, blockSize>>>(d_a, params.N);
    cudaDeviceSynchronize();

    if (nPairs > 0) {
        computeAccelerationsKernel<<<numBlocksPairs, blockSize>>>(
            d_r, d_a, d_pairList, d_drPair, d_rSqdPair, nPairs, params.rCutOff, params.N);
        cudaDeviceSynchronize();
    }

    velocityVerletStep2Kernel<<<numBlocks, blockSize>>>(d_v, d_a, params.dt, params.N);
    cudaDeviceSynchronize();
}

void MD4Simulation::step() {
    velocityVerlet();
    
    // Rescale velocities every 200 steps
    if (currentStep % 200 == 0) {
        rescaleVelocities();
    }
    
    // Update neighbor list periodically
    if (currentStep % params.updateInterval == 0) {
        updatePairList();
        updatePairSeparations();
    }
    
    currentStep++;
}

void MD4Simulation::rescaleVelocities() {
    copyDataFromDevice();
    
    int N = params.N;
    double vSqdSum = 0;
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            vSqdSum += h_v[n][i] * h_v[n][i];
    double lambda = sqrt(3 * (N - 1) * params.T / vSqdSum);
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            h_v[n][i] *= lambda;
    
    copyVelocityToDevice();
}

double MD4Simulation::getTemperature() {
    copyDataFromDevice();
    
    double sum = 0;
    for (int i = 0; i < params.N; i++)
        for (int k = 0; k < 3; k++)
            sum += h_v[i][k] * h_v[i][k];
    return sum / (3 * (params.N - 1));
}

void MD4Simulation::getPositions(float* positions) {
    // Copy positions from device to host, then convert to float
    double* h_r_flat = new double[params.N * 3];
    cudaMemcpy(h_r_flat, d_r, params.N * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < params.N * 3; i++) {
        positions[i] = (float)h_r_flat[i];
    }
    
    delete[] h_r_flat;
}
