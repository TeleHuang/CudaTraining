/*
   ConsoleApplicationMD4_CUDA.cu

   使用Verlet和动态近邻表方法计算液态氩的分子动力学 - CUDA加速版本
   
   相比MD3的改进：使用空间网格划分（Cell List）方法将近邻表更新从O(N²)降低到O(N)
   
   空间网格划分原理：
   - 将仿真盒子划分为多个立方体格子（cell），每个cell边长 >= rMax
   - 每个粒子只需要检查自身所在cell及周围26个邻居cell中的粒子
   - 使用固定容量数组存储每个cell中的粒子索引

   主循环当中包含每个物理帧进行一次的Verlet和每10个物理帧进行一次近邻表更新
   Verlet的时间复杂度是O(N)，近邻表更新的时间复杂度现在也是O(N)

   如需执行，请在终端输入如下两行命令：
   nvcc -arch=sm_61 ConsoleApplicationMD4_CUDA.cu -o MD4_CUDA
   ./MD4_CUDA.exe
   其中-arch=sm_61是GPU架构，这里对应的是1050显卡，可自行修改
*/

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// 仿真参数
int N = 864;              // 氩原子数（可以尝试更大的值，如20000）
double rho = 0.8;         // 密度
double T = 1.0;           // 温度
double L;                 // 仿真空间的边长（通过N和RHO计算）

// 主机端粒子数据
double** h_r, ** h_v, ** h_a;     // 位置，速度，加速度（主机端）

// 设备端粒子数据
double *d_r, *d_v, *d_a;          // 位置，速度，加速度（设备端）

// 用于实现动态近邻表的各种参数
double rCutOff = 2.5;     // 力计算的截断距离
double rMax = 3.3;        // 近邻表内粒子对最大距离
int nPairs;               // 当前近邻表粒子对数量
double* h_rSqdPair;       // 每个对的距离(i,j)（主机端）- 用于调试

// 设备端近邻表数据
int* d_nPairs;            // 当前近邻表粒子对数量（设备端）
int* d_pairList;          // 近邻表（设备端）
double* d_drPair;         // 每个对的朝向 (i,j)（设备端）
double* d_rSqdPair;       // 每个对的距离(i,j)（设备端）

// ============== Cell List 相关参数 ==============
double cellSize;          // 每个cell的边长，应 >= rMax
int numCellsX, numCellsY, numCellsZ;  // 各方向的cell数量
int totalCells;           // 总cell数量
const int MAX_PARTICLES_PER_CELL = 64;  // 每个cell最大粒子容量（可根据密度调整）

// Cell List 设备端数据
int* d_cellCount;         // 每个cell当前包含的粒子数量
int* d_cellList;          // cell列表，存储每个cell中的粒子索引
                          // 布局: d_cellList[cellIndex * MAX_PARTICLES_PER_CELL + offset] = particleIndex

int* d_cellOverflow;      // 记录cell溢出次数（用于警告）
// ================================================

int updateInterval = 10;  // 近邻表更新周期
int nPairsMax;            // 近邻表最大容量

// 错误检查宏
#define CUDA_CHECK_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 声明CPU函数
void initPositions();
void initVelocities();
void rescaleVelocities();
double instantaneousTemperature();
void allocateMemory();
void freeMemory();
void copyDataToDevice();
void copyDataFromDevice();

// 声明CUDA核函数
__global__ void computeAccelerationsKernel(double* d_r, double* d_a, int* d_pairList, 
                                          double* d_drPair, double* d_rSqdPair, 
                                          int nPairs, double rCutOff, int N);
__global__ void velocityVerletStep1Kernel(double* d_r, double* d_v, double* d_a, 
                                         double dt, double L, int N);
__global__ void velocityVerletStep2Kernel(double* d_v, double* d_a, double dt, int N);

// 物理循环开始前的位置、速度初始化
void initialize() {
    allocateMemory();
    initPositions();
    initVelocities();
}

// 计算Cell List的参数
void computeCellParameters() {
    // cellSize 应该 >= rMax，这样只需检查27个邻居cell即可覆盖所有可能的近邻
    cellSize = rMax;
    
    // 计算各方向的cell数量（至少为1）
    numCellsX = (int)floor(L / cellSize);
    numCellsY = (int)floor(L / cellSize);
    numCellsZ = (int)floor(L / cellSize);
    if (numCellsX < 1) numCellsX = 1;
    if (numCellsY < 1) numCellsY = 1;
    if (numCellsZ < 1) numCellsZ = 1;
    
    // 重新计算实际的cellSize（确保整除）
    cellSize = L / numCellsX;  // 假设立方体，各方向相同
    
    totalCells = numCellsX * numCellsY * numCellsZ;
    
    cout << "Cell List 参数:" << endl;
    cout << "  盒子边长 L = " << L << endl;
    cout << "  Cell 边长 = " << cellSize << endl;
    cout << "  Cell 数量 = " << numCellsX << " x " << numCellsY << " x " << numCellsZ 
         << " = " << totalCells << endl;
    cout << "  每个Cell最大容量 = " << MAX_PARTICLES_PER_CELL << endl;
}

// 分配内存
void allocateMemory() {
    // 主机内存分配
    h_r = new double* [N];
    h_v = new double* [N];
    h_a = new double* [N];
    for (int i = 0; i < N; i++) {
        h_r[i] = new double[3];
        h_v[i] = new double[3];
        h_a[i] = new double[3];
        // 初始化加速度为0，防止未初始化值导致数值不稳定
        for (int d = 0; d < 3; d++) {
            h_a[i][d] = 0.0;
        }
    }

    // 设备粒子内存分配
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_r, N * 3 * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_v, N * 3 * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_a, N * 3 * sizeof(double)));

    // 设备端近邻表内存分配
    // 近邻表大小估计：平均每个粒子约有 (4/3)π*rMax³*ρ 个邻居
    // 保守估计，给足够的空间
    double avgNeighbors = (4.0 / 3.0) * 3.14159 * rMax * rMax * rMax * rho;
    nPairsMax = (int)(N * avgNeighbors / 2.0 * 1.5);  // 除以2因为每对只算一次，乘1.5留余量
    if (nPairsMax < N * 50) nPairsMax = N * 50;  // 至少给每个粒子50个邻居的空间
    
    cout << "近邻表最大容量: " << nPairsMax << endl;
    
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_pairList, nPairsMax * 2 * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_drPair, nPairsMax * 3 * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_rSqdPair, nPairsMax * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_nPairs, sizeof(int)));

    // Cell List 内存分配（在知道L之后才能确定，这里先预分配）
    // 注意：实际的Cell参数在initPositions之后才能确定，这里先用估计值
    int estimatedCells = (int)(pow(N / rho, 1.0) / (rMax * rMax * rMax));
    if (estimatedCells < 1) estimatedCells = 1;
    if (estimatedCells < 27) estimatedCells = 27;  // 至少27个cell
    
    // 先分配一个较大的空间，后续可能需要重新分配
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_cellCount, estimatedCells * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_cellList, estimatedCells * MAX_PARTICLES_PER_CELL * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_cellOverflow, sizeof(int)));
}

// 重新分配Cell List内存（在知道确切的L之后调用）
void reallocateCellMemory() {
    // 释放旧的Cell内存
    if (d_cellCount) cudaFree(d_cellCount);
    if (d_cellList) cudaFree(d_cellList);
    
    // 根据实际参数重新分配
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_cellCount, totalCells * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_cellList, totalCells * MAX_PARTICLES_PER_CELL * sizeof(int)));
    
    cout << "Cell List 内存已重新分配: " << totalCells << " cells" << endl;
}

// 释放内存
void freeMemory() {
    // 释放主机内存
    for (int i = 0; i < N; i++) {
        delete[] h_r[i];
        delete[] h_v[i];
        delete[] h_a[i];
    }
    delete[] h_r;
    delete[] h_v;
    delete[] h_a;

    // 释放设备内存
    if (d_r) cudaFree(d_r);
    if (d_v) cudaFree(d_v);
    if (d_a) cudaFree(d_a);
    if (d_pairList) cudaFree(d_pairList);
    if (d_drPair) cudaFree(d_drPair);
    if (d_rSqdPair) cudaFree(d_rSqdPair);
    if (d_nPairs) cudaFree(d_nPairs);
    
    // 释放Cell List内存
    if (d_cellCount) cudaFree(d_cellCount);
    if (d_cellList) cudaFree(d_cellList);
    if (d_cellOverflow) cudaFree(d_cellOverflow);
}

// 将粒子数据从主机复制到设备
void copyDataToDevice() {
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

    CUDA_CHECK_ERROR(cudaMemcpy(d_r, h_r_flat, N * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_v, h_v_flat, N * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_a, h_a_flat, N * 3 * sizeof(double), cudaMemcpyHostToDevice));

    delete[] h_r_flat;
    delete[] h_v_flat;
    delete[] h_a_flat;
}

// 只复制速度数据到设备端（用于主循环中的速度缩放）
void copyVelocityToDevice() {
    double* h_v_flat = new double[N * 3];
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < 3; d++) {
            h_v_flat[i * 3 + d] = h_v[i][d];
        }
    }
    CUDA_CHECK_ERROR(cudaMemcpy(d_v, h_v_flat, N * 3 * sizeof(double), cudaMemcpyHostToDevice));
    delete[] h_v_flat;
}

// 将粒子数据从设备复制回主机
void copyDataFromDevice() {
    double* h_r_flat = new double[N * 3];
    double* h_v_flat = new double[N * 3];
    double* h_a_flat = new double[N * 3];

    CUDA_CHECK_ERROR(cudaMemcpy(h_r_flat, d_r, N * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(h_v_flat, d_v, N * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(h_a_flat, d_a, N * 3 * sizeof(double), cudaMemcpyDeviceToHost));

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

// ============== Cell List 核函数 ==============

// 设备端辅助函数：计算粒子所在的cell索引
__device__ int getCellIndex(double x, double y, double z, 
                            double cellSize, int numCellsX, int numCellsY, int numCellsZ, double L) {
    // 处理边界情况：确保坐标在[0, L)范围内
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

// CUDA核函数：构建Cell List
// 每个线程处理一个粒子，将其添加到对应的cell中
__global__ void buildCellListKernel(double* d_r, int* d_cellCount, int* d_cellList,
                                    int* d_cellOverflow,
                                    double cellSize, int numCellsX, int numCellsY, int numCellsZ,
                                    int maxParticlesPerCell, double L, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    // 获取粒子位置
    double x = d_r[i * 3];
    double y = d_r[i * 3 + 1];
    double z = d_r[i * 3 + 2];
    
    // 计算所属cell索引
    int cellIndex = getCellIndex(x, y, z, cellSize, numCellsX, numCellsY, numCellsZ, L);
    
    // 原子操作增加cell中的粒子计数，并获取写入位置
    int offset = atomicAdd(&d_cellCount[cellIndex], 1);
    
    // 检查是否超出容量
    if (offset < maxParticlesPerCell) {
        d_cellList[cellIndex * maxParticlesPerCell + offset] = i;
    } else {
        // 记录溢出（仅用于调试警告）
        atomicAdd(d_cellOverflow, 1);
    }
}

// CUDA核函数：基于Cell List构建近邻表
// 每个线程处理一个粒子，检查其所在cell及27个邻居cell中的粒子
__global__ void buildNeighborListKernel(double* d_r, int* d_cellCount, int* d_cellList,
                                        int* d_pairList, double* d_drPair, double* d_rSqdPair,
                                        int* d_nPairs, int nPairsMax,
                                        double cellSize, int numCellsX, int numCellsY, int numCellsZ,
                                        int maxParticlesPerCell, double rMax, double L, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    // 获取粒子i的位置
    double xi = d_r[i * 3];
    double yi = d_r[i * 3 + 1];
    double zi = d_r[i * 3 + 2];
    
    // 计算粒子i所在的cell坐标
    int cxi = min((int)(xi / cellSize), numCellsX - 1);
    int cyi = min((int)(yi / cellSize), numCellsY - 1);
    int czi = min((int)(zi / cellSize), numCellsZ - 1);
    cxi = max(0, cxi);
    cyi = max(0, cyi);
    czi = max(0, czi);
    
    double rMaxSqd = rMax * rMax;
    
    // 遍历3x3x3的邻居cell（包括自身）
    for (int dcx = -1; dcx <= 1; dcx++) {
        for (int dcy = -1; dcy <= 1; dcy++) {
            for (int dcz = -1; dcz <= 1; dcz++) {
                // 应用周期边界条件计算邻居cell坐标
                int ncx = cxi + dcx;
                int ncy = cyi + dcy;
                int ncz = czi + dcz;
                
                // 周期边界处理
                if (ncx < 0) ncx += numCellsX;
                if (ncx >= numCellsX) ncx -= numCellsX;
                if (ncy < 0) ncy += numCellsY;
                if (ncy >= numCellsY) ncy -= numCellsY;
                if (ncz < 0) ncz += numCellsZ;
                if (ncz >= numCellsZ) ncz -= numCellsZ;
                
                int neighborCellIndex = ncx + ncy * numCellsX + ncz * numCellsX * numCellsY;
                int cellParticleCount = d_cellCount[neighborCellIndex];
                
                // 遍历该cell中的所有粒子
                for (int k = 0; k < cellParticleCount && k < maxParticlesPerCell; k++) {
                    int j = d_cellList[neighborCellIndex * maxParticlesPerCell + k];
                    
                    // 只处理 i < j 的粒子对，避免重复
                    if (j <= i) continue;
                    
                    // 获取粒子j的位置
                    double xj = d_r[j * 3];
                    double yj = d_r[j * 3 + 1];
                    double zj = d_r[j * 3 + 2];
                    
                    // 计算距离（应用周期边界条件）
                    double dx = xi - xj;
                    double dy = yi - yj;
                    double dz = zi - zj;
                    
                    // 周期边界修正
                    if (dx >= 0.5 * L) dx -= L;
                    if (dx < -0.5 * L) dx += L;
                    if (dy >= 0.5 * L) dy -= L;
                    if (dy < -0.5 * L) dy += L;
                    if (dz >= 0.5 * L) dz -= L;
                    if (dz < -0.5 * L) dz += L;
                    
                    double rSqd = dx * dx + dy * dy + dz * dz;
                    
                    // 检查是否在近邻距离内
                    if (rSqd < rMaxSqd) {
                        // 原子操作添加到近邻表
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

// ================================================

// CUDA核函数：表内粒子距离更新
__global__ void computeSeparationKernel(double* d_r, int* d_pairList,
                                        double* d_drPair, double* d_rSqdPair,
                                        int* d_nPairs, double L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_nPairs[0]) {
        return;
    }

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

// 主机端调用核函数更新近邻表内粒子的距离
void updatePairSeparations() {
    CUDA_CHECK_ERROR(cudaMemcpy(&nPairs, d_nPairs, sizeof(int), cudaMemcpyDeviceToHost));
    
    if(nPairs > 0) {
        int blockSize = 256;
        int numBlocks = (nPairs + blockSize - 1) / blockSize;
        computeSeparationKernel<<<numBlocks, blockSize>>>(d_r, d_pairList,
            d_drPair, d_rSqdPair,
            d_nPairs, L);
        CUDA_CHECK_ERROR(cudaGetLastError());
    }
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// 主机端调用核函数更新近邻表（基于Cell List的O(N)版本）
void updatePairList() {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // 1. 清零cell计数和近邻表计数
    CUDA_CHECK_ERROR(cudaMemset(d_cellCount, 0, totalCells * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemset(d_nPairs, 0, sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemset(d_cellOverflow, 0, sizeof(int)));
    
    // 2. 构建Cell List
    buildCellListKernel<<<numBlocks, blockSize>>>(d_r, d_cellCount, d_cellList,
                                                   d_cellOverflow,
                                                   cellSize, numCellsX, numCellsY, numCellsZ,
                                                   MAX_PARTICLES_PER_CELL, L, N);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    // 检查溢出（可选，用于调试）
    int overflow = 0;
    CUDA_CHECK_ERROR(cudaMemcpy(&overflow, d_cellOverflow, sizeof(int), cudaMemcpyDeviceToHost));
    if (overflow > 0) {
        cerr << "警告: Cell溢出 " << overflow << " 个粒子，可能需要增加MAX_PARTICLES_PER_CELL" << endl;
    }
    
    // 3. 基于Cell List构建近邻表
    buildNeighborListKernel<<<numBlocks, blockSize>>>(d_r, d_cellCount, d_cellList,
                                                       d_pairList, d_drPair, d_rSqdPair,
                                                       d_nPairs, nPairsMax,
                                                       cellSize, numCellsX, numCellsY, numCellsZ,
                                                       MAX_PARTICLES_PER_CELL, rMax, L, N);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    // 4. 获取近邻表大小
    CUDA_CHECK_ERROR(cudaMemcpy(&nPairs, d_nPairs, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (nPairs >= nPairsMax) {
        cerr << "警告: 近邻表已满! nPairs=" << nPairs << ", nPairsMax=" << nPairsMax << endl;
    }
}

// CUDA核函数：置零加速度
__global__ void zeroAccelerationsKernel(double* d_a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        d_a[idx * 3] = 0.0;
        d_a[idx * 3 + 1] = 0.0;
        d_a[idx * 3 + 2] = 0.0;
    }
}

// CUDA核函数：计算加速度
__global__ void computeAccelerationsKernel(double* d_r, double* d_a, int* d_pairList, 
                                          double* d_drPair, double* d_rSqdPair, 
                                          int nPairs, double rCutOff, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nPairs) {
        int i = d_pairList[idx * 2];
        int j = d_pairList[idx * 2 + 1];
        double rSqd = d_rSqdPair[idx];

        double epsilon = 1e-9; 
        if (rSqd < epsilon) {
            return;
        }

        if (rSqd < rCutOff * rCutOff) {
            double r2Inv = 1.0 / rSqd;
            double r6Inv = r2Inv * r2Inv * r2Inv;
            double f = 24.0 * r2Inv * r6Inv * (2.0 * r6Inv - 1.0);

            // 使用原子操作更新加速度
            for (int d = 0; d < 3; d++) {
                double force = f * d_drPair[idx * 3 + d];
                
                // 对粒子i的加速度进行原子加法
                unsigned long long int* address_as_ull = (unsigned long long int*)&d_a[i * 3 + d];
                unsigned long long int old = *address_as_ull;
                unsigned long long int assumed;
                do {
                    assumed = old;
                    old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(force + __longlong_as_double(assumed)));
                } while (assumed != old);
                
                // 对粒子j的加速度进行原子加法
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
}

// CUDA核函数：Verlet第一步
__global__ void velocityVerletStep1Kernel(double* d_r, double* d_v, double* d_a, 
                                         double dt, double L, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        for (int d = 0; d < 3; d++) {
            d_r[idx * 3 + d] += d_v[idx * 3 + d] * dt + 0.5 * d_a[idx * 3 + d] * dt * dt;
            
            if (d_r[idx * 3 + d] < 0)
                d_r[idx * 3 + d] += L;
            if (d_r[idx * 3 + d] >= L)
                d_r[idx * 3 + d] -= L;
            
            d_v[idx * 3 + d] += 0.5 * d_a[idx * 3 + d] * dt;
        }
    }
}

// CUDA核函数：Verlet第二步
__global__ void velocityVerletStep2Kernel(double* d_v, double* d_a, double dt, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        for (int d = 0; d < 3; d++) {
            d_v[idx * 3 + d] += 0.5 * d_a[idx * 3 + d] * dt;
        }
    }
}

// 使用CUDA加速的Verlet
void velocityVerlet(double dt) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    int numBlocksPairs = (nPairs + blockSize - 1) / blockSize;

    // 第一步：更新位置和速度的一半
    velocityVerletStep1Kernel<<<numBlocks, blockSize>>>(d_r, d_v, d_a, dt, L, N);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // 更新表内粒子对距离
    updatePairSeparations();

    // 置零加速度
    zeroAccelerationsKernel<<<numBlocks, blockSize>>>(d_a, N);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // 计算新的加速度
    if (nPairs > 0) {
        computeAccelerationsKernel<<<numBlocksPairs, blockSize>>>(d_r, d_a, d_pairList, d_drPair, d_rSqdPair, nPairs, rCutOff, N);
        CUDA_CHECK_ERROR(cudaGetLastError());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    // 第二步：更新速度的另一半
    velocityVerletStep2Kernel<<<numBlocks, blockSize>>>(d_v, d_a, dt, N);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// 位置初始化函数
void initPositions() {
    L = pow(N / rho, 1.0 / 3);

    int M = 1;
    while (4 * M * M * M < N)
        ++M;
    double a = L / M;

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
    
    // 计算Cell参数（需要L已知）
    computeCellParameters();
    
    // 重新分配Cell内存
    reallocateCellMemory();
}

// 高斯随机数生成函数
double gasdev() {
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
    }
    else {
        available = false;
        return gset;
    }
}

// 速度初始化函数
void initVelocities() {
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
    double lambda = sqrt(3 * (N - 1) * T / vSqdSum);
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            h_v[n][i] *= lambda;
}

// 根据目标温度重新设置速度
void rescaleVelocities() {
    copyDataFromDevice();
    
    double vSqdSum = 0;
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            vSqdSum += h_v[n][i] * h_v[n][i];
    double lambda = sqrt(3 * (N - 1) * T / vSqdSum);
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            h_v[n][i] *= lambda;
    
    copyVelocityToDevice();
}

// 即时温度测量函数
double instantaneousTemperature() {
    copyDataFromDevice();
    
    double sum = 0;
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            sum += h_v[i][k] * h_v[i][k];
    return sum / (3 * (N - 1));
}

// 主函数与主循环
int main() {
    cout << "========================================" << endl;
    cout << "MD4_CUDA: 使用Cell List优化的分子动力学模拟" << endl;
    cout << "========================================" << endl;
    
    // 初始化CUDA设备
    int deviceCount;
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "没有找到支持CUDA的设备！" << endl;
        return -1;
    }
    
    CUDA_CHECK_ERROR(cudaSetDevice(0));
    
    // 打印设备信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "使用GPU: " << prop.name << endl;
    cout << "粒子数 N = " << N << endl;
    
    // 初始化指针为nullptr
    d_r = d_v = d_a = nullptr;
    d_pairList = nullptr;
    d_drPair = nullptr;
    d_rSqdPair = nullptr;
    d_nPairs = nullptr;
    d_cellCount = nullptr;
    d_cellList = nullptr;
    d_cellOverflow = nullptr;
    
    // 在CPU完成速度与位置的初始化
    initialize();
    
    // 将数据复制到设备
    copyDataToDevice();
    
    // 在GPU完成近邻表与加速度的初始化
    updatePairList();
    cout << "初始近邻表大小: " << nPairs << endl;
    
    if (nPairs > 0) {
        updatePairSeparations();
        
        // 计算初始加速度
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        zeroAccelerationsKernel<<<numBlocks, blockSize>>>(d_a, N);
        CUDA_CHECK_ERROR(cudaGetLastError());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        
        int numBlocksPairs = (nPairs + blockSize - 1) / blockSize;
        computeAccelerationsKernel<<<numBlocksPairs, blockSize>>>(d_r, d_a, d_pairList, d_drPair, d_rSqdPair, nPairs, rCutOff, N);
        CUDA_CHECK_ERROR(cudaGetLastError());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    double dt = 0.01;
    ofstream file("T4_CUDA.data");
    
    cout << "开始模拟..." << endl;
    
    // 主循环
    for (int i = 0; i < 1000; i++) {
        velocityVerlet(dt);
        
        if (i % 200 == 0)
            rescaleVelocities();
        
        if (i % updateInterval == 0) {
            updatePairList();
            updatePairSeparations();
        }

        double temp = instantaneousTemperature();
        file << temp << ' ' << nPairs << '\n';
        
        // 打印进度
        if (i % 100 == 0) {
            cout << "Step " << i << ": T = " << temp << ", nPairs = " << nPairs << endl;
        }
    }
    
    file.close();
    
    cout << "模拟完成！结果保存到 T4_CUDA.data" << endl;
    
    // 释放内存
    freeMemory();
    
    return 0;
}
