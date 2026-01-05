/*
   ConsoleApplicationMD3_CUDA.cu

   使用Verlet和动态近邻表方法计算液态氩的分子动力学 - CUDA加速版本

   主循环当中包含每个物理帧进行一次的Verlet和每10个物理帧进行一次近邻表更新
   Verlet的时间复杂度是O(N)，近邻表更新的时间复杂度是O(N²)，在CUDA化以后是O(1)，所以效率很高，但对应地空间复杂度增加了

   在主物理循环中跑的核心代码包括：
   ├──Verlet
   │   ├──Verlet前半步CUDA核函数
   │   ├──计算表内粒子对之间的距离
   │   │    └──表内距离CUDA核函数
   │   ├──置零加速度CUDA核函数
   │   ├──更新加速度CUDA核函数
   │   └──Verlet后半步CUDA核函数
   └──更新近邻表
        └──判别是否要加入近邻表的CUDA核函数
   

   如需执行，请在终端输入如下两行命令：
   nvcc -arch=sm_61 ConsoleApplicationMD3_CUDA.cu -o MD3_CUDA
   ./MD3_CUDA.exe
   其中-arch=sm_61是GPU架构，这里对应的是1050显卡，可自行修改

   调试运行命令
   nvcc -arch=sm_61 ConsoleApplicationMD3_CUDA.cu -g -G -o MD3_CUDA
    cuda-memcheck ./MD3_CUDA.exe
*/

//   现在我对它挺满意的，不过其实还有一个很好的算法是基于空间网格划分的并行计算，以后可以试试
//   经过实验发现，1050显卡在5000粒子数时能正常工作，七秒完成一千步计算
//   但是在2万粒子数时，任务管理器当中可以看到GPU行为的异常，3D核心占用量在爆满和空闲之间反复横跳，以5秒为周期
//   初始帧的计算时长超过5秒，疑似单个物理帧被拆解成多步进行




#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// 仿真参数
int N = 864;              // 氩原子数
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
int** h_pairList;         // 近邻表（主机端）
int* h_pairListMax;       // 最大近邻表（主机端）
double** h_drPair;        // 每个对的朝向 (i,j)（主机端）
double* h_rSqdPair;       // 每个对的距离(i,j)（主机端）

// 设备端近邻表数据
int* d_nPairs;			  // 当前近邻表粒子对数量（设备端）
int* d_pairList;          // 近邻表（设备端）
int* d_pairListMax;       // 最大近邻表（设备端）
double* d_drPair;         // 每个对的朝向 (i,j)（设备端）
double* d_rSqdPair;       // 每个对的距离(i,j)（设备端）

int updateInterval = 10;  // 近邻表更新周期

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
    int nPairsMax = N * (N - 1) / 2;
    h_pairListMax = new int[nPairsMax * 2]; // 添加主机端内存分配
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_pairList, nPairsMax * 2 * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_pairListMax, nPairsMax * 2 * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_drPair, nPairsMax * 3 * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_rSqdPair, nPairsMax * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_nPairs, sizeof(int)));

    //填充最大近邻表
    int pairIdx = 0;
    for (int p1 = 0; p1 < N; p1++) {
        for (int p2 = p1 + 1; p2 < N; p2++) {
            h_pairListMax[pairIdx * 2] = p1;
            h_pairListMax[pairIdx * 2 + 1] = p2;
            pairIdx++;
        }
    }

    //将最大近邻表复制到设备端
	CUDA_CHECK_ERROR(cudaMemcpy(d_pairListMax, h_pairListMax, nPairsMax * 2 * sizeof(int), cudaMemcpyHostToDevice));
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

    delete[] h_pairListMax; // 这个是在主机端分配了的，需要释放

    // 释放设备内存
    if (d_r) cudaFree(d_r);
    if (d_v) cudaFree(d_v);
    if (d_a) cudaFree(d_a);
    if (d_pairList) cudaFree(d_pairList);
    if (d_pairListMax) cudaFree(d_pairListMax); // 添加释放
    if (d_drPair) cudaFree(d_drPair);
    if (d_rSqdPair) cudaFree(d_rSqdPair);
    if (d_nPairs) cudaFree(d_nPairs);       // 添加释放
}

// 将粒子数据从主机复制到设备
void copyDataToDevice() {
    // 将位置、速度和加速度数据各自转换为一维数组，形式为：粒子1号的rx,ry,rz，粒子2号的rx,ry,rz，...
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

    // 复制到设备
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
    // 分配临时一维数组
    double* h_r_flat = new double[N * 3];
    double* h_v_flat = new double[N * 3];
    double* h_a_flat = new double[N * 3];

    // 从设备复制数据
    CUDA_CHECK_ERROR(cudaMemcpy(h_r_flat, d_r, N * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(h_v_flat, d_v, N * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(h_a_flat, d_a, N * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    // 转换回二维数组格式
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

// CUDA核函数：表内粒子距离更新
__global__ void computeSeparationKernal(double* d_r, int* d_pairList,
                                        double* d_drPair, double* d_rSqdPair,
                                        int* d_nPairs, double L) {
    // 计算当前线程的粒子对编号
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_nPairs[0]) {
		return;
	}

    // 获取两个粒子编号
	int p1 = d_pairList[2 * i];
	int p2 = d_pairList[2 * i + 1];

    // 计算周期边界条件下的距离
    d_rSqdPair[i] = 0;
    for (int d = 0; d < 3; d++) {// 遍历三个维度
		d_drPair[i * 3 + d] = d_r[p1 * 3 + d] - d_r[p2 * 3 + d]; // 距离

        //应用周期边界条件
        if (d_drPair[i * 3 + d] >= 0.5 * L)
            d_drPair[i * 3 + d] -= L;
        if (d_drPair[i * 3 + d] < -0.5 * L)
            d_drPair[i * 3 + d] += L;

        d_rSqdPair[i] += d_drPair[i * 3 + d] * d_drPair[i * 3 + d]; // 距离的平方
    }
}

// 主机端调用核函数更新近邻表内粒子的距离
void updatePairSeparations() {
        //使用computeSeparationKernal核函数计算表内粒子对之间的距离
    // 从设备获取当前近邻表数量
    CUDA_CHECK_ERROR(cudaMemcpy(&nPairs, d_nPairs, sizeof(int), cudaMemcpyDeviceToHost));
    
    if(nPairs > 0) {
        int blockSize = 256;
        int numBlocks = (nPairs + blockSize - 1) / blockSize;
        computeSeparationKernal<<<numBlocks, blockSize>>>(d_r, d_pairList,
            d_drPair, d_rSqdPair,
            d_nPairs, L);
        CUDA_CHECK_ERROR(cudaGetLastError());
    }
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// CUDA核函数：根据粒子对距离判定是否放入近邻表
__global__ void separationJudgeKernal(double* d_r, int* d_pairList, int* d_pairListMax,
                                 double* d_drPair, double* d_rSqdPair,
                                 int* d_nPairs, double rMax,  int nPairsMax, double L)
{
	// 计算当前线程的粒子对编号
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nPairsMax) {
        return;
        }

	// 最大的粒子对列表d_pairListMax是一个上三角矩阵
    // 如果N=3那么矩阵是这个样子：
    // [0 1 ，0 2]
    // [ -  ，1 2]，其中左下角没有元素，总元素数为N*(N-1)/2
    // 在主程序初始化阶段已经用简单的双循环创建固定的d_pairListMax
	// 下面根据d_pairListMax获取两个粒子的编号
    int p1 = d_pairListMax[2 * i];
    int p2 = d_pairListMax[2 * i + 1];

    //先全部计算一遍距离，也就是让第i个线程计算第i组粒子对的距离
    d_rSqdPair[i] = 0;
    for (int d = 0; d < 3; d++) {// 遍历三个维度
        // 从绝对位置计算相对位置
        d_drPair[i * 3 + d] = d_r[p1 * 3 + d] - d_r[p2 * 3 + d];

        //应用周期边界条件调整相对位置取值
        if (d_drPair[i * 3 + d] >= 0.5 * L)
            d_drPair[i * 3 + d] -= L;
        if (d_drPair[i * 3 + d] < -0.5 * L)
            d_drPair[i * 3 + d] += L;

        // 勾股定律计算距离的平方
        d_rSqdPair[i] += d_drPair[i * 3 + d] * d_drPair[i * 3 + d]; 
    }

	// 再判别距离,决定是否加入到近邻表原子加法
	if (d_rSqdPair[i] < rMax * rMax) {
        int idx = atomicAdd(d_nPairs, 1);
        d_pairList[idx * 2] = d_pairListMax[i * 2];
        d_pairList[idx * 2 + 1] = d_pairListMax[i * 2 + 1];
	}
}

// 主机端调用核函数更新近邻表
// 这实际上是本程序唯一一个计算需求以N²规律剧烈增长的部分
void updatePairList() {
    // 将当前近邻表粒子对数量初始化为0
    cudaMemset(d_nPairs, 0, sizeof(int));

    int nPairsMax = (N * (N - 1)) / 2; // 最大粒子对数
    // 先在最大近邻表上算全部N*（N-1）/2对距离，后判别距离，然后增长近邻表，对d_nPairs进行原子加法
	separationJudgeKernal << <(nPairsMax + 255) / 256, 256 >> > ( d_r, d_pairList, d_pairListMax,
                                                                  d_drPair, d_rSqdPair,
                                                                  d_nPairs, rMax, nPairsMax, L);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // 将d_nPairs从设备复制回主机
    CUDA_CHECK_ERROR(cudaMemcpy(&nPairs, d_nPairs, sizeof(int), cudaMemcpyDeviceToHost));

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
    // 每个线程计算自己的索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 计算粒子对之间的力
    if (idx < nPairs) {// 每个线程处理一个粒子对
        int i = d_pairList[idx * 2];
        int j = d_pairList[idx * 2 + 1];
        double rSqd = d_rSqdPair[idx];

        // --- 开始添加检查 ---
        // 添加一个小的阈值来防止距离过近导致的计算问题
        double epsilon = 1e-9; 
        if (rSqd < epsilon) {
            // 如果距离平方小于阈值，可以选择跳过这对粒子的力计算
            // 或者打印一个警告（但注意内核printf可能影响性能）
            // printf("Warning: Particle pair %d-%d too close (rSqd=%.6e), skipping force calculation.\\n", i, j, rSqd);
            return; // 直接返回，不计算这对的力贡献
        }
        // --- 结束添加检查 ---


        if (rSqd < rCutOff * rCutOff) {

            // 计算力
            double r2Inv = 1.0 / rSqd;
            double r6Inv = r2Inv * r2Inv * r2Inv;
            double f = 24.0 * r2Inv * r6Inv * (2.0 * r6Inv - 1.0);

            // 使用原子操作更新加速度（使用CAS循环实现double类型的原子加法，SM61不支持直接使用double类型原子加法）
            // 注意: Compute Capability 6.1 (sm_61) *确实* 支持 double 类型的 atomicAdd。
            // 可以简化为:
            // double force_component_i = f * d_drPair[idx * 3 + d];
            // atomicAdd(&d_a[i * 3 + d], force_component_i);
            // atomicAdd(&d_a[j * 3 + d], -force_component_i);
            // 但暂时保留原有的 CAS 实现，以防万一
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

// CUDA核函数：Verlet第一步（用上一次迭代的数据更新位置，完成速度更新的一半）
__global__ void velocityVerletStep1Kernel(double* d_r, double* d_v, double* d_a, 
                                         double dt, double L, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        for (int d = 0; d < 3; d++) {

            // 首先用上一次迭代的速度和加速度数据更新位置
            d_r[idx * 3 + d] += d_v[idx * 3 + d] * dt + 0.5 * d_a[idx * 3 + d] * dt * dt;
            
            // 应用周期边界条件，将超出边界的粒子传送到另一侧
            if (d_r[idx * 3 + d] < 0)
                d_r[idx * 3 + d] += L;
            if (d_r[idx * 3 + d] >= L)
                d_r[idx * 3 + d] -= L;
            
            // 根据上一个迭代的加速度数据的一半，来更新速度
            d_v[idx * 3 + d] += 0.5 * d_a[idx * 3 + d] * dt;
        }
    }
}

// CUDA核函数：Verlet第二步（根据本次迭代的加速度完成速度更新的另一半）
__global__ void velocityVerletStep2Kernel(double* d_v, double* d_a, double dt, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        for (int d = 0; d < 3; d++) {
            // 速度更新的另一半
            d_v[idx * 3 + d] += 0.5 * d_a[idx * 3 + d] * dt;
        }
    }
}

// 使用CUDA加速的Verlet
void velocityVerlet(double dt) {
    // 根据需要设置CUDA线程块和网格大小
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
    computeAccelerationsKernel<<<numBlocksPairs, blockSize>>>(d_r, d_a, d_pairList, d_drPair, d_rSqdPair, nPairs, rCutOff, N);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // 第二步：更新速度的另一半
    velocityVerletStep2Kernel<<<numBlocks, blockSize>>>(d_v, d_a, dt, N);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// 位置初始化函数
void initPositions() {
    // 通过总粒子数N和粒子数密度RHO计算正方体边长
    L = pow(N / rho, 1.0 / 3);

    // 找到一个足够大的晶胞数，M立方，以容纳全部N个原子
    int M = 1;
    while (4 * M * M * M < N) // 一个面心立方晶胞的原子数是4
        ++M;
    double a = L / M;           // 计算得到每个晶胞的边长a

    // 晶胞中 4 个原子的位置
    double xCell[4] = { 0.25, 0.75, 0.75, 0.25 };
    double yCell[4] = { 0.25, 0.75, 0.25, 0.75 };
    double zCell[4] = { 0.25, 0.25, 0.75, 0.75 };

    int n = 0;                  // 放置全部的N个原子
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
    // 单位标准差的高斯分布
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            h_v[n][i] = gasdev();
    
    // 将平均速度置零
    double vCM[3] = { 0, 0, 0 };
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            vCM[i] += h_v[n][i];
    for (int i = 0; i < 3; i++)
        vCM[i] /= N;
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            h_v[n][i] -= vCM[i];

    // 重新等比例缩放所有粒子的速度以达到想要的温度（只在主机端操作，不涉及设备端）
    double vSqdSum = 0;
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            vSqdSum += h_v[n][i] * h_v[n][i];
    double lambda = sqrt(3 * (N - 1) * T / vSqdSum);
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            h_v[n][i] *= lambda;
    // 注意：这里不调用copyDataToDevice()，因为设备端内存可能还未分配
    // 数据将在main()中的copyDataToDevice()调用时复制到设备
}

// 根据目标温度重新设置速度
void rescaleVelocities() {
    // 先从设备端同步最新的速度数据到主机端
    copyDataFromDevice();
    
    double vSqdSum = 0;
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            vSqdSum += h_v[n][i] * h_v[n][i];
    double lambda = sqrt(3 * (N - 1) * T / vSqdSum);
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            h_v[n][i] *= lambda;
    
    // 只更新设备端速度数据，不覆盖位置和加速度
    copyVelocityToDevice();
}

// 即时温度测量函数
double instantaneousTemperature() {
    // 从设备复制速度数据到主机
    copyDataFromDevice();
    
    double sum = 0;
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            sum += h_v[i][k] * h_v[i][k];
    return sum / (3 * (N - 1));
}

// 主函数与主循环
int main() {
    // 初始化CUDA设备
    int deviceCount;
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "没有找到支持CUDA的设备！" << endl;
        return -1;
    }
    
    // 选择第一个CUDA设备
    CUDA_CHECK_ERROR(cudaSetDevice(0));
    
    // 初始化指针为nullptr
    d_r = d_v = d_a = nullptr;
    d_pairList = nullptr;
    d_drPair = nullptr;
    d_rSqdPair = nullptr;
    d_nPairs = nullptr; // 确保 d_nPairs 也初始化为 nullptr
    
    // 在CPU完成速度与位置的初始化
    initialize();
    
    // 将数据复制到设备
    copyDataToDevice();
    
    // 在GPU完成近邻表与加速度的初始化
    updatePairList();
    CUDA_CHECK_ERROR(cudaMemcpy(&nPairs, d_nPairs, sizeof(int), cudaMemcpyDeviceToHost));
    
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
    ofstream file("T3_CUDA.data");
    
    // 主循环
    for (int i = 0; i < 1000; i++) {

        // 主物理运算，包含三个设备端函数
        velocityVerlet(dt);
        
        if (i % 200 == 0)
            rescaleVelocities();
        
        if (i % updateInterval == 0) {// 每10步更新一次近邻表
			updatePairList();// 这是一个O（n^2)操作，应当放入CUDA当中，但是还没做
            updatePairSeparations();
        }

        // 在文本文件中写入温度与近邻表大小
        file << instantaneousTemperature() << ' ' << nPairs << '\n';
    }
    
    file.close();
    
    // 释放内存
    freeMemory();
    
    return 0;
}