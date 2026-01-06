/*
 * md4_simulation.cuh
 * 
 * MD4 Molecular Dynamics Simulation Module - Header
 * Encapsulates the Cell List optimized MD simulation logic
 */

#ifndef MD4_SIMULATION_CUH
#define MD4_SIMULATION_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Simulation parameters structure
struct SimulationParams {
    int N;                  // Number of particles
    double rho;             // Density
    double T;               // Target temperature
    double L;               // Box length
    double dt;              // Time step
    double rCutOff;         // Force cutoff distance
    double rMax;            // Neighbor list cutoff
    int updateInterval;     // Neighbor list update interval
};

// MD4 Simulation class
class MD4Simulation {
public:
    // Constructor and destructor
    MD4Simulation(int numParticles = 864, double density = 0.8, double temperature = 1.0);
    ~MD4Simulation();
    
    // Initialization
    bool initialize();
    
    // Simulation step
    void step();
    
    // Get particle positions (copies from device to provided buffer)
    // positions should be pre-allocated with size N * 3
    void getPositions(float* positions);
    
    // Get simulation parameters
    const SimulationParams& getParams() const { return params; }
    int getNumParticles() const { return params.N; }
    double getBoxLength() const { return params.L; }
    double getTemperature();
    int getNeighborPairCount() const { return nPairs; }
    int getCurrentStep() const { return currentStep; }
    
    // Control
    void rescaleVelocities();

private:
    // Parameters
    SimulationParams params;
    int currentStep;
    
    // Host data
    double** h_r;           // Positions (host)
    double** h_v;           // Velocities (host)
    double** h_a;           // Accelerations (host)
    
    // Device data
    double* d_r;            // Positions (device)
    double* d_v;            // Velocities (device)
    double* d_a;            // Accelerations (device)
    
    // Neighbor list data
    int nPairs;
    int nPairsMax;
    int* d_nPairs;
    int* d_pairList;
    double* d_drPair;
    double* d_rSqdPair;
    
    // Cell list data
    double cellSize;
    int numCellsX, numCellsY, numCellsZ;
    int totalCells;
    int* d_cellCount;
    int* d_cellList;
    int* d_cellOverflow;
    
    static const int MAX_PARTICLES_PER_CELL = 64;
    
    // Private methods
    void allocateMemory();
    void freeMemory();
    void initPositions();
    void initVelocities();
    void computeCellParameters();
    void reallocateCellMemory();
    void copyDataToDevice();
    void copyDataFromDevice();
    void copyVelocityToDevice();
    void updatePairList();
    void updatePairSeparations();
    void velocityVerlet();
    double gasdev();
};

#endif // MD4_SIMULATION_CUH
