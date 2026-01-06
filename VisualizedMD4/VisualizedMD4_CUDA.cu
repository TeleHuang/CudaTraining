/*
 * VisualizedMD4_CUDA.cu
 * 
 * Main Program - Real-time 3D Visualization of MD4 Simulation
 * 
 * This program combines the MD4 molecular dynamics simulation with
 * real-time OpenGL visualization using GLFW.
 * 
 * Controls:
 *   - Left mouse drag: Rotate view
 *   - Scroll wheel: Zoom in/out
 *   - R: Reset camera view
 *   - Up/Down arrows: Increase/decrease particle size
 *   - ESC: Exit
 * 
 * Build instructions: See CMakeLists.txt
 */

#include "md4_simulation.cuh"
#include "visualization.h"
#include <iostream>
#include <chrono>
#include <thread>

// Configuration
struct AppConfig {
    // Simulation parameters
    int numParticles = 864;
    double density = 0.8;
    double temperature = 1.0;
    
    // Visualization parameters
    int stepsPerFrame = 5;      // Simulation steps per render frame
    int targetFPS = 60;         // Target frame rate
    bool showStats = true;      // Show statistics in title
    
    // Visualization style
    float pointSize = 5.0f;
    float backgroundColor[3] = {0.05f, 0.05f, 0.1f};
    float particleColor[3] = {0.3f, 0.7f, 1.0f};
};

void printUsage() {
    std::cout << "\n=== MD4 Visualized Molecular Dynamics Simulation ===" << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "  Left mouse drag : Rotate view" << std::endl;
    std::cout << "  Scroll wheel    : Zoom in/out" << std::endl;
    std::cout << "  R               : Reset camera" << std::endl;
    std::cout << "  Up/Down arrows  : Change particle size" << std::endl;
    std::cout << "  ESC             : Exit" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    AppConfig appConfig;
    
    // Parse command line arguments (simple version)
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            appConfig.numParticles = std::atoi(argv[++i]);
        } else if (arg == "-spf" && i + 1 < argc) {
            appConfig.stepsPerFrame = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -n <N>      Number of particles (default: 864)" << std::endl;
            std::cout << "  -spf <N>    Simulation steps per frame (default: 5)" << std::endl;
            return 0;
        }
    }
    
    printUsage();
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Particles: " << appConfig.numParticles << std::endl;
    std::cout << "  Steps per frame: " << appConfig.stepsPerFrame << std::endl;
    std::cout << std::endl;
    
    // Initialize simulation
    std::cout << "Initializing simulation..." << std::endl;
    MD4Simulation simulation(appConfig.numParticles, appConfig.density, appConfig.temperature);
    
    if (!simulation.initialize()) {
        std::cerr << "Failed to initialize simulation!" << std::endl;
        return -1;
    }
    
    // Initialize visualization
    std::cout << "Initializing visualization..." << std::endl;
    VisualizationConfig visConfig;
    visConfig.windowWidth = 1280;
    visConfig.windowHeight = 720;
    visConfig.windowTitle = "MD4 Molecular Dynamics";
    visConfig.pointSize = appConfig.pointSize;
    visConfig.backgroundColor[0] = appConfig.backgroundColor[0];
    visConfig.backgroundColor[1] = appConfig.backgroundColor[1];
    visConfig.backgroundColor[2] = appConfig.backgroundColor[2];
    visConfig.particleColor[0] = appConfig.particleColor[0];
    visConfig.particleColor[1] = appConfig.particleColor[1];
    visConfig.particleColor[2] = appConfig.particleColor[2];
    
    Visualization vis(visConfig);
    
    if (!vis.initialize()) {
        std::cerr << "Failed to initialize visualization!" << std::endl;
        return -1;
    }
    
    // Allocate position buffer
    int N = simulation.getNumParticles();
    float* positions = new float[N * 3];
    
    // Get initial positions
    simulation.getPositions(positions);
    vis.updateParticles(positions, N, (float)simulation.getBoxLength());
    
    std::cout << "\nStarting simulation loop..." << std::endl;
    std::cout << "Press ESC to exit.\n" << std::endl;
    
    // Timing variables
    auto lastTime = std::chrono::high_resolution_clock::now();
    double frameTimeTarget = 1.0 / appConfig.targetFPS;
    int frameCount = 0;
    double fpsTimer = 0.0;
    double avgFPS = 0.0;
    
    // Main loop
    while (!vis.shouldClose()) {
        auto frameStart = std::chrono::high_resolution_clock::now();
        
        // Process input
        vis.processInput();
        
        // Run simulation steps
        for (int s = 0; s < appConfig.stepsPerFrame; s++) {
            simulation.step();
        }
        
        // Get updated positions
        simulation.getPositions(positions);
        
        // Update visualization
        vis.updateParticles(positions, N, (float)simulation.getBoxLength());
        
        // Update info display
        if (appConfig.showStats) {
            double temp = simulation.getTemperature();
            vis.setInfoText(simulation.getCurrentStep(), temp, simulation.getNeighborPairCount());
        }
        
        // Render
        vis.render();
        
        // Frame timing
        auto frameEnd = std::chrono::high_resolution_clock::now();
        double frameTime = std::chrono::duration<double>(frameEnd - frameStart).count();
        
        // FPS calculation
        frameCount++;
        fpsTimer += frameTime;
        if (fpsTimer >= 1.0) {
            avgFPS = frameCount / fpsTimer;
            frameCount = 0;
            fpsTimer = 0.0;
            
            // Print stats to console periodically
            std::cout << "Step: " << simulation.getCurrentStep() 
                      << " | T: " << simulation.getTemperature()
                      << " | Pairs: " << simulation.getNeighborPairCount()
                      << " | FPS: " << avgFPS << std::endl;
        }
        
        // Frame rate limiting (if rendering is faster than target)
        if (frameTime < frameTimeTarget) {
            double sleepTime = frameTimeTarget - frameTime;
            std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime * 0.9));
        }
    }
    
    // Cleanup
    delete[] positions;
    
    std::cout << "\nSimulation ended." << std::endl;
    std::cout << "Final step: " << simulation.getCurrentStep() << std::endl;
    
    return 0;
}
