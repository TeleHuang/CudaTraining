/*
 * visualization.h
 * 
 * OpenGL Visualization Module - Header
 * Handles 3D rendering of particle positions using GLFW + OpenGL
 */

#ifndef VISUALIZATION_H
#define VISUALIZATION_H

// Forward declare GLFW types to avoid including heavy headers here
struct GLFWwindow;

// Visualization configuration
struct VisualizationConfig {
    int windowWidth;
    int windowHeight;
    const char* windowTitle;
    float pointSize;
    float backgroundColor[3];
    float particleColor[3];
    
    // Camera settings
    float cameraDistance;
    float rotationSpeed;
    float zoomSpeed;
    
    VisualizationConfig() :
        windowWidth(1024),
        windowHeight(768),
        windowTitle("MD4 Visualization"),
        pointSize(4.0f),
        cameraDistance(2.0f),
        rotationSpeed(0.5f),
        zoomSpeed(0.1f)
    {
        backgroundColor[0] = 0.1f;
        backgroundColor[1] = 0.1f;
        backgroundColor[2] = 0.15f;
        particleColor[0] = 0.2f;
        particleColor[1] = 0.6f;
        particleColor[2] = 1.0f;
    }
};

// Visualization class
class Visualization {
public:
    Visualization(const VisualizationConfig& config = VisualizationConfig());
    ~Visualization();
    
    // Initialize OpenGL context and window
    bool initialize();
    
    // Check if window should close
    bool shouldClose() const;
    
    // Update particle positions
    // positions: array of N*3 floats (x,y,z for each particle)
    // numParticles: number of particles
    // boxSize: size of the simulation box (for normalization)
    void updateParticles(const float* positions, int numParticles, float boxSize);
    
    // Render one frame
    void render();
    
    // Process input events
    void processInput();
    
    // Get window pointer (for advanced usage)
    GLFWwindow* getWindow() const { return window; }
    
    // Display info on screen (optional)
    void setInfoText(int step, double temperature, int nPairs);

private:
    VisualizationConfig config;
    GLFWwindow* window;
    
    // OpenGL objects
    unsigned int VAO, VBO;
    unsigned int shaderProgram;
    
    // Particle data
    float* particleBuffer;
    int numParticles;
    float boxSize;
    
    // Camera state
    float cameraAngleX;
    float cameraAngleY;
    float cameraZoom;
    bool mousePressed;
    double lastMouseX, lastMouseY;
    
    // Info display
    int displayStep;
    double displayTemperature;
    int displayNPairs;
    
    // Private methods
    bool initGLFW();
    bool initOpenGL();
    bool createShaders();
    void createBuffers();
    void setupCallbacks();
    void cleanup();
    
    // Static callbacks for GLFW
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
};

#endif // VISUALIZATION_H
