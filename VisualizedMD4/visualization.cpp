/*
 * visualization.cpp
 * 
 * OpenGL Visualization Module - Implementation
 * Uses GLFW for window management and modern OpenGL for rendering
 */

#include "visualization.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Vertex shader source
static const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float pointSize;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    gl_PointSize = pointSize;
}
)";

// Fragment shader source
static const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

uniform vec3 particleColor;

void main() {
    // Make circular points
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (length(coord) > 0.5)
        discard;
    
    // Simple shading for 3D effect
    float dist = length(coord) * 2.0;
    float shade = 1.0 - dist * 0.5;
    
    FragColor = vec4(particleColor * shade, 1.0);
}
)";

// Helper: create 4x4 identity matrix
static void mat4Identity(float* m) {
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

// Helper: create perspective projection matrix
static void mat4Perspective(float* m, float fov, float aspect, float nearZ, float farZ) {
    float tanHalfFov = tanf(fov / 2.0f);
    memset(m, 0, 16 * sizeof(float));
    m[0] = 1.0f / (aspect * tanHalfFov);
    m[5] = 1.0f / tanHalfFov;
    m[10] = -(farZ + nearZ) / (farZ - nearZ);
    m[11] = -1.0f;
    m[14] = -(2.0f * farZ * nearZ) / (farZ - nearZ);
}

// Helper: create look-at view matrix
static void mat4LookAt(float* m, float eyeX, float eyeY, float eyeZ,
                       float centerX, float centerY, float centerZ,
                       float upX, float upY, float upZ) {
    float fx = centerX - eyeX;
    float fy = centerY - eyeY;
    float fz = centerZ - eyeZ;
    float len = sqrtf(fx*fx + fy*fy + fz*fz);
    fx /= len; fy /= len; fz /= len;
    
    float sx = fy * upZ - fz * upY;
    float sy = fz * upX - fx * upZ;
    float sz = fx * upY - fy * upX;
    len = sqrtf(sx*sx + sy*sy + sz*sz);
    sx /= len; sy /= len; sz /= len;
    
    float ux = sy * fz - sz * fy;
    float uy = sz * fx - sx * fz;
    float uz = sx * fy - sy * fx;
    
    m[0] = sx;  m[4] = sy;  m[8]  = sz;  m[12] = -(sx*eyeX + sy*eyeY + sz*eyeZ);
    m[1] = ux;  m[5] = uy;  m[9]  = uz;  m[13] = -(ux*eyeX + uy*eyeY + uz*eyeZ);
    m[2] = -fx; m[6] = -fy; m[10] = -fz; m[14] = (fx*eyeX + fy*eyeY + fz*eyeZ);
    m[3] = 0;   m[7] = 0;   m[11] = 0;   m[15] = 1;
}

// Helper: multiply two 4x4 matrices
static void mat4Multiply(float* result, const float* a, const float* b) {
    float temp[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            temp[i * 4 + j] = 0;
            for (int k = 0; k < 4; k++) {
                temp[i * 4 + j] += a[k * 4 + j] * b[i * 4 + k];
            }
        }
    }
    memcpy(result, temp, 16 * sizeof(float));
}

// Helper: create rotation matrix around X axis
static void mat4RotateX(float* m, float angle) {
    mat4Identity(m);
    float c = cosf(angle);
    float s = sinf(angle);
    m[5] = c;  m[6] = s;
    m[9] = -s; m[10] = c;
}

// Helper: create rotation matrix around Y axis
static void mat4RotateY(float* m, float angle) {
    mat4Identity(m);
    float c = cosf(angle);
    float s = sinf(angle);
    m[0] = c;  m[2] = -s;
    m[8] = s;  m[10] = c;
}

// ============== Visualization Implementation ==============

Visualization::Visualization(const VisualizationConfig& cfg) 
    : config(cfg), window(nullptr), VAO(0), VBO(0), shaderProgram(0),
      particleBuffer(nullptr), numParticles(0), boxSize(1.0f),
      cameraAngleX(0.3f), cameraAngleY(0.0f), cameraZoom(1.0f),
      mousePressed(false), lastMouseX(0), lastMouseY(0),
      displayStep(0), displayTemperature(0), displayNPairs(0) {
}

Visualization::~Visualization() {
    cleanup();
}

bool Visualization::initialize() {
    if (!initGLFW()) return false;
    if (!initOpenGL()) return false;
    if (!createShaders()) return false;
    createBuffers();
    setupCallbacks();
    
    // Enable point sprites and depth testing
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    std::cout << "Visualization initialized successfully" << std::endl;
    return true;
}

bool Visualization::initGLFW() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);  // MSAA
    
    window = glfwCreateWindow(config.windowWidth, config.windowHeight, 
                               config.windowTitle, nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // VSync
    
    return true;
}

bool Visualization::initOpenGL() {
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }
    
    glViewport(0, 0, config.windowWidth, config.windowHeight);
    glEnable(GL_MULTISAMPLE);
    
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    
    return true;
}

bool Visualization::createShaders() {
    // Compile vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
        return false;
    }
    
    // Compile fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
        return false;
    }
    
    // Link shaders
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        return false;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return true;
}

void Visualization::createBuffers() {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Visualization::setupCallbacks() {
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);
}

void Visualization::cleanup() {
    if (particleBuffer) {
        delete[] particleBuffer;
        particleBuffer = nullptr;
    }
    
    if (VAO) glDeleteVertexArrays(1, &VAO);
    if (VBO) glDeleteBuffers(1, &VBO);
    if (shaderProgram) glDeleteProgram(shaderProgram);
    
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
}

bool Visualization::shouldClose() const {
    return window && glfwWindowShouldClose(window);
}

void Visualization::updateParticles(const float* positions, int count, float box) {
    numParticles = count;
    boxSize = box;
    
    // Reallocate buffer if needed
    if (!particleBuffer || numParticles != count) {
        delete[] particleBuffer;
        particleBuffer = new float[count * 3];
    }
    
    // Normalize positions to [-0.5, 0.5] range for rendering
    float halfBox = box * 0.5f;
    for (int i = 0; i < count; i++) {
        particleBuffer[i * 3 + 0] = (positions[i * 3 + 0] - halfBox) / box;
        particleBuffer[i * 3 + 1] = (positions[i * 3 + 1] - halfBox) / box;
        particleBuffer[i * 3 + 2] = (positions[i * 3 + 2] - halfBox) / box;
    }
    
    // Update VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, count * 3 * sizeof(float), particleBuffer, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Visualization::render() {
    // Clear screen
    glClearColor(config.backgroundColor[0], config.backgroundColor[1], 
                 config.backgroundColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    if (numParticles == 0) {
        glfwSwapBuffers(window);
        glfwPollEvents();
        return;
    }
    
    glUseProgram(shaderProgram);
    
    // Calculate camera position
    float camDist = config.cameraDistance * cameraZoom;
    float camX = camDist * sinf(cameraAngleY) * cosf(cameraAngleX);
    float camY = camDist * sinf(cameraAngleX);
    float camZ = camDist * cosf(cameraAngleY) * cosf(cameraAngleX);
    
    // Create matrices
    float model[16], view[16], projection[16];
    mat4Identity(model);
    mat4LookAt(view, camX, camY, camZ, 0, 0, 0, 0, 1, 0);
    
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    float aspect = (float)width / (float)height;
    mat4Perspective(projection, (float)(M_PI / 4.0), aspect, 0.1f, 100.0f);
    
    // Set uniforms
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, model);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, view);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, projection);
    glUniform1f(glGetUniformLocation(shaderProgram, "pointSize"), config.pointSize);
    glUniform3fv(glGetUniformLocation(shaderProgram, "particleColor"), 1, config.particleColor);
    
    // Draw particles
    glBindVertexArray(VAO);
    glDrawArrays(GL_POINTS, 0, numParticles);
    glBindVertexArray(0);
    
    // Draw wireframe box
    // (Simple box outline can be added here if desired)
    
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Visualization::processInput() {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    
    // Reset view with R key
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        cameraAngleX = 0.3f;
        cameraAngleY = 0.0f;
        cameraZoom = 1.0f;
    }
}

void Visualization::setInfoText(int step, double temperature, int nPairs) {
    displayStep = step;
    displayTemperature = temperature;
    displayNPairs = nPairs;
    
    // Update window title with info
    char title[256];
    snprintf(title, sizeof(title), "MD4 Visualization - Step: %d | T: %.3f | Pairs: %d", 
             step, temperature, nPairs);
    glfwSetWindowTitle(window, title);
}

// ============== GLFW Callbacks ==============

void Visualization::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void Visualization::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    Visualization* vis = static_cast<Visualization*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            vis->mousePressed = true;
            glfwGetCursorPos(window, &vis->lastMouseX, &vis->lastMouseY);
        } else if (action == GLFW_RELEASE) {
            vis->mousePressed = false;
        }
    }
}

void Visualization::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    Visualization* vis = static_cast<Visualization*>(glfwGetWindowUserPointer(window));
    if (vis->mousePressed) {
        double dx = xpos - vis->lastMouseX;
        double dy = ypos - vis->lastMouseY;
        
        vis->cameraAngleY += (float)dx * vis->config.rotationSpeed * 0.01f;
        vis->cameraAngleX += (float)dy * vis->config.rotationSpeed * 0.01f;
        
        // Clamp vertical angle
        if (vis->cameraAngleX > (float)(M_PI / 2.0 - 0.1)) 
            vis->cameraAngleX = (float)(M_PI / 2.0 - 0.1);
        if (vis->cameraAngleX < (float)(-M_PI / 2.0 + 0.1)) 
            vis->cameraAngleX = (float)(-M_PI / 2.0 + 0.1);
        
        vis->lastMouseX = xpos;
        vis->lastMouseY = ypos;
    }
}

void Visualization::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    Visualization* vis = static_cast<Visualization*>(glfwGetWindowUserPointer(window));
    vis->cameraZoom -= (float)yoffset * vis->config.zoomSpeed;
    if (vis->cameraZoom < 0.3f) vis->cameraZoom = 0.3f;
    if (vis->cameraZoom > 5.0f) vis->cameraZoom = 5.0f;
}

void Visualization::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        Visualization* vis = static_cast<Visualization*>(glfwGetWindowUserPointer(window));
        
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, true);
                break;
            case GLFW_KEY_R:
                vis->cameraAngleX = 0.3f;
                vis->cameraAngleY = 0.0f;
                vis->cameraZoom = 1.0f;
                break;
            case GLFW_KEY_UP:
                vis->config.pointSize += 1.0f;
                if (vis->config.pointSize > 20.0f) vis->config.pointSize = 20.0f;
                break;
            case GLFW_KEY_DOWN:
                vis->config.pointSize -= 1.0f;
                if (vis->config.pointSize < 1.0f) vis->config.pointSize = 1.0f;
                break;
        }
    }
}
