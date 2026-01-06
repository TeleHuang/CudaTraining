cmake_minimum_required(VERSION 3.18)
project(VisualizedMD4_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

# GLFW - use local pre-compiled binaries
set(GLFW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libs/glfw-3.4.bin.WIN64")
set(GLFW_INCLUDE_DIR "${GLFW_ROOT}/include")
set(GLFW_LIBRARY "${GLFW_ROOT}/lib-vc2022/glfw3.lib")

message(STATUS "Using local GLFW from: ${GLFW_ROOT}")

# Create imported target for GLFW
add_library(glfw STATIC IMPORTED)
set_target_properties(glfw PROPERTIES
    IMPORTED_LOCATION "${GLFW_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE_DIR}"
)

# GLAD - use local pre-generated files
message(STATUS "Using local GLAD from: ${CMAKE_CURRENT_SOURCE_DIR}/glad")

# Source files
set(CUDA_SOURCES
    md4_simulation.cu
    VisualizedMD4_CUDA.cu
)

set(CPP_SOURCES
    visualization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/glad/src/glad.c
)

set(HEADER_FILES
    md4_simulation.cuh
    visualization.h
)

# Create executable
add_executable(${PROJECT_NAME}
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
    ${HEADER_FILES}
)

# GLAD include directory
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/glad/src/glad.c")
    message(FATAL_ERROR "GLAD source file not found! Please download GLAD from https://glad.dav1d.de/ and place glad.c in ${CMAKE_CURRENT_SOURCE_DIR}/glad/src/")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/glad/include
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
    OpenGL::GL
    glfw
)

# CUDA specific settings
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61;70;75;80;86"
)

# Compiler options
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -O2>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
endif()

# Copy executable to source directory for convenience
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Copying executable to source directory"
)

message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "OpenGL Found: ${OPENGL_FOUND}")
