cmake_minimum_required(VERSION 3.23.5)


set(OpenCV_DIR "C:\\OpenCV\\opencv\\build\\x64\\vc16")

########### ACCELERATOR SECTION
set(CPU "CPU")
set(GPU "GPU")
if (${ACCELERATOR} MATCHES ${GPU})
    set(ACCELERATE_GPU 1)
else ()
    set(ACCELERATE_CPU 1)
endif ()
########### METADATA
set(PROJECT_NAME InferenceTest)
set(SOURCES
        main.cpp
        src/FileUtils.cpp
        src/ImageUtils.cpp
        src/GpuInference.cpp
        src/include/Model.h
        src/include/MathUtils.h
        src/include/Logger.h
        src/include/ImageUtils.h
        src/include/FileUtils.h
        src/include/AbstractInference.h
        src/include/GpuInference.h
        src/include/CpuInference.h
        src/CpuInference.cpp
)

########### EXECUTABLE
project(${PROJECT_NAME} LANGUAGES CXX)
add_executable(${PROJECT_NAME} ${SOURCES})

########### COMPILER FLAGS
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


########### INCLUDE OPEN_CV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OPENCV include: ${OpenCV_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
message("OPENCV Successfully linked.")


########### HANDLE CPU/GPU
IF (${ACCELERATE_GPU})
    ########### INCLUDE CUDA
    find_package(CUDAToolkit REQUIRED)
    message("CUDA FOUND")

    list(APPEND PLUGINS "nvinfer_10")
    list(APPEND PLUGINS "nvinfer_plugin_10")
    list(APPEND PLUGINS "nvonnxparser_10")

    foreach (libName ${PLUGINS})
        find_library(${libName}_lib NAMES ${libName} PATH_SUFFIXES lib REQUIRED)
        list(APPEND PLUGIN_LIBS "${${libName}_lib}")
    endforeach ()
    message("TensorRT LIBS: ${PLUGINS}")

    target_link_libraries(${PROJECT_NAME} CUDA::cudart ${PLUGIN_LIBS})
    target_include_directories(${PROJECT_NAME} PRIVATE src/include)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ACCELERATE_GPU=${ACCELERATE_GPU})
ELSE ()
    ########### INCLUDE FRUGALLY_DEEP AND ITS DEPENDENCIES
    include_directories(C:/TensorRT/include)
    target_include_directories(${PROJECT_NAME}
            PRIVATE
            ${PROJECT_SOURCE_DIR}/libs/FunctionalPlus/include
            ${PROJECT_SOURCE_DIR}/libs/json/single_include
            ${PROJECT_SOURCE_DIR}/libs/eigen
            ${PROJECT_SOURCE_DIR}/libs/frugally-deep/include
            src/include
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE ACCELERATE_CPU=${ACCELERATE_CPU})
ENDIF ()