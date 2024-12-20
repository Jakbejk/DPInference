#pragma once

constexpr int MODEL_INPUT_SIZE = 28;
constexpr int MODEL_OUTPUT_CLASS = 10;

#if !defined(ACCELERATE_CPU) && !defined(ACCELERATE_GPU)
    #define ACCELERATE_CPU 1
#endif
