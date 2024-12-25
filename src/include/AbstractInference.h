#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * Output Tensor.
 */
typedef struct OutTensor
{
    std::vector<float> predictions; // Probabilities of output classes.
    float milliseconds = -1; // Elapsed milliseconds of inference.
} OutTensor;

/**
 * Output Tensor for Single inference of Parallel processing.
 */
typedef struct OutParTensor
{
    std::vector<float> predictions; // Probabilities of output classes.
    float milliseconds = -1; // Elapsed milliseconds of inference.
    float offset_milliseconds = -1; // Elapsed milliseconds from the start of inference.
} OutParTensor;

/**
 * Output Tensor for All inferences of Parallel processing.
 */
typedef struct OutParTensors
{
    std::vector<OutParTensor> out_tensors; // All Output Tensors.
    float milliseconds = -1; // Elapsed milliseconds of all inferences.
} OutParTensors;

class AbstractInference
{
public:
    virtual ~AbstractInference() = default;

    [[nodiscard]] virtual OutTensor predict(const cv::Mat& image, std::size_t out_class) const = 0;

    [[nodiscard]] virtual OutParTensors predict_all(const std::vector<cv::Mat>& images, std::size_t out_class) const =
    0;
};
