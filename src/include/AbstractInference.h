#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

typedef struct OutTensor {
    std::vector<float> predictions;
    float milliseconds = -1;
} OutTensor;

typedef struct OutMulTensor {
    std::vector<float> predictions;
    float milliseconds = -1;
    float offset_milliseconds = -1;
} OutMulTensor;

typedef struct OutMulTensors {
    std::vector<OutMulTensor> out_tensors;
    float milliseconds = -1;
} OutMulTensors;

class AbstractInference {
public:
    virtual ~AbstractInference() = default;

    [[nodiscard]] virtual OutTensor predict(const cv::Mat &image, std::size_t out_class) const = 0;

    [[nodiscard]] virtual OutMulTensors predict_all(const std::vector<cv::Mat> &images, std::size_t out_class) const =
    0;
};
