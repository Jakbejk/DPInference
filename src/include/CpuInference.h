#pragma once
#if ACCELERATE_CPU

#include <fdeep/fdeep.hpp>
#include "AbstractInference.h"

class CpuInference : public AbstractInference {
    fdeep::model model_;

    static fdeep::tensor to_tensor(const cv::Mat &image);
public:

    CpuInference(const std::string &model_path);

    ~CpuInference();

    [[nodiscard]] OutTensor predict(const cv::Mat &image, std::size_t out_class) const;

    [[nodiscard]] OutParTensors predict_all(const std::vector<cv::Mat> &images, std::size_t out_class) const;
};
#endif

