//
// Created by hanah on 05.12.2024.
//

#include "ImageUtils.h"
#include <onnx>


cv::Mat load_image(const std::string &path) {
    return imread(path, cv::IMREAD_GRAYSCALE);
}

cv::Mat modify_image(const cv::Mat &image) {
    auto modified = cv::Mat(image);
    resize(modified, modified, cv::Size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE));
    modified.convertTo(modified, CV_32FC1);
    modified /= 255.0f;
    return modified;
}

std::vector<float> to_vector_input(const cv::Mat &image) {
    std::vector<float> input;
    for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
        for (int j = 0; j < MODEL_INPUT_SIZE; j++) {
            input.push_back(image.at<float>(i, j));
        }
    }
    return input;
}