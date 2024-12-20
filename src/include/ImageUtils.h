#pragma once


#include <opencv2/opencv.hpp>
#include "Model.h"

cv::Mat load_image(const std::string &path);

cv::Mat modify_image(const cv::Mat &image);

std::vector<float> to_vector_input(const cv::Mat &image);