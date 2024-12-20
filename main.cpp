#include <ImageUtils.h>
#include <fstream>

#include "CpuInference.h"
#include "FileUtils.h"
#include "GpuInference.h"

std::vector<cv::Mat> load_images() {
    const std::vector<std::string> image_paths = find_all_images(R"(C:\Users\honza\Downloads\numbers)");
    std::vector<cv::Mat> images;
    images.reserve(image_paths.size());
    for (const auto &image_path: image_paths) {
        images.push_back(load_image(image_path));
    }
    return images;
}

#if ACCELERATE_GPU

void test_parallel(const GpuInference &inference, const std::vector<cv::Mat> &images) {
    const auto par_tensors = inference.predict_all(images, MODEL_OUTPUT_CLASS);

    std::cout << "Offset;Duration;\n";
    for (const auto &par_tensor: par_tensors.out_tensors) {
        std::cout << par_tensor.offset_milliseconds << ";" << par_tensor.milliseconds << ";\n";
    }
}

void test() {
    const auto images = load_images();
    const auto inference = GpuInference(R"(C:\Users\honza\CLionProjects\InferenceTest\models\model.onnx)");
    test_parallel(inference, images);
}

#elif ACCELERATE_CPU

void test_parallel(const CpuInference &inference, const std::vector<cv::Mat> &mats) {
    const auto out_mul_tensor = inference.predict_all(mats, MODEL_OUTPUT_CLASS);
    for (const auto &par_tensor: out_mul_tensor.out_tensors) {
        std::cout << par_tensor.offset_milliseconds << ";" << par_tensor.milliseconds << ";\n";
    }
}

void test() {
    const auto images = load_images();
    const auto inference = CpuInference(R"(C:\Users\honza\CLionProjects\InferenceTest\models\fdeep_model.json)");
    test_parallel(inference, images);
}

#endif

int main() {
    test();
}
