#include <ImageUtils.h>
#include <fstream>

#include "CpuInference.h"
#include "FileUtils.h"
#include "GpuInference.h"
#include "MathUtils.h"

std::ofstream ofstream("output.csv");

std::pair<std::string, std::string> parse_argument(const std::string& arg)
{
    if (arg.size() < 4 || arg.substr(0, 2) != "--")
    {
        return {"", ""}; // Invalid format
    }

    const size_t split_position = arg.find('=');
    size_t value_offset = 0;
    if (split_position == std::string::npos || split_position <= 2)
    {
        return {"", ""}; // Invalid format
    }
    const size_t start_quote = arg.find('"', split_position + 1);
    if (start_quote == split_position + 1)
    {
        value_offset = 1;
    }

    std::string name = arg.substr(2, split_position - 2);
    std::string value = arg.substr(split_position + 1 + value_offset, arg.size() - 1 - value_offset);
    return {name, value};
}

std::map<std::string, std::string> get_args(const int argv, char* argc[])
{
    std::map<std::string, std::string> arguments;
    for (int i = 1; i < argv; i++)
    {
        const auto arg = parse_argument(argc[i]);
        if (!arg.first.empty())
        {
            arguments[arg.first] = arg.second;
        }
    }
    return arguments;
}

std::vector<cv::Mat> load_images(const std::string& data_root)
{
    const std::vector<std::string> image_paths = find_all_images(data_root);
    std::vector<cv::Mat> images;
    images.reserve(image_paths.size());
    for (const auto& image_path : image_paths)
    {
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

void test_parallel(const CpuInference& inference, const std::vector<cv::Mat>& mats, const bool log = true)
{
    const auto output_mul_tensors = inference.predict_all(mats, MODEL_OUTPUT_CLASS);
    if (log)
    {
        ofstream << "Offset;Duration;Class\n";
        for (const auto& out_tensor : output_mul_tensors.out_tensors)
        {
            ofstream << out_tensor.offset_milliseconds << ";" << out_tensor.milliseconds << ";" << argmax(
                out_tensor.predictions) << ";\n";
        }
    }
    std::cout << "Parallel: " << output_mul_tensors.milliseconds << "ms!\n";
}

void test_sequential(const CpuInference& inference, const std::vector<cv::Mat>& mats)
{
    float t = 0;
    for (const auto& mat : mats)
    {
        const auto output_tensor = inference.predict(mat, MODEL_OUTPUT_CLASS);
        t += output_tensor.milliseconds;
    }
    std::cout << "Sequential: " << t << "ms!\n";
}

void test(const std::string& model_path, const std::string& data_root)
{
    if (model_path.empty() || data_root.empty())
    {
        throw std::runtime_error("Arguments error.");
    }
    const auto images = load_images(data_root);
    const auto inference = CpuInference(model_path);
    test_parallel(inference, images);
    test_sequential(inference, images);
}

#endif

int main(const int argc, char* argv[])
{
    const auto arguments = get_args(argc, argv);
    const auto model = arguments.at("model");
    const auto data = arguments.at("data_root");
    test(model, data);
}
