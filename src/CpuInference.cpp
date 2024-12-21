#include "ImageUtils.h"
#if ACCELERATE_CPU
#include "CpuInference.h"


CpuInference::CpuInference(const std::string& model_path): model_(fdeep::load_model(model_path))
{
}

CpuInference::~CpuInference() = default;

OutTensor CpuInference::predict(const cv::Mat& image, std::size_t out_class) const
{
    OutTensor out_tensor;
    const std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::high_resolution_clock::now();
    auto input_tensor = to_tensor(image);
    const auto output_tensors = this->model_.predict({input_tensor});
    const std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<float, std::milli> milliseconds(end - start);
    out_tensor.milliseconds = milliseconds.count();
    out_tensor.predictions = output_tensors[0].to_vector();
    return out_tensor;
}

OutMulTensors CpuInference::predict_all(const std::vector<cv::Mat>& images, std::size_t out_class) const
{
    OutMulTensors out_mul_tensors;
    std::vector<OutMulTensor> out_mul_tensor_vector;
    out_mul_tensor_vector.reserve(images.size());
    std::vector<std::shared_future<OutMulTensor>> futures;
    const auto start = std::chrono::high_resolution_clock::now();
    for (const auto& image : images)
    {
        auto future = std::async(std::launch::async, [&]()
        {
            OutMulTensor out_mul_tensor;
            const auto current_start = std::chrono::high_resolution_clock::now();
            auto vector = to_tensor(image);
            const auto out_tensor = this->model_.predict({vector});
            const auto current_end = std::chrono::high_resolution_clock::now();
            out_mul_tensor.offset_milliseconds = static_cast<float>((current_start - start).count()) / 1000.0;
            out_mul_tensor.milliseconds = static_cast<float>((current_end - current_start).count()) / 1000.0;
            out_mul_tensor.predictions = out_tensor[0].to_vector();
            return out_mul_tensor;
        });
        futures.push_back(future.share());
    }
    for (const auto& future : futures)
    {
        out_mul_tensor_vector.push_back(future.get());
    }
    const auto end = std::chrono::high_resolution_clock::now();
    out_mul_tensors.milliseconds = static_cast<float>((end - start).count()) / 1000.0;
    out_mul_tensors.out_tensors = out_mul_tensor_vector;
    return out_mul_tensors;
}

fdeep::tensor CpuInference::to_tensor(const cv::Mat& image)
{
    const auto converted_image = modify_image(image);
    const auto float_array = to_vector_input(converted_image);
    auto tensor = fdeep::tensor(fdeep::tensor_shape{28, 28, 1}, float_array);
    return tensor;
}

#endif
