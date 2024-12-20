#pragma once

#if ACCELERATE_GPU
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "AbstractInference.h"
#include "Logger.h"

class GpuInference final : public AbstractInference {
    static constexpr int input_index = 0;
    static constexpr int output_index = 1;

    nvinfer1::ICudaEngine *engine_;
    ProgramLogger *program_logger_;


    static void init_cuda_buffers(std::array<void *, 2> &buffers, const std::vector<float> &image_data,
                                  std::size_t out_class);

    static void destroy_cuda_buffers(const std::array<void *, 2> &buffers);

    static void copy_to_host(std::vector<float> &predictions, const std::array<void *, 2> &buffers,
                             std::size_t out_class);

    static std::vector<float> process_image(const cv::Mat &image);

    static float compute_milliseconds(const cudaEvent_t &start, const cudaEvent_t &end);

public:
    explicit GpuInference(const std::string &model_path);

    ~GpuInference() override;

    [[nodiscard]] OutTensor predict(const cv::Mat &image, std::size_t out_class) const override;

    [[nodiscard]] OutMulTensors predict_all(const std::vector<cv::Mat> &images, std::size_t out_class) const override;

    static nvinfer1::ICudaEngine *build_engine(const std::string &onnx_model_path, ProgramLogger &logger);
};

#endif
