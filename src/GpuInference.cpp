#if ACCELERATE_GPU
#include <future>
#include "ImageUtils.h"
#include "GpuInference.h"
#include "Logger.h"
#include <Windows.h>

GpuInference::GpuInference(const std::string &model_path) {
    this->program_logger_ = new ProgramLogger();
    this->engine_ = build_engine(model_path, *this->program_logger_);
}

GpuInference::~GpuInference() {
    delete this->engine_;
    delete this->program_logger_;
}

OutTensor GpuInference::predict(const cv::Mat &image, const std::size_t out_class) const {
    cudaEvent_t end, start;
    cudaEventCreate(&end);
    cudaEventCreate(&start);
    cudaEventRecord(start, nullptr);

    OutTensor out_tensor;
    const auto context = this->engine_->createExecutionContext();

    std::array<void *, 2> buffers{};

    const auto image_data = process_image(image);
    init_cuda_buffers(buffers, image_data, out_class);
    if (!context->executeV2(buffers.data())) {
        throw std::runtime_error("Failed to enqueue context");
    }

    std::vector<float> predictions(out_class);
    copy_to_host(predictions, buffers, out_class);
    destroy_cuda_buffers(buffers);
    delete context;

    cudaEventRecord(end, nullptr);
    out_tensor.milliseconds = compute_milliseconds(start, end);

    cudaEventDestroy(end);
    cudaEventDestroy(start);
    return out_tensor;
}

OutMulTensors GpuInference::predict_all(const std::vector<cv::Mat> &images, const std::size_t out_class) const {
    OutMulTensors out_mul_tensors;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, nullptr);
    std::vector<std::shared_future<OutMulTensor> > futures;
    for (const auto &image: images) {
        auto future = std::async(std::launch::async, [this, image, out_class, start] {
            OutMulTensor out_mul_tensor;
            cudaEvent_t iteration_start;
            cudaEventCreate(&iteration_start);
            cudaEventRecord(iteration_start, nullptr);
            const auto out_tensor = this->predict(image, out_class);
            const auto offset = compute_milliseconds(start, iteration_start);
            out_mul_tensor.milliseconds = out_tensor.milliseconds;
            out_mul_tensor.offset_milliseconds = offset;
            out_mul_tensor.predictions = out_tensor.predictions;
            cudaEventDestroy(iteration_start);
            return out_mul_tensor;
        });
        futures.push_back(future.share());
    }
    for (const auto &shared_future: futures) {
        out_mul_tensors.out_tensors.push_back(shared_future.get());
    }
    cudaEventRecord(end, nullptr);
    out_mul_tensors.milliseconds = compute_milliseconds(start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return out_mul_tensors;
}

nvinfer1::ICudaEngine *GpuInference::build_engine(const std::string &onnx_model_path, ProgramLogger &logger) {
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    constexpr auto explicitBatch = 1U << 0;
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);

    const auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnx_model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX model.");
    }
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30); // 1GB WORKSPACE
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr) {
        throw std::runtime_error("Failed to build engine.");
    }
    delete config;
    delete parser;
    delete network;
    return engine;
}


void GpuInference::init_cuda_buffers(std::array<void *, 2> &buffers, const std::vector<float> &image_data,
                                     const std::size_t out_class) {
    cudaMalloc(&buffers[input_index], image_data.size() * sizeof(float));
    cudaMemcpy(buffers[input_index], image_data.data(), image_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&buffers[output_index], out_class * sizeof(float));
}

void GpuInference::destroy_cuda_buffers(const std::array<void *, 2> &buffers) {
    cudaFree(buffers[input_index]);
    cudaFree(buffers[output_index]);
}

void GpuInference::copy_to_host(std::vector<float> &predictions, const std::array<void *, 2> &buffers,
                                const std::size_t out_class) {
    cudaMemcpy(predictions.data(), buffers[output_index], out_class * sizeof(float), cudaMemcpyDeviceToHost);
}

std::vector<float> GpuInference::process_image(const cv::Mat &image) {
    return to_vector_input(modify_image(image));
}

float GpuInference::compute_milliseconds(const cudaEvent_t &start, const cudaEvent_t &end) {
    float milliseconds = -1;
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&milliseconds, start, end);
    return milliseconds;
}

#endif
