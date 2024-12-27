#if ACCELERATE_GPU
#include <future>
#include "ImageUtils.h"
#include "GpuInference.h"
#include "Logger.h"
#include <Windows.h>

GpuInference::GpuInference(const std::string &model_path,
                           const nvinfer1::ILogger::Severity &severity = nvinfer1::ILogger::Severity::kINFO) {
    this->program_logger_ = new ProgramLogger(severity);
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

OutParTensors GpuInference::predict_all(const std::vector<cv::Mat> &images, const std::size_t out_class) const {
    OutParTensors out_par_tensors;
    cudaEvent_t start, end;
    cudaError_t err;

    typedef struct TmpDataHolder {
        cudaEvent_t current_start{}, current_end{};
        cudaStream_t stream{};
        std::array<void *, 2> buffers{};
        OutParTensor tensor{};
        nvinfer1::IExecutionContext *context;

        explicit TmpDataHolder(nvinfer1::IExecutionContext *context) {
            this->context = context;
        }
    } TmpDataHolder;
    std::vector<TmpDataHolder> tmp_data;

    err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create start event");
    }
    err = cudaEventCreate(&end);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create end event");
    }
    err = cudaEventRecord(start, nullptr);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to record start event");
    }

    const auto complete_threads = [start](std::vector<TmpDataHolder> &completables,
                                          std::vector<OutParTensor> &tensors) {
        cudaError_t err;
        for (auto &completable: completables) {
            err = cudaStreamSynchronize(completable.stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to synchronize stream.");
            }

            err = cudaEventRecord(completable.current_end, completable.stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to record end event");
            }
            completable.tensor.milliseconds = compute_milliseconds(completable.current_start, completable.current_end);
            completable.tensor.offset_milliseconds = compute_milliseconds(start, completable.current_start);
            err = cudaEventDestroy(completable.current_start);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to destroy start event");
            }
            err = cudaEventDestroy(completable.current_end);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to destroy end event");
            }
            err = cudaStreamDestroy(completable.stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to destroy stream");
            }
            delete completable.context;
            tensors.push_back(completable.tensor);
        }
    };

    for (const auto &image: images) {
        if (tmp_data.size() >= get_SmCores()) {
            complete_threads(tmp_data, out_par_tensors.out_tensors);
            tmp_data.clear();
        }
        const auto context = this->engine_->createExecutionContext();
        TmpDataHolder tmp(context);

        err = cudaStreamCreate(&tmp.stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create stream");
        }
        err = cudaEventCreate(&tmp.current_start);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create item start event");
        }
        err = cudaEventCreate(&tmp.current_end);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create item end event");
        }
        err = cudaEventRecord(tmp.current_start, tmp.stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to record start event");
        }
        const auto image_data = process_image(image);
        err = cudaMallocAsync(&tmp.buffers[input_index], image_data.size() * sizeof(float),
                              tmp.stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate input buffer");
        }
        err = cudaMallocAsync(&tmp.buffers[output_index], out_class * sizeof(float), tmp.stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate output buffer");
        }
        err = cudaMemcpyAsync(tmp.buffers[input_index], image_data.data(),
                              image_data.size() * sizeof(float),
                              cudaMemcpyHostToDevice, tmp.stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy input buffer");
        }

        if (!context->setInputTensorAddress("args_0", tmp.buffers[input_index])) {
            throw std::runtime_error("Failed to set input tensor address.");
        }
        if (!context->
            setOutputTensorAddress("dense_1", tmp.buffers[output_index])) {
            throw std::runtime_error("Failed to set output tensor address.");
        }
        if (!context->enqueueV3(tmp.stream)) {
            throw std::runtime_error("Failed to enqueue upon stream on context.");
        }
        tmp.tensor.predictions.resize(out_class);
        err = cudaMemcpyAsync(tmp.tensor.predictions.data(), tmp.buffers[output_index], out_class * sizeof(float),
                              cudaMemcpyDeviceToHost, tmp.stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy back to output buffer");
        }
        tmp_data.push_back(tmp);
    }
    complete_threads(tmp_data, out_par_tensors.out_tensors);
    cudaEventRecord(end, nullptr);
    out_par_tensors.milliseconds = compute_milliseconds(start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return out_par_tensors;
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

long GpuInference::get_SmCores() {
    int deviceId;
    cudaDeviceProp deviceProps{};
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&deviceProps, deviceId);
    return ConvertSMVer2Cores(deviceProps.major, deviceProps.minor) * deviceProps.multiProcessorCount;
}

int GpuInference::ConvertSMVer2Cores(const int major, const int minor) {
    typedef struct {
        int SM; // SM Major version
        int Cores; // Number of operational cores
    } s_mto_cores;

    const s_mto_cores n_gpu_arch_cores_per_sm[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60, 64},
        {0x61, 128},
        {0x62, 128},
        {0x70, 64},
        {0x72, 64},
        {0x75, 64},
        {-1, -1}
    };

    int index = 0;
    while (n_gpu_arch_cores_per_sm[index].SM != -1) {
        if (n_gpu_arch_cores_per_sm[index].SM == ((major << 4) + minor)) {
            return n_gpu_arch_cores_per_sm[index].Cores;
        }

        index++;
    }
    return n_gpu_arch_cores_per_sm[index - 1].Cores;
}

#endif
