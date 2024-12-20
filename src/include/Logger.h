#pragma once

#if ACCELERATE_GPU

#include <NvInfer.h>
#include <iostream>


class ProgramLogger final : public nvinfer1::ILogger {
    Severity severity_;

public:
    explicit ProgramLogger(const Severity &severity = Severity::kINFO) {
        this->severity_ = severity;
    }

    void log(Severity severity, const char *msg) noexcept override {
        if (static_cast<int32_t>(severity) <= static_cast<int32_t>(this->severity_)) {
            const std::vector<std::string> ERRORS = {
                "FATAL   ",
                "ERROR   ",
                "WARNING ",
                "INFO    ",
                "DEBUG   ",
            };
            std::cout << ERRORS[static_cast<int32_t>(severity)] << " : " << msg << std::endl;
        }
    }
};
#endif
