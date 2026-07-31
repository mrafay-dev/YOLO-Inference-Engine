#ifndef YOLO_INFERENCE_ENGINE_HPP
#define YOLO_INFERENCE_ENGINE_HPP

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include "types.hpp"
#include "tensor_buffer.hpp"

namespace yolo {

class InferenceEngine {
private:
    std::unique_ptr<Ort::Session> session;
    Ort::Env env;
    Ort::MemoryInfo memory_info;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    ModelConfig config;

public:
    InferenceEngine(const std::string& model_path, const ModelConfig& cfg)
        : env(ORT_LOGGING_LEVEL_WARNING, "YoloInference"),memory_info(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
            config(cfg) {

                Ort::SessionOptions session_options;
                session_options.SetIntraOpNumThreads(4);
                session_options.SetInterOpNumThreads(1);
                session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

                session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

                //prepare input/output name
                input_names = {"images"};
                output_names = {"output0"};
            }

        // run inference on preprocessed tensor
        Ort::Value infer(const TensorBuffer& buffer) {
            //create ONNX tensor from buffer
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(buffer.getData()),
                buffer.size(),
                buffer.getShape(),
                buffer.getShapeSize()
            );

            // run inference
            auto output_tensors = session->Run(
                Ort::RunOptions(nullptr),
                input_names.data(),
                &input_tensor,
                1, 
                output_names.data(),
                1
            );

            //store for later access
            return std::move(output_tensors[0]);

        }
};

}

#endif