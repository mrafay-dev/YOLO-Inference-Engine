#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main(){
    // setting up env
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MinimalEnv");

    // loading model
    Ort::SessionOptions session_options;
    Ort::Session session(env, "adder.onnx", session_options);

    // preparing inputs
    std::vector<float> input_data_x {1.90f};
    std::vector<float> input_data_y {20.0f};

    // telling onnx shape of data
    std::vector<int64_t> input_shape {1}; //1d array, size 1

    //wrapping raw data into onnx objects
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor_x = Ort::Value::CreateTensor<float>(
        memory_info, input_data_x.data(), input_data_x.size(), input_shape.data(), input_shape.size()
    );

    Ort::Value input_tensor_y = Ort::Value::CreateTensor<float>(
        memory_info, input_data_y.data(), input_data_y.size(), input_shape.data(), input_shape.size()
    );    

    // telling onnx the names of inputs and outputs
    // pointer to const c style arrays. length derived by compiler
    const char* input_names[] = {"x", "y"};
    const char* output_names[] = {"sum"};

    // run model
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor_x, 2, //2 inputs
        output_names, 1 // 1 ouput
    );

    //get result out
    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    std::cout << "Result: " << output_data[0] <<std::endl;
    
    return 0;


}

