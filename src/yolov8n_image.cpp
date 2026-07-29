#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

//COCO class names
static const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

// logging macro
#define LOG_INFO(msg) std::cout << "[INFO] " << msg << std::endl;
#define LOG_DEBUG(msg) std::cout << "[DEBUG] " << msg << std::endl;
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl;

int main(int argc, char* argv[]){

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>\n" << std::endl;
        return 1; 
    }

    const std::string image_path = argv[1];
    const std::string model_path = "models/yolov8n.onnx";

    const int INPUT_SIZE = 640;
    const int NUM_CLASSES = 80;
    const int NUM_DETECTIONS = 8400;
    const float CONFIDENCE_THRESHOLD = 0.25f;
    const float NMS_THRESHOLD = 0.45f;
    
    // loading image
    LOG_INFO("Loading image" + image_path);

    // setting up env
    // preparing inputs
    // 1 batch, 3 colours RGB, 640 x 640 height & width
    cv::Mat image = cv::imread(image_path);
    if (image.empty()){
        std::cerr << "Could not load image" << std::endl;
        return -1;
    }
    
    int original_width = image.cols;
    int original_height = image.rows;
    LOG_INFO("Original size: " + std::to_string(original_width) +
            " x " + std::to_string(original_height));
    
    // setting up ONNX runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YoloEnv");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    // preprocess img
    LOG_DEBUG("Preprocessing...");
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(INPUT_SIZE, INPUT_SIZE));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0/255.0);

    // create input tensor -> [1, 3, 640, 640]
    std::vector<float> input_tensor(1 * 3 * INPUT_SIZE * INPUT_SIZE);
    for (int y = 0; y < INPUT_SIZE; ++y) {
        for (int x = 0; x < INPUT_SIZE; ++x) {
            cv::Vec3f pixel = resized.at<cv::Vec3f>(y, x);
            const int IDX = y * INPUT_SIZE + x;
            input_tensor[0 * INPUT_SIZE * INPUT_SIZE + IDX] = pixel[0];
            input_tensor[1 * INPUT_SIZE * INPUT_SIZE + IDX] = pixel[1];
            input_tensor[2 * INPUT_SIZE * INPUT_SIZE + IDX] = pixel[2];
        }
    }

    // running inference
    LOG_DEBUG("Running inference...");
    auto start = std::chrono::high_resolution_clock::now();

    // telling onnx shape of data - [batch, channels, height, width]
    std::vector<int64_t> input_shape {1, 3, INPUT_SIZE, INPUT_SIZE};
    
    //wrapping raw data into onnx objects
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size()
    );

    // telling onnx the names of inputs and outputs
    // pointer to const c style arrays. length derived by compiler
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};

    // run model
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor_ort, 1, //2 inputs
        output_names, 1 // 1 ouput
    );

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOG_INFO("Inference time: " + std::to_string(duration.count()) + " ms");

    //post process
    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    float x_scale = static_cast<float>(original_width) / INPUT_SIZE;
    float y_scale = static_cast<float>(original_height) / INPUT_SIZE;

    //output shape = [1, 84, 8400]
    // 84 => 4 (x, y, x, h) + 80(classes)

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    
    for (size_t i = 0; i < NUM_DETECTIONS; ++i) {
        //first 4 are xyzh
        //next 80 are confidence scores
        //find best class

        float best_score = 0.0f;
        int best_class = -1;

        for (int c = 0; c < NUM_CLASSES; ++c) {
            // access data like 2d array [84, 8400]
            // 
            float score = output_data[(4+c) * NUM_DETECTIONS + i];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < CONFIDENCE_THRESHOLD) continue;
        
        // getting coords in box format
        float x = output_data[0 * NUM_DETECTIONS + i];
        float y = output_data[1 * NUM_DETECTIONS + i];
        float w = output_data[2 * NUM_DETECTIONS + i];
        float h = output_data[3 * NUM_DETECTIONS + i];

        // convert to og img coords
        float left = (x - w * 0.5f) * x_scale;
        float top  = (y - h * 0.5f) * y_scale;
        float width  = w * x_scale;
        float height = h * y_scale;

        boxes.emplace_back(
            static_cast<int>(left),
            static_cast<int>(top),
            static_cast<int>(width),
            static_cast<int>(height)
        );
        scores.push_back(best_score);
        class_ids.push_back(best_class);
    }
    LOG_DEBUG("Detections before NMS: " + std::to_string(boxes.size()));

    // apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(
        boxes,
        scores,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD,
        indices
    );
    LOG_DEBUG("Detections after NMS: " + std::to_string(indices.size()));

    // draw scores
    for (int idx : indices) {
        // draw box
        cv::rectangle(
            image, 
            boxes[idx],
            cv::Scalar(0, 255, 0),
            2
        );

        // label with class name
        std::string label =
            COCO_CLASSES[class_ids[idx]] + " " + 
            cv::format("%.2f", scores[idx]);

        // draw label
        cv::putText(
            image,
            label,
            boxes[idx].tl(),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6, 
            cv::Scalar(0, 255, 0),
            2
        );

    }; 

    // save then display
    cv::imwrite("result.jpg", image);
    LOG_INFO("Result saved to results.jpg");

    cv::imshow("YOLOv8 Detection", image);
    cv::waitKey(0);
    
    return 0;
    
    
}

