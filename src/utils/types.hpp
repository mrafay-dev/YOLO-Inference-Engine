//centralises data structures and configs

#ifndef YOLO_TYPES_HPP
#define YOLO_TYPES_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace yolo {

// for future upgrades
enum class ModelType { 
    YOLOV8N,
    YOLOV8M
};

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
    std::string class_name;
};

//results for 1 frame
struct DetectionResults {
    std::vector<Detection> detections;
    uint64_t frame_id;
    int64_t capture_time_us;
    int64_t inference_time_us;
    int total_time_us; // E2E time

    DetectionResults () 
        : frame_id(0), capture_time_us(0), inference_time_us(0), total_time_us(0){}
};

//model config
struct ModelConfig {
    ModelType type;
    std::string model_path;
    int input_size;
    int num_classes;
    int num_detections;
    float confidence_threshold;
    float nms_threshold;

    static ModelConfig getConfig(ModelType type) {
        ModelConfig config;
        config.type = type;
        config.input_size = 640;
        config.confidence_threshold = 0.25f;
        config.nms_threshold = 0.45f;
        config.num_detections = 8400;
        config.num_classes = 80;

        switch(type) {
            case ModelType::YOLOV8N:
                config.model_path = "models/yolov8n.onnx";
                break;
            case ModelType::YOLOV8M:
                config.model_path = "models/yolov8m.onnx";
                break;            
        }
        return config;
    }

    static std::string getModelName(ModelType type) {
        switch(type) {
            case ModelType::YOLOV8N: return "YOLOV8N";
            case ModelType::YOLOV8M: return "YOLOV8M";
            default: return "Unknown";
        }
    }
};

struct FrameWithResults {
    cv::Mat frame;              //original frame
    DetectionResults results;   //detection results
};

// COCO class names 
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

}  //end of namespace

#endif