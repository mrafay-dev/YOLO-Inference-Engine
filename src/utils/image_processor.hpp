#ifndef YOLO_IMAGE_PROCESSOR_HPP
#define YOLO_IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <algorithm>
#include <string>
#include "types.hpp"
#include "tensor_buffer.hpp"

namespace yolo {

//structure for processed images
struct ProcessedImage {
    uint64_t frame_id;
    int original_width;
    int original_height;
    float x_scale;
    float y_scale;
    int64_t capture_time_us;
};

class ImageProcessor {
public:
    static ProcessedImage preprocessImage(
        const cv::Mat& frame,
        uint64_t frame_id,
        TensorBuffer& buffer
    ) {
        ProcessedImage result;
        result.frame_id = frame_id;
        result.original_width = frame.cols;
        result.original_height = frame.rows;
        result.x_scale = static_cast<float>(frame.cols) / buffer.getInputSize();
        result.y_scale = static_cast<float>(frame.rows) / buffer.getInputSize();
        result.capture_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();

        cv::Mat blob = cv::dnn::blobFromImage(
            frame, 
            1.0 / 255.0,
            cv::Size(buffer.getInputSize(), buffer.getInputSize()),
            cv::Scalar(0, 0, 0),
            true,       //swap RB (BGR to RGB)
            false,      //crop
            CV_32F
        );

        std::memcpy(buffer.getData(), blob.ptr<float>(), buffer.size() * sizeof(float));

        return result;
    }


//post process inference results
static DetectionResults postprocessDetections(
    const float* output_data,
    const ProcessedImage& processed,
    int64_t inference_start_time_us,
    const ModelConfig& config
) {
    DetectionResults results;
    results.frame_id = processed.frame_id;
    results.capture_time_us = processed.capture_time_us;
    results.total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count() - inference_start_time_us;

    const int num_detections = config.num_detections;
    const int num_classes = config.num_classes;

    static thread_local std::vector<cv::Rect> boxes;
    static thread_local std::vector<float> scores;
    static thread_local std::vector<int> class_ids;

    boxes.clear();
    scores.clear();
    class_ids.clear();

    boxes.reserve(100);
    scores.reserve(100);
    class_ids.reserve(100);
    
    for (int i = 0; i < num_detections; ++i) {
        //first 4 are xyzh
        //next 80 are confidence scores
        //find best class
        float best_score = 0.0f;
        int best_class = -1;

        for (int c = 0; c < num_classes; ++c) {
            // access data like 2d array [84, 8400]
            // 
            float score = output_data[(4+c) * num_detections + i];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < config.confidence_threshold) continue;
        
        // getting coords in centre format
        float cx = output_data[0 * num_detections + i];
        float cy = output_data[1 * num_detections + i];
        float w = output_data[2 * num_detections + i];
        float h = output_data[3 * num_detections + i];

        // convert to og img coords
        float left = (cx - w * 0.5f) * processed.x_scale; 
        float top  = (cy - h * 0.5f) * processed.y_scale;
        float width  = w * processed.x_scale;
        float height = h * processed.y_scale;

        // clip to img boundaries
        left = std::max(0.0f, left);
        top = std::max(0.0f, top);
        width = std::min(width, static_cast<float>(processed.original_width - left));
        height = std::min(height, static_cast<float>(processed.original_height - top));

        boxes.emplace_back(
            static_cast<int>(left),
            static_cast<int>(top),
            static_cast<int>(width),
            static_cast<int>(height)
        );
        scores.push_back(best_score);
        class_ids.push_back(best_class);
    }

    // apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(
        boxes,
        scores,
        config.confidence_threshold,
        config.nms_threshold,
        indices
    );

    // draw detections
    for (int idx : indices) {
        Detection det;
        det.box = boxes[idx];
        det.confidence = scores[idx];
        det.class_id = class_ids[idx];
        det.class_name = COCO_CLASSES[class_ids[idx]];
        results.detections.push_back(det);
    }

    results.inference_time_us = results.total_time_us;

    return results;
       
    }
};

}

#endif
