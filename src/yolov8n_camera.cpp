#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <csignal>
#include <string>
#include <optional>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "utils/types.hpp"
#include "utils/tensor_buffer.hpp"
#include "utils/lock_free_queue.hpp"
#include "utils/image_processor.hpp"
#include "utils/inference_engine.hpp"
#include <utils/metrics.hpp>

// logging macro
#define LOG_INFO(msg) std::cout << "[INFO] " << msg << std::endl;
#define LOG_DEBUG(msg) std::cout << "[DEBUG] " << msg << std::endl;
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl;
#define LOG_WARN(msg) std::cerr << "[WARN] " << msg << std::endl;

std::atomic<bool> running(true);
yolo::LatencyTracker e2e_tracker;

void signalHandler(int signal) {
    LOG_INFO("Received signal: " + std::to_string(signal) + " , shutting down...");
    running = false;
}

void producerThread(cv::VideoCapture& cap,
                yolo::LockFreeQueue<cv::Mat>& frame_queue,
                std::atomic<bool>& is_running) {

    //LOG_INFO("Producer thread started");

    int empty_frame_retries = 0;
    const int max_retries = 30;
    
    while(is_running) {
        auto capture_start = std::chrono::high_resolution_clock::now();
        
        cv::Mat frame;
        cap >> frame;

        if(frame.empty()) {
            ++empty_frame_retries;
            if (empty_frame_retries > max_retries) {
                LOG_WARN("Producer: Empty frame captured");
                is_running = false;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        empty_frame_retries = 0;

        auto capture_end = std::chrono::high_resolution_clock::now();
        auto capture_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            capture_end - capture_start).count();
        
        if (capture_ms > 10) {
            //LOG_DEBUG("Producer: Capture took " + std::to_string(capture_ms) + "ms");
        }

        cv::Mat frame_clone = frame.clone();
        
        // enqueue, drop if full
        auto enqueue_start = std::chrono::high_resolution_clock::now();
        if (!frame_queue.enqueue(std::move(frame_clone))) {
            //LOG_DEBUG("Producer: Queue full, dropping frame");
        }
        auto enqueue_end = std::chrono::high_resolution_clock::now();
        auto enqueue_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            enqueue_end - enqueue_start).count();
        
        // Log queue size every 30 frames
        static int producer_counter = 0;
        if (++producer_counter % 30 == 0) {
            //LOG_DEBUG("Producer: Queue size=" + std::to_string(frame_queue.size()) + 
            //         ", Enqueue time=" + std::to_string(enqueue_ms) + "us");
        }
    }
    //LOG_INFO("Producer: Thread stopped");
}

// consumer thread
void consumerThread(yolo::InferenceEngine& engine,
                    yolo::TensorBuffer& buffer,
                    const yolo::ModelConfig& config,
                    yolo::LockFreeQueue<cv::Mat>& frame_queue,
                    yolo::LockFreeQueue<yolo::FrameWithResults>& result_queue,
                    std::atomic<bool>& is_running) {

    //LOG_INFO("Consumer: Thread started");
    static uint64_t frame_counter = 0;
    static uint64_t frame_skip_counter = 0;

    while (is_running) {
        auto total_start = std::chrono::steady_clock::now();
        
        // Dequeue frame
        auto dequeue_start = std::chrono::high_resolution_clock::now();
        auto frame_opt = frame_queue.dequeue();
        auto dequeue_end = std::chrono::high_resolution_clock::now();
        auto dequeue_us = std::chrono::duration_cast<std::chrono::microseconds>(
            dequeue_end - dequeue_start).count();
        
        if(!frame_opt.has_value()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Skip frames
        if (++frame_skip_counter % 3 != 0) {
            // Still log dequeue time occasionally
            if (frame_skip_counter % 30 == 0) {
                //LOG_DEBUG("Consumer: Skipped frame, dequeue time=" + std::to_string(dequeue_us) + "us");
            }
            continue;
        }

        cv::Mat frame = std::move(frame_opt.value());
        ++frame_counter;

        // Preprocess timing
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        yolo::ProcessedImage processed = yolo::ImageProcessor::preprocessImage(
            frame, 
            frame_counter,
            buffer
        );
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        auto preprocess_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            preprocess_end - preprocess_start).count();
        auto preprocess_us = std::chrono::duration_cast<std::chrono::microseconds>(
            preprocess_end - preprocess_start).count();

        // Run inference
        try {
            auto inference_start = std::chrono::high_resolution_clock::now();
            float* output_data = engine.infer(buffer);
            auto inference_end = std::chrono::high_resolution_clock::now();
            auto inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                inference_end - inference_start).count();
            auto inference_us = std::chrono::duration_cast<std::chrono::microseconds>(
                inference_end - inference_start).count();

            // Postprocess timing
            auto postprocess_start = std::chrono::high_resolution_clock::now();
            yolo::DetectionResults results = yolo::ImageProcessor::postprocessDetections(
                output_data,
                processed,
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()
                ).count(),
                config
            );
            auto postprocess_end = std::chrono::high_resolution_clock::now();
            auto postprocess_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                postprocess_end - postprocess_start).count();
            auto postprocess_us = std::chrono::duration_cast<std::chrono::microseconds>(
                postprocess_end - postprocess_start).count();

            // Enqueue result
            yolo::FrameWithResults frame_with_results;
            frame_with_results.frame = std::move(frame);
            frame_with_results.results = std::move(results);

            auto enqueue_start = std::chrono::high_resolution_clock::now();
            if (!result_queue.enqueue(std::move(frame_with_results))) {
                //LOG_DEBUG("Consumer: Result queue full, dropping results");
            }
            auto enqueue_end = std::chrono::high_resolution_clock::now();
            auto enqueue_us = std::chrono::duration_cast<std::chrono::microseconds>(
                enqueue_end - enqueue_start).count();

            // Total time
            auto total_end = std::chrono::steady_clock::now();
            auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                total_end - total_start).count();
            auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(
                total_end - total_start).count();

            e2e_tracker.record(total_us);
            // Detailed logging every frame (or every N frames to reduce spam)
            // if (frame_counter % 5 == 0) {
            //     LOG_INFO("=== Frame " + std::to_string(frame_counter) + " Timing ===");
            //     LOG_INFO("  Dequeue:     " + std::to_string(dequeue_us) + " us");
            //     LOG_INFO("  Preprocess:  " + std::to_string(preprocess_ms) + " ms (" + 
            //             std::to_string(preprocess_us) + " us)");
            //     LOG_INFO("  Inference:   " + std::to_string(inference_ms) + " ms (" + 
            //             std::to_string(inference_us) + " us)");
            //     LOG_INFO("  Postprocess: " + std::to_string(postprocess_ms) + " ms (" + 
            //             std::to_string(postprocess_us) + " us)");
            //     LOG_INFO("  Enqueue:     " + std::to_string(enqueue_us) + " us");
            //     LOG_INFO("  TOTAL:       " + std::to_string(total_ms) + " ms");
            //     LOG_INFO("  Detections:  " + std::to_string(frame_with_results.results.detections.size()));
            //     LOG_INFO("  Queue sizes: Frame=" + std::to_string(frame_queue.size()) + 
            //             ", Result=" + std::to_string(result_queue.size()));
            // }
            
        } catch (const std::exception& e) {
            //LOG_ERROR("Consumer: Inference failed: " + std::string(e.what()));
        }
    }

    //LOG_INFO("Consumer: Thread stopped");
}

//display thread function - shows results
void displayThread(yolo::LockFreeQueue<yolo::FrameWithResults>& result_queue,
                    std::atomic<bool>& is_running) {

    //LOG_INFO("Display thread started");
    cv::namedWindow("Real-time detection", cv::WINDOW_NORMAL);

    //FPS Tracking
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (is_running) {
        auto display_start = std::chrono::high_resolution_clock::now();
        
        auto results_opt = result_queue.dequeue();
        if (!results_opt.has_value()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto dequeue_end = std::chrono::high_resolution_clock::now();
        auto dequeue_us = std::chrono::duration_cast<std::chrono::microseconds>(
            dequeue_end - display_start).count();

        yolo::FrameWithResults frame_with_results = std::move(results_opt.value());
        cv::Mat frame = std::move(frame_with_results.frame);
        yolo::DetectionResults& results = frame_with_results.results;
        ++frame_count;

        // FPS counter
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time);
        
        if(elapsed.count() > 1000){  //fps updates every sec
            float current_fps = frame_count * 1000.0f / elapsed.count();
            //LOG_DEBUG("Display FPS: " + std::to_string(current_fps));
            start_time = current_time;
            frame_count = 0;
        }

        // Drawing timing
        auto draw_start = std::chrono::high_resolution_clock::now();
        
        for (const auto& detection : results.detections) {
            cv::rectangle(frame, detection.box, cv::Scalar(0, 255, 0), 2);

            std::string label = detection.class_name + " " + 
                               std::to_string(detection.confidence).substr(0, 4);
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                                0.5, 1, &baseline);
            
            cv::rectangle(frame, 
                         cv::Point(detection.box.x, detection.box.y - text_size.height - baseline),
                         cv::Point(detection.box.x + text_size.width, detection.box.y),
                         cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(frame, label, 
                       cv::Point(detection.box.x, detection.box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        auto draw_end = std::chrono::high_resolution_clock::now();
        auto draw_us = std::chrono::duration_cast<std::chrono::microseconds>(
            draw_end - draw_start).count();

        // Show frame
        auto imshow_start = std::chrono::high_resolution_clock::now();
        cv::imshow("Real-time detection", frame);
        auto imshow_end = std::chrono::high_resolution_clock::now();
        auto imshow_us = std::chrono::duration_cast<std::chrono::microseconds>(
            imshow_end - imshow_start).count();

        // Check for key press
        char key = cv::waitKey(1);
        if(key == 'q' || key == 27) {
            //LOG_INFO("Display: Quitting...");
            is_running = false;
            break;
        }

        // Log display timing occasionally
        if (frame_with_results.results.frame_id % 30 == 0) {
            //LOG_DEBUG("Display: Dequeue=" + std::to_string(dequeue_us) + 
            //         "us, Draw=" + std::to_string(draw_us) + 
            //         "us, Imshow=" + std::to_string(imshow_us) + "us");
        }
    }

    cv::destroyAllWindows();
    //LOG_INFO("Display thread stopped");
}

int main(int argc, char* argv[]){

    //signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    //parse cmd line args
    int camera_idx = 0;
    if (argc >= 2) {
        camera_idx = std::stoi(argv[1]);
    }

    const std::string model_path = "models/yolov8n.onnx";

    //model configs
    const int INPUT_SIZE = 640;
    const int NUM_CLASSES = 80;
    const int NUM_DETECTIONS = 8400;
    const float CONFIDENCE_THRESHOLD = 0.25f;
    const float NMS_THRESHOLD = 0.45f;

    yolo::ModelConfig config;
    config.input_size = INPUT_SIZE;
    config.num_classes = NUM_CLASSES;
    config.num_detections = NUM_DETECTIONS;
    config.confidence_threshold = CONFIDENCE_THRESHOLD;
    config.nms_threshold = NMS_THRESHOLD;
    
    // opening camera
    LOG_INFO("Opening camera at index" + std::to_string(camera_idx));
    cv::VideoCapture cap(camera_idx);
    //cv::VideoCapture cap(camera_idx, cv::CAP_V4L2);

    // setting up env
    if (!cap.isOpened()){
        LOG_ERROR("Couldn't open camera!");
        return -1;
    }

    // get camera properties
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    LOG_INFO("Camera opened: " + std::to_string(frame_width) +
            " x " + std::to_string(frame_height) + 
            " at " + std::to_string(fps) + " FPS");
            
    //init. inference engine
    std::unique_ptr<yolo::InferenceEngine> inference_engine;
    try{
        inference_engine = std::make_unique<yolo::InferenceEngine>(
            model_path, 
            config);
        LOG_INFO("Inference engine initialised successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to initalised inference engine: " + std::string(e.what()));
        cap.release();
        return -1;
    }

    //create tensor buffer
    yolo::TensorBuffer buffer(config);
    LOG_INFO("Tensor buffer created with " + std::to_string(buffer.size()) + 
            " elements");
    
    //queues for producer-consumer pattern
    yolo::LockFreeQueue<cv::Mat> frame_queue(5);
    yolo::LockFreeQueue<yolo::FrameWithResults> result_queue(5);
    LOG_INFO("Queues created : Frame queue capacity=" + std::to_string(frame_queue.capacity()) + 
            ", Result queue capacity=" + std::to_string(result_queue.capacity()));
    
    //start threads
    std::thread producer (producerThread, 
                            std::ref(cap), 
                            std::ref(frame_queue), 
                            std::ref(running));
    std::thread consumer (consumerThread, 
                            std::ref(*inference_engine), 
                            std::ref(buffer), 
                            std::ref(config),
                            std::ref(frame_queue),
                            std::ref(result_queue),
                            std::ref(running));
    std::thread display (displayThread, 
                        std::ref(result_queue), 
                        std::ref(running));
    
    LOG_INFO("All threads started. 'q' or ESC to quit");

    if (producer.joinable()) producer.join();
    if (consumer.joinable()) consumer.join();
    if (display.joinable()) display.join();

    cap.release();
    cv::destroyAllWindows();

    e2e_tracker.printStats("End-to-End pipeline");

    LOG_INFO("Program terminated successfully");
    return 0;
}