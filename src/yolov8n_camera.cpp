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

    int camera_idx = 0;
    if (argc == 2) {
        camera_idx = std::stoi(argv[1]);
    }

    const std::string model_path = "models/yolov8n.onnx";

    const int INPUT_SIZE = 640;
    const int NUM_CLASSES = 80;
    const int NUM_DETECTIONS = 8400;
    const float CONFIDENCE_THRESHOLD = 0.25f;
    const float NMS_THRESHOLD = 0.45f;
    
    // opening camera
    LOG_INFO("Opening camera at index" + std::to_string(camera_idx));
    cv::VideoCapture cap(camera_idx);

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
            
    // setting up ONNX runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YoloEnv");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);
    
    //wrapping raw data into onnx objects
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // telling onnx the names of inputs and outputs
    // pointer to const c style arrays. length derived by compiler
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};
    
    // create input tensor -> [1, 3, 640, 640]
    std::vector<int64_t> input_shape{1, 3, INPUT_SIZE, INPUT_SIZE};

    cv::namedWindow("Real-time detection", cv::WINDOW_NORMAL);

    // main loop
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (true) {

        cv::Mat frame;
        cap >> frame; //capture new frame

        if(frame.empty()){
            LOG_ERROR("Failed to capture frame");
            break;
        }

        ++frame_count;
        int original_width = frame.cols;
        int original_height = frame.rows;

        // preprocess img
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(INPUT_SIZE, INPUT_SIZE));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32F, 1.0/255.0);
        
        // creating input tensor - [batch, channels, height, width]
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
        Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size()
        );

        // run model
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor_ort, 1, //2 inputs
            output_names, 1 // 1 ouput
        );

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

        // apply NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(
            boxes,
            scores,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD,
            indices
        );

        // draw detections
        cv::Mat display_frame = frame.clone(); //copy of it
        for (int idx : indices) {
            // draw box
            cv::rectangle(
                display_frame, 
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
                display_frame,
                label,
                boxes[idx].tl(),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6, 
                cv::Scalar(0, 255, 0),
                2
            );
    
        }; 

        // fps counter
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
        if(elapsed.count() > 1000){  //fps updates every sec
            float current_fps = frame_count / elapsed.count() * 1000.0f;
            LOG_DEBUG("FPS: " + std::to_string(current_fps));
            start_time = current_time;
            frame_count = 0;
        }
    
        cv::imshow("Real-time detection", display_frame);

        char key = cv::waitKey(1);
        if(key == 'q' || key == 27) {
            LOG_INFO("Quitting...");
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    
    return 0;    
}

