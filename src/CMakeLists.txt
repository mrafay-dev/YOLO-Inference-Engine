#header only file, and makes it avaiable to be included by other files
add_library(yolo_utils INTERFACE)
target_include_directories(yolo_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(yolov8n_camera yolov8n_camera.cpp)

#links all libs
target_link_libraries(yolov8n_camera PRIVATE
    onnxruntime         #model inference
    opencv_core         #core funcs
    opencv_imgproc      #resizing etc
    opencv_imgcodecs    #read/write img
    opencv_highgui      #displaying windows
    opencv_dnn          #deep neural network support
    yolo_utils          #util lib
    opencv_videoio      #video in/out
    pthread             #multithreading
)

add_subdirectory(utils)