#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;
namespace fs = std::filesystem;

// Logger for TensorRT
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

// Check CUDA error
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__ )
void check(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << result << " \"" << func << "\" \n";
        exit(EXIT_FAILURE);
    }
}

// Load Engine
ICudaEngine* loadEngine(const std::string& engineFile, IRuntime* runtime) {
    std::ifstream file(engineFile, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading engine file" << std::endl;
        return nullptr;
    }

    return runtime->deserializeCudaEngine(buffer.data(), size);
}

// Preprocessing parameters
const int NUM_FRAMES = 32;
const int SAMPLING_RATE = 5;
const int CROP_SIZE = 64;
const std::vector<float> MEAN = {0.45, 0.45, 0.45};
const std::vector<float> STD = {0.225, 0.225, 0.225};

// Preprocess images
void preprocess(const std::string& folderPath, float* hostInputBuffer) {
    std::vector<std::string> imageFiles;
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            imageFiles.push_back(entry.path().string());
        }
    }
    std::sort(imageFiles.begin(), imageFiles.end());

    if (imageFiles.empty()) {
        std::cerr << "No images found in " << folderPath << std::endl;
        exit(EXIT_FAILURE);
    }

    // Sampling logic
    std::vector<cv::Mat> frames;
    int totalFrames = imageFiles.size();
    int clipLength = NUM_FRAMES * SAMPLING_RATE;
    
    // Center crop in time if we have enough frames, otherwise loop or take what we have
    int startFrame = 0;
    if (totalFrames > clipLength) {
        startFrame = (totalFrames - clipLength) / 2;
    }

    for (int i = 0; i < NUM_FRAMES; ++i) {
        int frameIdx = startFrame + i * SAMPLING_RATE;
        if (frameIdx >= totalFrames) {
            frameIdx = frameIdx % totalFrames; // Simple loop padding
        }
        
        cv::Mat img = cv::imread(imageFiles[frameIdx]);
        if (img.empty()) {
            std::cerr << "Failed to read image: " << imageFiles[frameIdx] << std::endl;
            exit(EXIT_FAILURE);
        }

        // Resize and Center Crop
        // Strategy: Resize short side to CROP_SIZE, then center crop
        int h = img.rows;
        int w = img.cols;
        int newH, newW;
        if (h < w) {
            newH = CROP_SIZE;
            newW = w * CROP_SIZE / h;
        } else {
            newW = CROP_SIZE;
            newH = h * CROP_SIZE / w;
        }
        cv::resize(img, img, cv::Size(newW, newH));

        int startX = (newW - CROP_SIZE) / 2;
        int startY = (newH - CROP_SIZE) / 2;
        cv::Rect cropRegion(startX, startY, CROP_SIZE, CROP_SIZE);
        img = img(cropRegion);

        // Convert to float and normalize
        img.convertTo(img, CV_32FC3, 1.0 / 255.0);
        
        // HWC to CHW and Normalize
        // Input tensor shape: 1 x 3 x 32 x 64 x 64
        // Memory layout: [Batch, Channel, Time, Height, Width]
        // We iterate frames (Time), then Channels, then H, W
        
        // However, OpenCV is HWC (BGR). We need RGB.
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < CROP_SIZE; ++y) {
                for (int x = 0; x < CROP_SIZE; ++x) {
                    float pixel = img.at<cv::Vec3f>(y, x)[c];
                    // Normalize
                    pixel = (pixel - MEAN[c]) / STD[c];
                    
                    // Calculate index
                    // Index = c * (T * H * W) + t * (H * W) + y * W + x
                    int idx = c * (NUM_FRAMES * CROP_SIZE * CROP_SIZE) + 
                              i * (CROP_SIZE * CROP_SIZE) + 
                              y * CROP_SIZE + x;
                    hostInputBuffer[idx] = pixel;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <image_folder>" << std::endl;
        return -1;
    }

    std::string engineFile = argv[1];
    std::string imageFolder = argv[2];

    // Initialize TensorRT
    IRuntime* runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create runtime" << std::endl;
        return -1;
    }

    ICudaEngine* engine = loadEngine(engineFile, runtime);
    if (!engine) {
        std::cerr << "Failed to load engine" << std::endl;
        return -1;
    }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return -1;
    }

    // Allocate memory
    // Input: 1 x 3 x 32 x 64 x 64
    size_t inputSize = 1 * 3 * NUM_FRAMES * CROP_SIZE * CROP_SIZE * sizeof(float);
    // Output: 1 x 2 (Num Classes)
    size_t outputSize = 1 * 2 * sizeof(float);

    void* deviceInput;
    void* deviceOutput;
    checkCudaErrors(cudaMalloc(&deviceInput, inputSize));
    checkCudaErrors(cudaMalloc(&deviceOutput, outputSize));

    float* hostInput = new float[inputSize / sizeof(float)];
    float* hostOutput = new float[outputSize / sizeof(float)];

    // Preprocess
    std::cout << "Preprocessing images from " << imageFolder << "..." << std::endl;
    preprocess(imageFolder, hostInput);

    // Copy to device
    checkCudaErrors(cudaMemcpy(deviceInput, hostInput, inputSize, cudaMemcpyHostToDevice));

    // Run inference
    std::cout << "Running inference..." << std::endl;
    void* bindings[] = {deviceInput, deviceOutput};
    
    // Note: For dynamic shapes, you might need to set input dimensions explicitly
    // context->setInputShape("input_name", Dims...);
    // Assuming the engine is built with the correct static shape or we use the default profile
    
    bool status = context->executeV2(bindings);
    if (!status) {
        std::cerr << "Inference failed" << std::endl;
        return -1;
    }

    // Copy back
    checkCudaErrors(cudaMemcpy(hostOutput, deviceOutput, outputSize, cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Inference Results:" << std::endl;
    std::cout << "Class 0: " << hostOutput[0] << std::endl;
    std::cout << "Class 1: " << hostOutput[1] << std::endl;

    // Cleanup
    delete[] hostInput;
    delete[] hostOutput;
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
