#ifndef _ATK_MOBILENET_OBJECT_CLASSIFICATION_H
#define _ATK_MOBILENET_OBJECT_CLASSIFICATION_H

// 标准库
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>
#include <signal.h>
#include <malloc.h>
#include <dlfcn.h>
#include <mutex>
#include <math.h>
#include <opencv2/dnn.hpp>
#include <thread>
#include <vector>
#include <queue>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <chrono>

// RKNN相关
#include "rknn_api.h"

// OpenCV相关
#include "opencv2/opencv.hpp"

// HTTP服务器相关
#include "mongoose.h"


// 函数声明
static unsigned char *load_model(const char *filename, int *model_size);
static int rknn_GetResult(float *prob_data, struct ClassificationResult *result);

struct ClassificationResult {
    int class_id;
    float probability;
};

// 分类结果结构体
struct FusionResult {
    int class_id;
    float probability;
    float svm_score;
    float rknn_score;
};

// 添加结果保存函数声明
void save_inference_result(const FusionResult& result,
                         double processing_time);

#endif // _ATK_MOBILENET_OBJECT_CLASSIFICATION_H