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
#include <mutex>
#include <condition_variable>

// RKNN相关
#include "rknn_api.h"

// OpenCV相关
#include "opencv2/opencv.hpp"

// HTTP服务器相关
#include "mongoose.h"

// 分类结果结构体
typedef struct {
    uint32_t class_id;    // 分类结果（0或1）
    float probability;    // 概率值
} ClassificationResult;

// 函数声明
static unsigned char *load_model(const char *filename, int *model_size);
static int rknn_GetResult(float *prob_data, ClassificationResult *result);

#endif // _ATK_MOBILENET_OBJECT_CLASSIFICATION_H