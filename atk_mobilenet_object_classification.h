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

// RKNN相关
#include "rknn_api.h"

// OpenCV相关
#include "opencv2/opencv.hpp"

// HTTP服务器相关
#include "mongoose.h"

// 分类结果结构体
typedef struct {
    uint32_t classes[5];  // 前5个分类的索引
    float probs[5];       // 对应的概率值
} ClassificationResult;

// 函数声明
static unsigned char *load_model(const char *filename, int *model_size);
static int rknn_GetTop(float *pfProb, float *pfMaxProb, uint32_t *pMaxClass,
                      uint32_t outputCount, uint32_t topNum);

#endif // _ATK_MOBILENET_OBJECT_CLASSIFICATION_H