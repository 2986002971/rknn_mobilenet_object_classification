// Copyright 2020 Fuzhou Rockchip Electronics Co., Ltd. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "atk_mobilenet_object_classification.h"
#include "mongoose.h"

// Base64解码表
static const unsigned char base64_table[256] = {
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,62,255,255,255,63,
    52,53,54,55,56,57,58,59,60,61,255,255,255,0,255,255,
    255,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
    15,16,17,18,19,20,21,22,23,24,25,255,255,255,255,255,
    255,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
    41,42,43,44,45,46,47,48,49,50,51,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
};

// Base64解码函数
std::vector<unsigned char> base64_decode(const std::string &encoded_string) {
    size_t in_len = encoded_string.size();
    size_t i = 0, j = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::vector<unsigned char> decoded_data;

    while (in_len-- && (encoded_string[i] != '=')) {
        unsigned char c = base64_table[(unsigned char)encoded_string[i++]];
        if (c == 255) continue; // 跳过无效字符
        
        char_array_4[j++] = c;
        if (j == 4) {
            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (j = 0; j < 3; j++)
                decoded_data.push_back(char_array_3[j]);
            j = 0;
        }
    }

    if (j) {
        for (size_t k = j; k < 4; k++)
            char_array_4[k] = 0;

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (size_t k = 0; k < j - 1; k++)
            decoded_data.push_back(char_array_3[k]);
    }

    return decoded_data;
}

// 添加softmax函数
static void softmax(float* input, size_t size) {
    float max_val = input[0];
    float sum = 0.0f;
    
    // 找最大值
    for (size_t i = 0; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // 计算exp并求和
    for (size_t i = 0; i < size; i++) {
        input[i] = exp(input[i] - max_val);  // 减去最大值避免数值溢出
        sum += input[i];
    }
    
    // 归一化
    for (size_t i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

// SVM Model class
class SVMModel {
public:
    SVMModel(const std::string& model_path) {
        net = cv::dnn::readNetFromONNX(model_path);
    }

    float predict(const std::vector<float>& features) {
        // 使用训练时相同的标准化方法
        std::vector<float> normalized(features.size());
        for (size_t i = 0; i < features.size(); i++) {
            normalized[i] = features[i];  // 先复制原始数据
        }
        
        cv::Mat input(1, normalized.size(), CV_32F, (void*)normalized.data());
        net.setInput(input);
        cv::Mat output = net.forward();
        
        // 检查输出形状 - 现在应该是1x3的输出
        if (output.rows != 1 || output.cols != 3) {
            printf("⚠️ 模型输出形状错误: %d x %d (应为 1x3)\n", output.rows, output.cols);
            return 0.0f;
        }
        
        // 找出最大概率的类别并应用softmax
        float probs[3];
        for (int i = 0; i < 3; i++) {
            probs[i] = output.at<float>(0, i);
        }
        softmax(probs, 3);
        
        float max_prob = probs[0];
        int max_index = 0;
        for (int i = 1; i < 3; i++) {
            if (probs[i] > max_prob) {
                max_prob = probs[i];
                max_index = i;
            }
        }

        // 在predict函数中添加调试输出
    printf("原始输出值: ");
    for (int i = 0; i < 3; i++) {
        printf("%.4f ", output.at<float>(0, i));
    }
    printf("\n");

    printf("Softmax后概率: ");
    for (int i = 0; i < 3; i++) {
        printf("%.4f ", probs[i]);
    }
    printf("\n");
        
        return static_cast<float>(max_index);  // 返回类别索引
    }
    
private:
    cv::dnn::Net net;
};

// Weighted fusion function
FusionResult weighted_fusion(float svm_score, float rknn_score,
                           float svm_weight = 0.5f, float rknn_weight = 0.5f) {
    FusionResult res;
    // 如果两个模型预测结果一致，就使用该结果
    if (int(svm_score) == int(rknn_score)) {
        res.class_id = int(svm_score);
        res.probability = (svm_weight * svm_score + rknn_weight * rknn_score);
    } else {
        // 如果预测不一致，使用概率较高的结果
        if (svm_weight * svm_score > rknn_weight * rknn_score) {
            res.class_id = int(svm_score);
            res.probability = svm_score;
        } else {
            res.class_id = int(rknn_score);
            res.probability = rknn_score;
        }
    }
    res.svm_score = svm_score;
    res.rknn_score = rknn_score;
    return res;
}

// Global SVM model instance
static SVMModel* svm_model = nullptr;

#define HTTP_PORT "8080"
static const char *s_listen_addr = "http://0.0.0.0:" HTTP_PORT;
static struct mg_mgr mgr;

// 封装原有分类逻辑
static ClassificationResult classify_image(const void *data, size_t len) {
  ClassificationResult res = {0, 0.0f};  // 简单初始化：class_id=0, probability=0.0
  
  // 将静态变量改为局部静态确保线程安全
  static std::once_flag init_flag;
  static rknn_context ctx;
  static rknn_input_output_num io_num;
  static int model_width = 0, model_height = 0;

  std::call_once(init_flag, [&](){
      const char *model_path = "./model.rknn";
      int model_len = 0;
      unsigned char *model = load_model(model_path, &model_len);
      if (rknn_init(&ctx, model, model_len, 0) < 0) {
          fprintf(stderr, "Model init failed\n");
          exit(1);
      }

      // 查询输入输出数量
      if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) < 0) {
          fprintf(stderr, "查询输入输出数量失败\n");
          exit(1);
      }
      printf("模型信息: 输入数量=%d, 输出数量=%d\n", 
             io_num.n_input, io_num.n_output);

      // 初始化模型尺寸
      rknn_tensor_attr input_attr = {0};
      input_attr.index = 0;
      rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
      model_width = input_attr.dims[1];
      model_height = input_attr.dims[2];
  });

  // 将二进制数据解码为OpenCV Mat
  cv::Mat img = cv::imdecode(cv::Mat(1, len, CV_8U, (void*)data), cv::IMREAD_COLOR);
  if (img.empty()) {
    fprintf(stderr, "Image decode failed\n");
    return res;
  }

  // 图像预处理
  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(model_width, model_height));

  // 设置输入
  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].buf = resized_img.data;
  inputs[0].size = (uint32_t)(model_width * model_height * 3);
  inputs[0].pass_through = false;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  
  if (rknn_inputs_set(ctx, 1, inputs) < 0) {
    fprintf(stderr, "设置输入失败\n");
    return res;
  }

  // 执行推理
  if (rknn_run(ctx, nullptr) < 0) {
    fprintf(stderr, "Inference failed\n");
    return res;
  }

  // 获取输出
  rknn_output outputs[io_num.n_output];
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < io_num.n_output; i++) {
      outputs[i].want_float = 1;
  }
  
  if (rknn_outputs_get(ctx, io_num.n_output, outputs, NULL) < 0) {
      fprintf(stderr, "获取输出失败\n");
      return res;
  }

  // 处理每个输出
  for (int i = 0; i < io_num.n_output; i++) {
      float *buffer = (float *)outputs[i].buf;
      
      printf("处理输出 %d: 大小=%u 字节\n", i, outputs[i].size);
      
      // 只处理第一个输出（假设这是分类结果）
      if (i == 0) {
          rknn_GetResult(buffer, &res);
          
          // 打印结果用于调试
          printf("分类结果:\n");
          printf("  class=%d, probability=%.4f\n", 
                 res.class_id, res.probability);
      }
  }

  // 确保释放输出资源
  rknn_outputs_release(ctx, io_num.n_output, outputs);
  
  return res;
}

// 添加自定义方法比较函数
static int method_cmp(struct mg_str method, const char *expected) {
  size_t n = strlen(expected);
  return method.len != n || strncasecmp(method.buf, expected, n) != 0;
}

// HTTP事件处理
static void fn(struct mg_connection *c, int ev, void *ev_data) {
  if (ev == MG_EV_HTTP_MSG) {
    struct mg_http_message *hm = (struct mg_http_message *)ev_data;
    
    // 添加详细的请求日志
    printf("\n=== 收到新请求 ===\n");
    printf("客户端地址: %s\n", c->rem.ip);
    printf("请求方法: %.*s\n", (int)hm->method.len, hm->method.buf);
    printf("请求路径: %.*s\n", (int)hm->uri.len, hm->uri.buf);
    printf("协议版本: %.*s\n", (int)hm->proto.len, hm->proto.buf);
    
    // 修正headers的访问方式
    printf("请求头数量: %d\n", (int)(sizeof(hm->headers)/sizeof(hm->headers[0])));
    for (size_t i = 0; i < sizeof(hm->headers)/sizeof(hm->headers[0]); i++) {
      if (hm->headers[i].name.len == 0) break;
      printf("Header: %.*s => %.*s\n", 
             (int)hm->headers[i].name.len, hm->headers[i].name.buf,
             (int)hm->headers[i].value.len, hm->headers[i].value.buf);
    }
    
    printf("请求体大小: %zu bytes\n", hm->body.len);
    
    struct mg_str uri_pattern = mg_str("/api/classify");
    if (mg_match(hm->uri, uri_pattern, NULL)) {
      if (method_cmp(hm->method, "POST")) {
        printf("⚠️ 方法不匹配 | 实际方法: %.*s\n", 
               (int)hm->method.len, hm->method.buf);
        mg_http_reply(c, 405, "", "{\"error\":\"Method not allowed\"}");
        return;
      }
      
      printf("✅ 开始处理图像分类...\n");
      struct timespec start, end;
      clock_gettime(CLOCK_MONOTONIC, &start);
      
      // Simple JSON parser
      std::string json_str(hm->body.buf, hm->body.len);
      
      // Extract image data
      std::string image_data;
      size_t image_pos = json_str.find("\"image\":");
      if (image_pos != std::string::npos) {
          size_t start = json_str.find("\"", image_pos + 8);
          if (start != std::string::npos) {
              size_t end = json_str.find("\"", start + 1);
              if (end != std::string::npos) {
                  image_data = json_str.substr(start + 1, end - start - 1);
              }
          }
      }
      
      if (image_data.empty()) {
          printf("⚠️ JSON解析失败: 未找到image字段\n");
          mg_http_reply(c, 400, "", "{\"error\":\"Invalid JSON: missing image field\"}");
          return;
      }
      
      // 打印部分图像数据用于调试
      printf("图像数据大小: %zu bytes\n", image_data.size());
      printf("图像数据前100字节: ");
      for (size_t i = 0; i < std::min(image_data.size(), (size_t)100); i++) {
          printf("%02x ", (unsigned char)image_data[i]);
          if ((i + 1) % 16 == 0) printf("\n");
      }
      printf("\n");
      
      // Base64解码图像数据
      printf("开始Base64解码图像数据\n");
      std::vector<unsigned char> decoded_image = base64_decode(image_data);
      printf("Base64解码完成，解码后数据大小: %zu bytes\n", decoded_image.size());
      
      if (decoded_image.empty()) {
          printf("⚠️ Base64解码失败\n");
          mg_http_reply(c, 400, "", "{\"error\":\"Failed to decode base64 image\"}");
          return;
      }
      
      ClassificationResult rknn_res = classify_image(
          decoded_image.data(), decoded_image.size());
      
      // Extract features with error handling
      std::vector<float> features;
      size_t features_pos = json_str.find("\"features\":[");
      if (features_pos != std::string::npos) {
          size_t start = features_pos + 11;
          size_t end = json_str.find("]", start);
          if (end != std::string::npos) {
              std::string features_str = json_str.substr(start, end - start);
              
              // 去除所有空格和方括号
              features_str.erase(std::remove(features_str.begin(), features_str.end(), ' '), features_str.end());
              features_str.erase(std::remove(features_str.begin(), features_str.end(), '['), features_str.end());
              features_str.erase(std::remove(features_str.begin(), features_str.end(), ']'), features_str.end());
              
              // 验证字符串是否只包含数字和逗号
              if (features_str.find_first_not_of("0123456789,.-") != std::string::npos) {
                  printf("⚠️ 特征格式错误: 包含非法字符\n");
                  printf("原始特征字符串: %s\n", features_str.c_str());
                  mg_http_reply(c, 400, "", "{\"error\":\"Invalid features format: contains invalid characters\"}");
                  return;
              }
              
              size_t pos = 0;
              try {
                  while ((pos = features_str.find(',')) != std::string::npos) {
                      std::string num_str = features_str.substr(0, pos);
                      if (!num_str.empty()) {
                          features.push_back(std::stof(num_str));
                      }
                      features_str.erase(0, pos + 1);
                  }
                  if (!features_str.empty()) {
                      features.push_back(std::stof(features_str));
                  }
              } catch (const std::invalid_argument& e) {
                  printf("⚠️ 特征解析失败: %s\n", e.what());
                  printf("原始特征字符串: %s\n", features_str.c_str());
                  mg_http_reply(c, 400, "", "{\"error\":\"Invalid features format: failed to parse numbers\"}");
                  return;
              }
          }
      }
      
      // 检查特征数量
      if (features.size() != 34) {
          printf("⚠️ 特征数量错误: 期望34个，实际收到%zu个\n", features.size());
          mg_http_reply(c, 400, "", "{\"error\":\"Invalid features: expected 34 features\"}");
          return;
      }
      
      float svm_score = svm_model->predict(features);
      
      // Combine results
      FusionResult final_res = weighted_fusion(svm_score, rknn_res.probability);
      
      clock_gettime(CLOCK_MONOTONIC, &end);
      double elapsed = (end.tv_sec - start.tv_sec) +
                      (end.tv_nsec - start.tv_nsec) / 1e9;
      printf("🕒 处理耗时: %.3f 秒\n", elapsed);
      
      printf("📤 发送响应...\n");

      // Generate JSON response
      char json_response[512];
      snprintf(json_response, sizeof(json_response),
          "{\"class\":%d,\"probability\":%.4f,\"blood_score\":%.4f,\"rknn_score\":%.4f}",
          final_res.class_id,
          final_res.probability,
          final_res.svm_score,
          final_res.rknn_score);

      printf("融合结果: 类别=%d, 概率=%.4f, 血常规分数=%.4f, RKNN分数=%.4f\n",
             final_res.class_id, final_res.probability,
             final_res.svm_score, final_res.rknn_score);

      // Send response
      mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%s",
                   json_response);

      // 确保数据发送完成
      c->is_resp = 1;  // 标记为响应已发送
      c->is_draining = 1;  // 确保所有数据都被发送

      // 保存推理结果
      save_inference_result(final_res, elapsed);
    } else {
      printf("⚠️ 拒绝请求：路径未找到\n");
      mg_http_reply(c, 404, "", "{\"error\":\"Not Found\"}");
    }
    printf("=== 请求处理结束 ===\n\n");
  }
}

int main(int argc, char *argv[]) {
  // Initialize SVM model
  svm_model = new SVMModel("nn_model.onnx");
  
  mg_mgr_init(&mgr);
  mg_http_listen(&mgr, s_listen_addr, fn, NULL);
  printf("🚀 服务器已启动，监听地址: %s\n", s_listen_addr);
  printf("📡 等待客户端连接...\n");
  
  // 主事件循环
  for (;;) {
    mg_mgr_poll(&mgr, 50); // 50ms timeout
  }
  
  mg_mgr_free(&mgr);
  return 0;
}

static unsigned char *load_model(const char *filename, int *model_size) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("fopen %s fail!\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  unsigned int model_len = ftell(fp);
  unsigned char *model = (unsigned char *)malloc(model_len);
  fseek(fp, 0, SEEK_SET);

  if (model_len != fread(model, 1, model_len, fp)) {
    printf("fread %s fail!\n", filename);
    free(model);
    return NULL;
  }
  *model_size = model_len;

  if (fp) {
    fclose(fp);
  }
  return model;
}

// 修改结果处理函数
// 保存推理结果到CSV文件
void save_inference_result(const FusionResult& result, double processing_time) {
    static std::mutex file_mutex;
    std::lock_guard<std::mutex> lock(file_mutex);
    
    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    
    // 打开文件追加写入
    std::ofstream outfile("inference_results.csv", std::ios::app);
    if (!outfile.is_open()) {
        printf("⚠️ 无法打开结果文件\n");
        return;
    }
    
    // 写入CSV格式数据
    outfile << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S") << ","
            << result.class_id << ","
            << result.probability << ","
            << result.svm_score << ","
            << result.rknn_score << ","
            << processing_time << "\n";
}

static int rknn_GetResult(float *prob_data, struct ClassificationResult *result) {
    // 对输出进行softmax处理
    softmax(prob_data, 3);  // 修改为3个类别
    
    // 找出最大概率的类别
    float max_prob = prob_data[0];
    int max_index = 0;
    
    for(int i = 1; i < 3; i++) {
        if(prob_data[i] > max_prob) {
            max_prob = prob_data[i];
            max_index = i;
        }
    }
    
    result->class_id = max_index;  // 0: 健康, 1: 细菌感染, 2: 支原体感染
    result->probability = max_prob;
    
    return 0;
}