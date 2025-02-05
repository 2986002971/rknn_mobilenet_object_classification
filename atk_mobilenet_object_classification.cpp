// Copyright 2020 Fuzhou Rockchip Electronics Co., Ltd. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "atk_mobilenet_object_classification.h"
#include "mongoose.h"

#define HTTP_PORT "8080"
static const char *s_listen_addr = "http://0.0.0.0:" HTTP_PORT;
static struct mg_mgr mgr;

// 分类结果结构体
struct ClassificationResult {
  uint32_t classes[5];
  float probs[5];
};

// 封装原有分类逻辑
static ClassificationResult classify_image(const void *data, size_t len) {
  ClassificationResult res = {{0}, {0}};
  
  // 将二进制数据解码为OpenCV Mat
  cv::Mat img = cv::imdecode(cv::Mat(1, len, CV_8U, (void*)data), cv::IMREAD_COLOR);
  if (img.empty()) {
    fprintf(stderr, "Image decode failed\n");
    return res;
  }

  // 复用原有处理流程
  static rknn_context ctx;
  static unsigned char *model = NULL;
  static int model_width = 0, model_height = 0;
  
  // 首次加载模型
  if (model == NULL) {
    int model_len = 0;
    const char *model_path = "./mobilenet_v1_rv1109_rv1126.rknn";
    model = load_model(model_path, &model_len);
    if (rknn_init(&ctx, model, model_len, 0) < 0) {
      fprintf(stderr, "Model init failed\n");
      return res;
    }
    
    // 获取模型输入尺寸（原有逻辑）
    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    rknn_tensor_attr input_attr;
    input_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    model_width = input_attr.dims[1];
    model_height = input_attr.dims[2];
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
  
  rknn_inputs_set(ctx, 1, inputs);

  // 执行推理
  if (rknn_run(ctx, nullptr) < 0) {
    fprintf(stderr, "Inference failed\n");
    return res;
  }

  // 获取输出
  rknn_output outputs[1];
  outputs[0].want_float = 1;
  rknn_outputs_get(ctx, 1, outputs, NULL);
  
  // 处理结果
  float *buffer = (float *)outputs[0].buf;
  uint32_t sz = outputs[0].size / 4;
  rknn_GetTop(buffer, res.probs, res.classes, sz, 5);
  
  // 调整类别索引（根据原始代码逻辑）
  for (int i = 0; i < 5; i++) res.classes[i] -= 1;

  rknn_outputs_release(ctx, 1, outputs);
  return res;
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
      if (mg_casecmp(hm->method.buf, "POST") != 0) {
        printf("⚠️ 拒绝请求：不支持的HTTP方法\n");
        mg_http_reply(c, 405, "", "{\"error\":\"Method not allowed\"}");
        return;
      }
      
      printf("✅ 开始处理图像分类...\n");
      struct timespec start, end;
      clock_gettime(CLOCK_MONOTONIC, &start);
      
      ClassificationResult res = classify_image(hm->body.buf, hm->body.len);
      
      clock_gettime(CLOCK_MONOTONIC, &end);
      double elapsed = (end.tv_sec - start.tv_sec) + 
                      (end.tv_nsec - start.tv_nsec) / 1e9;
      printf("🕒 分类耗时: %.3f 秒\n", elapsed);
      
      printf("📤 发送响应...\n");
      mg_http_reply(c, 200, "Content-Type: application/json\r\n",
                   "{\"results\":["
                   "{\"class\":%d,\"prob\":%.4f},"
                   "{\"class\":%d,\"prob\":%.4f},"
                   "{\"class\":%d,\"prob\":%.4f},"
                   "{\"class\":%d,\"prob\":%.4f},"
                   "{\"class\":%d,\"prob\":%.4f}]}",
                   res.classes[0], res.probs[0],
                   res.classes[1], res.probs[1],
                   res.classes[2], res.probs[2],
                   res.classes[3], res.probs[3],
                   res.classes[4], res.probs[4]);
      printf("✅ 响应已发送\n");
    } else {
      printf("⚠️ 拒绝请求：路径未找到\n");
      mg_http_reply(c, 404, "", "{\"error\":\"Not Found\"}");
    }
    printf("=== 请求处理结束 ===\n\n");
  }
}

int main(int argc, char *argv[]) {
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


static unsigned char *load_model(const char *filename, int *model_size)
{
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL)
  {
    printf("fopen %s fail!\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  unsigned int model_len = ftell(fp);
  unsigned char *model = (unsigned char *)malloc(model_len);
  fseek(fp, 0, SEEK_SET);

  if (model_len != fread(model, 1, model_len, fp))
  {
    printf("fread %s fail!\n", filename);
    free(model);
    return NULL;
  }
  *model_size = model_len;

  if (fp)
  {
    fclose(fp);
  }
  return model;
}


static void printRKNNTensor(rknn_tensor_attr *attr)
{
  printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d "
         "fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2],
         attr->dims[1], attr->dims[0], attr->n_elems, attr->size, 0, attr->type,
         attr->qnt_type, attr->fl, attr->zp, attr->scale);
}


int rgb24_resize(unsigned char *input_rgb, unsigned char *output_rgb, 
                 int width,int height, int outwidth, int outheight)
{
  rga_buffer_t src =wrapbuffer_virtualaddr(input_rgb, width, height, RK_FORMAT_RGB_888);
  rga_buffer_t dst = wrapbuffer_virtualaddr(output_rgb, outwidth, outheight,RK_FORMAT_RGB_888);
  rga_buffer_t pat = {0};
  im_rect src_rect = {0, 0, width, height};
  im_rect dst_rect = {0, 0, outwidth, outheight};
  im_rect pat_rect = {0};
  IM_STATUS STATUS = improcess(src, dst, pat, src_rect, dst_rect, pat_rect, 0);
  if (STATUS != IM_STATUS_SUCCESS)
  {
    printf("imcrop failed: %s\n", imStrError(STATUS));
    return -1;
  }
  return 0;
}


static int rknn_GetTop(float *pfProb, float *pfMaxProb,uint32_t *pMaxClass, uint32_t outputCount, uint32_t topNum)
{
  uint32_t i, j;
#define MAX_TOP_NUM 20
  if (topNum > MAX_TOP_NUM)
  {
    return 0;
  }
  
  memset(pfMaxProb, 0, sizeof(float) * topNum);
  memset(pMaxClass, 0xff, sizeof(float) * topNum);

  for (j = 0; j < topNum; j++)
  {
    for (i = 0; i < outputCount; i++)
    {
      if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) ||
          (i == *(pMaxClass + 3)) || (i == *(pMaxClass + 4)))
      {
        continue;
      }

      if (pfProb[i] > *(pfMaxProb + j))
      {
        *(pfMaxProb + j) = pfProb[i];
        *(pMaxClass + j) = i;
      }
    }
  }

  return 1;
}


void *rkmedia_rknn_thread(void *args)
{
  pthread_detach(pthread_self());

  int ret;
  rknn_context ctx;
  int model_len = 0;
  unsigned char *model;
  static char *model_path = "./mobilenet_v1_rv1109_rv1126.rknn";

  // Load RKNN Model
  printf("Loading model ...\n");            
  model = load_model(model_path, &model_len);
  ret = rknn_init(&ctx, model, model_len, 0);
  if (ret < 0)
  {
    printf("rknn_init fail! ret=%d\n", ret);
    return NULL;
  }

  // Get Model Input Output Info
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC)
  {
    printf("rknn_query fail! ret=%d\n", ret);
    return NULL;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input,io_num.n_output);

  // print input tensor
  printf("input tensors:\n");
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (unsigned int i = 0; i < io_num.n_input; i++)
  {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
    {
      printf("rknn_query fail! ret=%d\n", ret);
      return NULL;
    }
    printRKNNTensor(&(input_attrs[i]));
  }

  // print output tensor
  printf("output tensors:\n");
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (unsigned int i = 0; i < io_num.n_output; i++)
  {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
    {
      printf("rknn_query fail! ret=%d\n", ret);
      return NULL;
    }
    printRKNNTensor(&(output_attrs[i]));
  }

  int model_height = 0;
  int model_width = 0;
  int model_channel = 0;
  switch (input_attrs->fmt)
  {
  case RKNN_TENSOR_NHWC:
      model_height = input_attrs->dims[2];
      model_width = input_attrs->dims[1];
      model_channel = input_attrs->dims[0];
      break;
  case RKNN_TENSOR_NCHW:
      model_height = input_attrs->dims[1];
      model_width = input_attrs->dims[0];
      model_channel = input_attrs->dims[2];
      break;
  default:
      printf("meet unsupported layout\n");
      return NULL;
  }
  printf("###w=%d,h=%d,c=%d, fmt=%d\n", model_width, model_height, model_channel, input_attrs->fmt);

  while (!quit)
  {
    MEDIA_BUFFER src_mb = NULL;
    src_mb = RK_MPI_SYS_GetMediaBuffer(RK_ID_RGA, 0, -1);
    if (!src_mb)
    {
      printf("ERROR: RK_MPI_SYS_GetMediaBuffer get null buffer!\n");
      break;
    }

    /*================================================================================
      =========================使用drm拷贝，可以使用如下代码===========================
      ================================================================================*/
    /*
    rga_context rga_ctx;
    drm_context drm_ctx;
    memset(&rga_ctx, 0, sizeof(rga_context));
    memset(&drm_ctx, 0, sizeof(drm_context));

     // DRM alloc buffer
    int drm_fd = -1;
    int buf_fd = -1; // converted from buffer handle
    unsigned int handle;
    size_t actual_size = 0;
    void *drm_buf = NULL;

    drm_fd = drm_init(&drm_ctx);
    drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, video_width, video_height, 3 * 8, &buf_fd, &handle, &actual_size);
    memcpy(drm_buf, (uint8_t *)RK_MPI_MB_GetPtr(src_mb) , video_width * video_height * 3);
    void *resize_buf = malloc(model_width * model_height * 3);
    // init rga context
    RGA_init(&rga_ctx);
    img_resize_slow(&rga_ctx, drm_buf, video_width, video_height, resize_buf, model_width, model_height);
    uint32_t input_model_image_size = model_width * model_height * 3;

    // Set Input Data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = input_model_image_size;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = resize_buf;
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0)
    {
      printf("ERROR: rknn_inputs_set fail! ret=%d\n", ret);
      return NULL;
    }
    free(resize_buf);
    drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
    drm_deinit(&drm_ctx, drm_fd);
    RGA_deinit(&rga_ctx);
    */

    /*================================================================================
      =========================不使用drm拷贝，可以使用如下代码===========================
      ================================================================================*/
    
    unsigned char *orig_image_buf = (unsigned char *)RK_MPI_MB_GetPtr(src_mb);

    if (video_width != model_width || video_height != model_height)
    {
      uint32_t input_model_image_size = model_width * model_height * 3;
      unsigned char *input_model_image_buf = (unsigned char *)malloc(input_model_image_size);
      rgb24_resize(orig_image_buf, input_model_image_buf, video_width, video_height, model_width, model_height);

      // Set Input Data
      rknn_input inputs[1];
      memset(inputs, 0, sizeof(inputs));
      inputs[0].index = 0;
      inputs[0].type = RKNN_TENSOR_UINT8;
      inputs[0].size = input_model_image_size;
      inputs[0].fmt = RKNN_TENSOR_NHWC;
      inputs[0].buf = input_model_image_buf;
      ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
      if (ret < 0)
      {
        printf("ERROR: rknn_inputs_set fail! ret=%d\n", ret);
        return NULL;
      }
      free(input_model_image_buf);
    }
    

    // Run
    printf("rknn_run\n");
    ret = rknn_run(ctx, nullptr);
    if (ret < 0)
    {
      printf("ERROR: rknn_run fail! ret=%d\n", ret);
      return NULL;
    }

    // Get Output
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (unsigned int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 1;
    }
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
      printf("ERROR: rknn_outputs_get fail! ret=%d\n", ret);
      return NULL;
    }

    // Post Process
    for (unsigned int i = 0; i < io_num.n_output; i++)
    {
      uint32_t MaxClass[5];
      float fMaxProb[5];
      float *buffer = (float *)outputs[i].buf;
      uint32_t sz = outputs[i].size / 4;

      rknn_GetTop(buffer, fMaxProb, MaxClass, sz, 5);
      printf(" --- Top5 ---\n");
      for (int i = 0; i < 5; i++)
      {
        printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);

        // 采用opencv来绘制矩形框,颜色格式是B、G、R
        using namespace cv;
        Mat orig_img = Mat(video_height, video_width, CV_8UC3, RK_MPI_MB_GetPtr(src_mb));

        std::string index = std::to_string(MaxClass[i]-1);
        putText(orig_img, index, Point(756+32, 12+64+i*48), FONT_HERSHEY_TRIPLEX, 2, Scalar(0,0,255),2,8,0);

        std::string prob = std::to_string(fMaxProb[i]);
        putText(orig_img, prob, Point(756+260, 12+64+i*48), FONT_HERSHEY_TRIPLEX, 2, Scalar(0,0,255),2,8,0);
      }
    }

    rknn_outputs_release(ctx, io_num.n_output, outputs);

    rga_buffer_t src , dst ;
    MB_IMAGE_INFO_S dst_ImageInfo = {(RK_U32)video_width, (RK_U32)video_height, (RK_U32)video_width, 
                                     (RK_U32)video_height, IMAGE_TYPE_RGB888};
    MEDIA_BUFFER dst_mb = RK_MPI_MB_CreateImageBuffer(&dst_ImageInfo, RK_TRUE, 0);
    dst = wrapbuffer_fd(RK_MPI_MB_GetFD(dst_mb), video_width, video_height,RK_FORMAT_RGB_888);
    src = wrapbuffer_fd(RK_MPI_MB_GetFD(src_mb), video_width, video_height,RK_FORMAT_RGB_888);
    
    im_rect src_rect , dst_rect;
    src_rect = {0, 0, 2592, 1944};
    dst_rect = {0};
    ret = imcheck(src, dst, src_rect, dst_rect, IM_CROP);
    if (IM_STATUS_NOERROR != ret) 
    {
      printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
      break;
    }

    IM_STATUS CROP_STATUS = imcrop(src, dst, src_rect);
    if (CROP_STATUS != IM_STATUS_SUCCESS)
    {
      printf("ERROR: imcrop failed: %s\n", imStrError(CROP_STATUS));
    }

    RK_MPI_SYS_SendMediaBuffer(RK_ID_VO, 0, dst_mb);
    RK_MPI_MB_ReleaseBuffer(dst_mb);
    RK_MPI_MB_ReleaseBuffer(src_mb);
  
    src_mb = NULL;
    dst_mb= NULL;
  }

  if (model)
  {
    free(model);
  }

  if (ctx >= 0)
  {
    rknn_destroy(ctx);
  }
    
  return NULL;
}

