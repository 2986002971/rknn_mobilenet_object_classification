// Copyright 2020 Fuzhou Rockchip Electronics Co., Ltd. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "atk_mobilenet_object_classification.h"


int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <image_path>\n", argv[0]);
    return -1;
  }

  // 读取输入图片
  cv::Mat img = cv::imread(argv[1]);
  if (img.empty()) {
    printf("ERROR: Cannot read image file %s\n", argv[1]);
    return -1;
  }

  // 初始化RKNN
  rknn_context ctx;
  int model_len = 0;
  unsigned char *model;
  const char *model_path = "./mobilenet_v1_rv1109_rv1126.rknn";

  printf("Loading model...\n");
  model = load_model(model_path, &model_len);
  if (!model) {
    return -1;
  }

  int ret = rknn_init(&ctx, model, model_len, 0);
  if (ret < 0) {
    printf("rknn_init fail! ret=%d\n", ret);
    free(model);
    return -1;
  }

  // 获取模型信息
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }

  // 获取输入tensor信息
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (unsigned int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
  }

  // 获取模型输入尺寸
  int model_width = input_attrs[0].dims[1];
  int model_height = input_attrs[0].dims[2];

  // 调整图片尺寸
  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(model_width, model_height));

  // 准备输入数据
  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].size = model_width * model_height * 3;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].buf = resized_img.data;

  ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
  if (ret < 0) {
    printf("rknn_inputs_set fail! ret=%d\n", ret);
    return -1;
  }

  // 运行推理
  ret = rknn_run(ctx, nullptr);
  if (ret < 0) {
    printf("rknn_run fail! ret=%d\n", ret);
    return -1;
  }

  // 获取输出
  rknn_output outputs[io_num.n_output];
  memset(outputs, 0, sizeof(outputs));
  for (unsigned int i = 0; i < io_num.n_output; i++) {
    outputs[i].want_float = 1;
  }
  ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
  if (ret < 0) {
    printf("rknn_outputs_get fail! ret=%d\n", ret);
    return -1;
  }

  // 处理结果
  uint32_t MaxClass[5];
  float fMaxProb[5];
  float *buffer = (float *)outputs[0].buf;
  uint32_t sz = outputs[0].size / 4;

  rknn_GetTop(buffer, fMaxProb, MaxClass, sz, 5);
  printf("\nClassification Results:\n");
  printf("----------------------\n");
  for (int i = 0; i < 5; i++) {
    printf("Top-%d: Class %d, Probability: %.6f\n", i+1, MaxClass[i]-1, fMaxProb[i]);
  }

  // 清理资源
  rknn_outputs_release(ctx, io_num.n_output, outputs);
  rknn_destroy(ctx);
  free(model);

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

