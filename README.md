# 图像分类服务器文档

## 1. 概述
基于RKNN的HTTP图像分类服务，提供RESTful API接口，支持实时物体分类。使用MobileNet模型实现高效推理，适用于嵌入式设备部署。

## 2. 技术特性
- **推理框架**：RKNN Runtime 1.7.5
- **模型支持**：MobileNet v1 (输入224x224 RGB)
- **协议支持**：HTTP/1.1
- **平均延迟**：<50ms (RV1126)
- **最大并发**：未测试

## 3. API接口

### 3.1 分类请求
```http
POST /api/classify HTTP/1.1
Content-Type: image/jpeg
Content-Length: <file_size>

<binary_image_data>
```

**请求参数**：
- 图片格式：JPEG
- 推荐分辨率：224x224
- 最大文件大小：4MB

**成功响应**：
```json
{
  "results": [
    {"class": 0, "prob": 0.8123},
    {"class": 1, "prob": 0.1021},
    {"class": 2, "prob": 0.0456},
    {"class": 3, "prob": 0.0211},
    {"class": 4, "prob": 0.0089}
  ]
}
```

**错误代码**：
- 400：无效图片数据
- 404：路径不存在
- 405：方法不支持
- 500：推理引擎错误

## 4. 使用示例

### 4.1 CURL调用
```bash
curl -X POST http://192.168.1.100:8080/api/classify \
  -H "Content-Type: image/jpeg" \
  --data-binary @test.jpg
```

### 4.2 Python客户端
```python
import requests

def classify_image(url, image_path):
    with open(image_path, 'rb') as f:
        response = requests.post(url, data=f, headers={
            'Content-Type': 'image/jpeg'
        })
    return response.json()

result = classify_image('http://192.168.1.100:8080/api/classify', 'cat.jpg')
print(result)
```

## 5. 常见问题

**Q1: 返回全零概率**
- 检查模型文件路径
- 验证输入图片解码是否成功
- 查看rknn_init返回值

**Q2: 端口占用错误**
```bash
netstat -tuln | grep 8080
kill -9 <PID>
```

**Q3: 内存不足**
- 减少并发请求数
- 优化图片预处理流程
