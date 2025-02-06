# 图像分类服务器文档

## 1. 概述
基于RKNN的HTTP图像分类服务，提供RESTful API接口。使用MobileNet模型实现高效推理，适用于嵌入式设备部署。

## 2. 技术特性
- **推理框架**：RKNN Runtime 1.7.0
- **模型支持**：MobileNet v2 (输入224x224 RGB)
- **协议支持**：HTTP/1.1
- **平均延迟**：<20ms (RV1126)，首次推理会慢一些
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
  "class": 1,
  "probability": 0.6593
}
```

**错误代码**：
405 请求方法错误
500 服务器内部错误

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



# 给同事的使用说明

## 快速启动
1. 启动服务器：
````bash
cd ~/mobilenet
./atk_mobilenet_object_classification
````

服务器将在本地8080端口启动，等待分类请求。

2. 测试服务：
````bash
# 在另一个终端中运行
cd ~/mobilenet
./http_mini_client 11.jpg
````

如果一切正常，您将看到类似以下的分类结果：
````json
{"class":1,"probability":0.6593}
````
其中：
- `class`: 分类结果（0或1）
- `probability`: 对应类别的概率值（0-1之间）

## 注意事项
- 确保服务器和测试客户端在同一目录下运行
- 服务器启动后会一直运行，使用 Ctrl+C 可以终止
- 如果8080端口被占用，请先结束占用进程
- 测试客户端支持任意JPEG图片，只需替换文件路径即可

## 问题排查
如果测试失败，请检查：
1. 服务器是否正常启动（无报错信息）
2. 8080端口是否被占用 `netstat -tuln | grep 8080`
3. 测试图片是否存在且为JPEG格式
