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
Content-Type: application/json
Content-Length: <json_size>

{
  "image": "<base64_encoded_image>",
  "features": [<feature1>, <feature2>, ..., <feature34>]
}
```

**请求参数**：
- image: Base64编码的JPEG图片
- features: 34个浮点数特征值（服务器会自动进行标准化处理）
- 图片格式：JPEG
- 推荐分辨率：224x224
- 最大文件大小：未测试，推荐分辨率为224x224

**成功响应**：
```json
{
  "class": 1,
  "probability": 0.6593,
  "blood_score": 0.7226,
  "rknn_score": 0.6593
}
```

**错误代码**：
405 请求方法错误
500 服务器内部错误

## 4. 使用示例

### 4.1 CURL调用
```bash
curl -X POST http://192.168.5.222:8080/api/classify \
  -H "Content-Type: application/json" \
  -d '{"image":"'$(base64 -w0 test.jpg)'",
    "features":[5,1,0,1,10.27,4.59,131,38.9,84.7,28.5,337,0.01,0.1,394,39.3,12.8,8.9,9,16.9,0.36,7.09,2.5,0.59,0.05,0.04,69.1,24.3,5.7,0.5,0.4,0.07,0.7,68.4,7]}'
```

### 4.2 Python客户端
```python
import requests
import json
import base64

url = "http://192.168.5.222:8080/api/classify"

with open("test.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

data = {
    "image": encoded_string,
    "features": [5,1,0,1,10.27,4.59,131,38.9,84.7,28.5,
                337,0.01,0.1,394,39.3,12.8,8.9,9,16.9,0.36,
                7.09,2.5,0.59,0.05,0.04,69.1,24.3,5.7,0.5,0.4,
                0.07,0.7,68.4,7]
}

headers = {'Content-Type': 'application/json'}
response = requests.post(
    url, data=json.dumps(data, separators=(",", ":")), headers=headers
)
print(response.json())
```

## 5. 常见问题

**Q1: 返回全零概率**
- 检查模型文件路径
- 验证输入图片base64编解码是否成功
- 确认特征数据格式正确（34个浮点数）
- JSON 格式要求：
  - JSON 字符串中不能包含不必要的空格或换行符
  - Python客户端推荐使用 `json.dumps(data, separators=(',', ':'))` 确保格式正确

**Q2: Invalid JSON: missing image field**
- 请检查输入图片路径是否正确
- 请检查图片base64编码是否成功
- 请检查image字段后是否存在多余空格或回车

**Q3: Invalid features: expected 34 features**
- 请检查特征数量是否正确
- 请检查features字段后是否存在多余空格或回车

**Q4: 端口占用错误**
```bash
netstat -tuln | grep 8080
kill -9 <PID>
```


# 给同事的使用说明

在rv1126开发板`/our_project`目录下保存了服务器程序与测试程序，可以使用测试客户端`http_mini_client`验证服务器是否正常工作

## 快速启动
1. 启动服务器：
````bash
cd /our_project
./atk_mobilenet_object_classification
````

服务器将在本地8080端口启动，等待分类请求。

2. 测试服务：
````bash
# 在另一个终端中运行
cd /our_project
./http_mini_client 11.jpg
````

如果一切正常，您将看到类似以下的分类结果：
````json
{
  "class": 1,
  "probability": 0.6593,
  "blood_score": 0.7226,
  "rknn_score": 0.6593
}
````
其中：
- `class`: 分类结果（0或1）表示细菌性肺炎的阳性与否，1为阳性
- `probability`: 对应类别的概率值（0-1之间）

## 注意事项
- 确保服务器和测试客户端在同一目录下运行
- 服务器启动后会一直运行，使用 Ctrl+C 可以终止
- 如果8080端口被占用，请先结束占用进程
- 测试客户端支持任意JPEG图片，只需替换文件路径即可，但图片过大可能导致编解码失败，推荐分辨率为224x224
- JSON 格式要求：
  - `image` 字段必须紧跟在冒号后面，不能有空格
  - JSON 字符串中不能包含不必要的空格或换行符
  - Python客户端推荐使用 `json.dumps(data, separators=(',', ':'))` 确保格式正确

## 问题排查
如果测试失败，请检查：
1. 服务器是否正常启动（无报错信息）
2. 8080端口是否被占用 `netstat -tuln | grep 8080`
3. 测试图片是否存在且为JPEG格式
