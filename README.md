基于DeepSeek-OCR的古迹识别
1. 运行环境
1.1. Python 3.12
1.2. PyTorch 2.6.0
1.3. CUDA 11.8（本DSW平台自带版本为12.4）

2. 运行方法
2.1. 在运行之前请先下载模型文件，并放在项目根目录下。
注意文件不是源文件，而是含有cogfig.json,model.safetensors,preprocessor_config.json,tokenizer.json,special_tokens_map.json等文件的模型权重文件。

下载：
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR

2.2. 配置文件
进入项目后端目录，安装必要依赖：
pip install -r requirements.txt

2.3. 运行代码
示例运行：

进入你的项目backend目录下，在终端1中运行以下命令：
uvicorn app:app --reload

进入frontend目录下，在终端2中运行以下命令：
python -m http.server 3000

3. 访问方式
在浏览器中访问：https://你的服务器IP地址或者云服务的公网IP/3000/
比如，使用阿里云DSW的访问地址为：https://dsw-gateway-cn-shanghai.data.aliyun.com/dsw-******/ide/proxy/3000/

4. 上传文件识别即可


