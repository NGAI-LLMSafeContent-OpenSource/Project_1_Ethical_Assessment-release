# 道德评估数据预处理与二分类训练组件

发布方 NGAI-LLMSafeContent；版本 1.0；日期 2026-11-04；Apache-2.0。
源仓库 NGAI-LLMSafeContent/Project_1_Ethical_Assessment，固定提交 a9fe3944a18ad5c8aaae9e571dabda06a8821631。

此最小版本提供 JSONL 上传、二分类数据预处理与训练配置界面，保留 UI 实际导入的 do_detox.py/modules 源码闭包；不公开用户上传、原始训练数据、模型和训练输出。历史去害界面与未使用评估入口未纳入。

## 前置条件与安装

Python >=3.12、pip >=22.3；首次需联网访问 PyPI 与官方 PyTorch CPU wheel 源，下载体积较大。建议新建 venv，根目录运行：

```sh
python -m pip install -r requirements.txt
python app_ui_2.py
```

浏览器打开 http://127.0.0.1:17891 。APP_HOST/APP_PORT 可调整绑定地址与端口。关键词字段按上传 JSONL 配置；examples/binary.example.jsonl 是三条新编写的中性样例，标签仅演示二分类格式，不是真实道德标签。输入字段 text，标签字段 label，正类值 1。

## 训练另需准备

二分类训练需额外安装 requirements-training.txt，并自行准备有权使用的基础模型、实际训练数据和适配设备。默认 UI 本地模型路径 models/Qwen2.5-0.5B-Instruct 未随包提供，应替换为自己的路径。CPU requirements 用于最小 UI/预处理验证；真实 GPU 训练应按设备安装适配 PyTorch，并验证 ms-swift 与模型兼容性。模型获取版本、真实训练和训练效果未完成验收，不把预处理测试当作训练通过。

`python tools/release_smoke.py` 自动创建隔离环境安装 CPU 依赖，导入真实界面与业务源码，使用中性样例验证 JSONL 字段转换、类别统计和单类拒绝，再实际启动 Gradio HTTP 页面。测试不下载或加载模型，不执行训练/去害。用户数据与输出保留本地，不提交版本库。

第三方条件见 THIRD_PARTY_NOTICES.md；文件、固定源提交和校验见 RELEASE_MANIFEST.json、SHA256SUMS。
