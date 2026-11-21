# 道德评估数据预处理与二分类训练

提供JSONL数据预处理和二分类评估模型训练界面。

发布单位：NGAI-LLMSafeContent。发布分支：`release/1.0`；更新日期：2026-11-21；自研代码许可：Apache-2.0，第三方保留原许可。

## 功能

- 上传JSONL并选择文本字段、标签字段和正类值
- 转换为ms-swift二分类训练格式并查看类别统计
- 在界面配置基础模型与训练参数，调用原有训练封装

## 安装与快速开始

Python环境及与设备兼容的PyTorch、ms-swift等依赖；完整版本见原仓requirements.txt。训练需要合法基础模型、数据和相应设备。

从公开仓下载同一发布分支：

```bash
git clone --branch release/1.0 https://github.com/NGAI-LLMSafeContent-OpenSource/Project_1_Ethical_Assessment-release.git
cd Project_1_Ethical_Assessment-release
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell改用 .venv\Scripts\Activate.ps1
```

以下命令均以仓库根目录为起点，注释中的外部条件需先准备：

```bash
python -m pip install -r requirements.txt
python app_ui_2.py
```

## 使用与输出

浏览器访问 `http://127.0.0.1:17891`。先上传JSONL并设置字段，再进行数据预处理；训练页填写自己的模型路径与训练参数后启动。处理数据写入 `data/users/processed/`，训练结果写入配置的输出目录。

## 配置与外部依赖

输入每行一个JSON对象，例如 `{"text":"中性示例文本","label":1}`，并准备包含两类的真实数据。历史参考界面与主README明确标为非当前组件的入口未捆绑。模型和用户数据不随源码分发。

真实凭据只在自己的运行环境或平台Secrets中配置，不提交.env、用户上传、模型权重或运行结果。原仓文件中的内网服务地址经发布负责人确认按原样保留，使用者应在本地配置自己的可达服务。

## 验证与已知问题

完成对应依赖准备后运行：

```bash
python tools/release_check.py
```

检查包含发布文件完整性、原仓文件一致性与声明的最小测试。它不会自动下载模型或申请API，也不代表全部真实业务/准确率验收。保留的原仓错误、未运行项及外部依赖见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)。

## 发布版本与文件清单

本版本以原开发仓主分支提交 `a9fe3944a18ad5c8aaae9e571dabda06a8821631` 为源。功能实现与对应原文件保持一致；README、许可说明和发布辅助材料单独登记。当前功能的必要源码、配置、界面与运行资源保留，明确弃用的原型、历史结果和开发缓存不捆绑。

- [逐文件发布清单](RELEASE_FILES.md)
- [原仓文件与Git对象对应关系](SOURCE_ALIGNMENT.json)
- [完整文件SHA-256及权限](RELEASE_MANIFEST.json)，校验表为SHA256SUMS

校验值按Git存储字节生成；若原.gitattributes规定检出换行转换，请用Git对象或发布检查脚本校验，避免将LF/CRLF误认为源码变化。各公开平台release/1.0使用同一个提交，平台生成的运行分支不属于此发布清单。

## 反馈、贡献与许可证

请通过 [公开仓Issues](https://github.com/NGAI-LLMSafeContent-OpenSource/Project_1_Ethical_Assessment-release/issues) 提交复现步骤、环境和日志摘要，勿附真实凭据或用户数据。代码问题由课题维护者在原开发仓处理，发布负责人同步审核后的版本；不要仅修一个平台造成分歧。

自研代码采用[Apache-2.0](LICENSE)。模型、数据集、外部服务和第三方代码遵循各自许可与使用条件，见[第三方说明](THIRD_PARTY_NOTICES.md)。
