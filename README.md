SoulChat Fine-Tune (ChatGLM2)
基于 ChatGLM2-6B 的中文心理疏导对话微调项目，使用 LoRA 高效训练，支持命令行与 Web 推理。

🔧 环境配置
建议使用 Conda 环境：
conda env create -f environment.yaml
conda activate soulchat

🚀 快速开始
# 训练模型
python train.py

# 启动 Web UI（Gradio）
python web_ui.py
