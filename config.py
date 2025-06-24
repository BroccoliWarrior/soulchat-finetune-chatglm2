#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoulChat ChatGLM2-6b LoRA微调配置文件
统一管理所有训练和推理参数
"""

import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelConfig:
    """模型相关配置"""
    # 基础模型路径
    base_model_path: str = "models/chatglm2-6b"
    # 是否信任远程代码
    trust_remote_code: bool = True
    # 模型精度
    torch_dtype: str = "float16"  # float16, bfloat16, float32
    # 设备映射
    device_map: str = "auto"

@dataclass
class LoRAConfig:
    """LoRA微调配置"""
    # LoRA rank
    r: int = 8
    # LoRA alpha参数
    alpha: int = 32
    # Dropout率
    dropout: float = 0.1
    # 目标模块
    target_modules: List[str] = None
    # 偏置参数
    bias: str = "none"
    # 任务类型
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            # ChatGLM2-6b的默认目标模块
            self.target_modules = [
                "query_key_value",
                "dense",
                "dense_h_to_4h", 
                "dense_4h_to_h"
            ]

@dataclass
class DataConfig:
    """数据相关配置"""
    # 原始训练数据路径
    raw_train_data_path: str = "data/PsyDTCorpus_train_mulit_turn_packing.json"
    # 原始测试数据路径
    raw_test_data_path: str = "data/PsyDTCorpus_test_single_turn_split.json"
    # 处理后的训练数据路径
    processed_train_data_path: str = "data/sft_train_data.json"
    # 处理后的测试数据路径
    processed_test_data_path: str = "data/sft_test_data.json"
    # 最大输入长度
    max_source_length: int = 512
    # 最大输出长度
    max_target_length: int = 512
    # 最大序列长度
    max_seq_length: int = 1024

@dataclass
class TrainingConfig:
    """训练相关配置"""
    # 输出目录
    output_dir: str = "output"
    # 每设备批次大小
    per_device_train_batch_size: int = 1
    # 梯度累积步数
    gradient_accumulation_steps: int = 4
    # 学习率
    learning_rate: float = 2e-5
    # 训练轮数
    num_train_epochs: int = 3
    # 学习率调度器
    lr_scheduler_type: str = "cosine"
    # 预热比例
    warmup_ratio: float = 0.1
    # 日志步数
    logging_steps: int = 10
    # 保存步数
    save_steps: int = 500
    # 最大保存检查点数
    save_total_limit: int = 3
    # 是否使用fp16
    fp16: bool = True
    # 是否使用bf16
    bf16: bool = False
    # 梯度检查点
    gradient_checkpointing: bool = True
    # 数据加载器固定内存
    dataloader_pin_memory: bool = False
    # 移除未使用的列
    remove_unused_columns: bool = False
    # 报告工具
    report_to: str = "tensorboard"
    # 随机种子
    seed: int = 42

@dataclass
class InferenceConfig:
    """推理相关配置"""
    # 最大生成长度
    max_length: int = 2048
    # 温度参数
    temperature: float = 0.7
    # Top-p采样
    top_p: float = 0.8
    # Top-k采样
    top_k: int = 50
    # 是否采样
    do_sample: bool = True
    # 重复惩罚
    repetition_penalty: float = 1.1

@dataclass
class PathConfig:
    """路径相关配置"""
    # LoRA权重保存路径
    lora_weights_path: str = "output/lora_weights"
    # 合并模型保存路径
    merged_model_path: str = "output/merged_model"
    # 训练日志路径
    logs_path: str = "output/logs"
    # 检查点路径
    checkpoint_path: str = "output/checkpoints"

class Config:
    """主配置类"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.lora = LoRAConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.paths = PathConfig()
        
        # 创建必要的目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.training.output_dir,
            self.paths.lora_weights_path,
            self.paths.merged_model_path,
            self.paths.logs_path,
            self.paths.checkpoint_path,
            "data"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_training_args_dict(self):
        """获取训练参数字典"""
        return {
            "output_dir": self.training.output_dir,
            "per_device_train_batch_size": self.training.per_device_train_batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "learning_rate": self.training.learning_rate,
            "num_train_epochs": self.training.num_train_epochs,
            "lr_scheduler_type": self.training.lr_scheduler_type,
            "warmup_ratio": self.training.warmup_ratio,
            "logging_steps": self.training.logging_steps,
            "save_steps": self.training.save_steps,
            "save_total_limit": self.training.save_total_limit,
            "fp16": self.training.fp16,
            "bf16": self.training.bf16,
            "gradient_checkpointing": self.training.gradient_checkpointing,
            "dataloader_pin_memory": self.training.dataloader_pin_memory,
            "remove_unused_columns": self.training.remove_unused_columns,
            "report_to": self.training.report_to,
            "logging_dir": self.paths.logs_path,
            "seed": self.training.seed
        }
    
    def get_lora_config_dict(self):
        """获取LoRA配置字典"""
        return {
            "r": self.lora.r,
            "lora_alpha": self.lora.alpha,
            "lora_dropout": self.lora.dropout,
            "target_modules": self.lora.target_modules,
            "bias": self.lora.bias,
            "task_type": self.lora.task_type
        }
    
    def print_config(self):
        """打印配置信息"""
        print("=" * 60)
        print("SoulChat ChatGLM2-6b LoRA微调配置")
        print("=" * 60)
        
        print(f"\n📂 模型配置:")
        print(f"  - 基础模型: {self.model.base_model_path}")
        print(f"  - 精度: {self.model.torch_dtype}")
        print(f"  - 设备映射: {self.model.device_map}")
        
        print(f"\n🔧 LoRA配置:")
        print(f"  - Rank: {self.lora.r}")
        print(f"  - Alpha: {self.lora.alpha}")
        print(f"  - Dropout: {self.lora.dropout}")
        print(f"  - 目标模块: {', '.join(self.lora.target_modules)}")
        
        print(f"\n📊 数据配置:")
        print(f"  - 最大输入长度: {self.data.max_source_length}")
        print(f"  - 最大输出长度: {self.data.max_target_length}")
        print(f"  - 训练数据: {self.data.processed_train_data_path}")
        
        print(f"\n🎯 训练配置:")
        print(f"  - 批次大小: {self.training.per_device_train_batch_size}")
        print(f"  - 梯度累积: {self.training.gradient_accumulation_steps}")
        print(f"  - 学习率: {self.training.learning_rate}")
        print(f"  - 训练轮数: {self.training.num_train_epochs}")
        print(f"  - 精度: {'fp16' if self.training.fp16 else 'fp32'}")
        
        print(f"\n🚀 推理配置:")
        print(f"  - 最大长度: {self.inference.max_length}")
        print(f"  - 温度: {self.inference.temperature}")
        print(f"  - Top-p: {self.inference.top_p}")
        
        print("=" * 60)

# 全局配置实例
config = Config()

# 测试用例配置
TEST_CASES = [
    "我最近很焦虑，不知道怎么办",
    "我感觉很孤独，没有人理解我", 
    "工作压力很大，每天都很累",
    "我和朋友吵架了，心情很不好",
    "我对未来感到迷茫和不安",
    "失恋了，心里很痛苦",
    "学习成绩不好，父母很失望",
    "我觉得自己一无是处",
    "最近睡眠质量很差",
    "我很担心家人的健康状况"
]

if __name__ == "__main__":
    # 打印配置信息
    config.print_config() 