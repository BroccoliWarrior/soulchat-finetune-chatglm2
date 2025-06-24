#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGLM2-6b LoRA微调训练脚本
"""

import os
import json
import torch
import logging
import gc
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from datasets import Dataset

from transformers import (
    AutoTokenizer, 
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU内存已清理")

def find_target_modules(model):
    """自动查找ChatGLM2-6b的目标模块"""
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split('.')
            target_modules.add(names[-1])
    
    # ChatGLM2-6b的常见模块
    common_modules = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h']
    found_modules = [m for m in common_modules if m in target_modules]
    
    logger.info(f"找到的目标模块: {found_modules}")
    logger.info(f"所有Linear模块: {sorted(list(target_modules))}")
    
    return found_modules if found_modules else list(target_modules)[:4]

def main():
    """主训练函数"""
    logger.info("开始初始化训练环境...")
    
    # 设置环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 清理GPU内存
    clear_gpu_memory()
    
    # 检查CUDA
    if not torch.cuda.is_available():
        logger.error("未检测到CUDA设备")
        exit(1)
    
    logger.info(f"检测到CUDA设备: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 设置随机种子
    set_seed(42)
    
    # 创建输出目录
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载tokenizer
    logger.info("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "models/chatglm2-6b",
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载模型
    logger.info("加载模型...")
    model = AutoModel.from_pretrained(
        "models/chatglm2-6b",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # 3. 准备模型进行训练
    logger.info("准备模型进行训练...")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # 关键：启用输入梯度
    
    # 4. 自动查找目标模块
    target_modules = find_target_modules(model)
    
    # 5. 配置LoRA
    logger.info("配置LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 6. 确保模型在训练模式
    model.train()
    
    # 验证可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数数量: {trainable_params:,}")
    
    if trainable_params == 0:
        logger.error("没有可训练参数！")
        exit(1)
    
    clear_gpu_memory()
    
    # 7. 加载数据
    logger.info("加载数据...")
    if not os.path.exists("data/sft_train_data.json"):
        logger.error("数据文件不存在")
        exit(1)
    
    with open("data/sft_train_data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 限制数据量
    data = data[:5000]  # 进一步减少到5000
    logger.info(f"使用数据量: {len(data)}")
    
    dataset = Dataset.from_list(data)
    
    # 8. 数据预处理
    def preprocess_function(examples):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for i in range(len(examples['prompt'])):
            prompt = examples['prompt'][i]
            response = examples['response'][i]
            
            # 限制长度
            if len(prompt) > 400:
                prompt = prompt[:400]
            if len(response) > 200:
                response = response[:200]
            
            input_text = prompt + response
            
            # Tokenize
            inputs = tokenizer(
                input_text,
                max_length=400,  # 减少到400
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            prompt_inputs = tokenizer(
                prompt,
                max_length=300,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            prompt_length = len(prompt_inputs["input_ids"])
            labels = [-100] * prompt_length + inputs["input_ids"][prompt_length:]
            
            if len(labels) > len(inputs["input_ids"]):
                labels = labels[:len(inputs["input_ids"])]
            elif len(labels) < len(inputs["input_ids"]):
                labels += [-100] * (len(inputs["input_ids"]) - len(labels))
            
            model_inputs["input_ids"].append(inputs["input_ids"])
            model_inputs["attention_mask"].append(inputs["attention_mask"])
            model_inputs["labels"].append(labels)
        
        return model_inputs
    
    logger.info("预处理数据...")
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names
    )
    
    # 9. 数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
    
    # 10. 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,  # 提高学习率
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        logging_dir=f"{output_dir}/logs"
    )
    
    # 11. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 12. 开始训练
    logger.info("开始训练...")
    try:
        # 训练前验证
        logger.info("验证模型设置...")
        sample_batch = next(iter(trainer.get_train_dataloader()))
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in sample_batch.items()})
        
        model.train()
        logger.info("模型验证通过，开始训练...")
        
        trainer.train()
        logger.info("训练完成！")
        
        # 保存模型
        trainer.save_model()
        model.save_pretrained(f"{output_dir}/lora_weights")
        tokenizer.save_pretrained(f"{output_dir}/lora_weights")
        logger.info(f"模型已保存到 {output_dir}/")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试保存当前状态
        if hasattr(trainer, 'state') and trainer.state.global_step > 0:
            trainer.save_model(f"{output_dir}/emergency_checkpoint")
            logger.info("紧急保存完成")
    
    clear_gpu_memory()
    logger.info("训练结束")

if __name__ == "__main__":
    main() 