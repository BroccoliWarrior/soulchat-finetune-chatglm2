#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据转换脚本：将SoulChat2.0数据集转换为ChatGLM2-6b的SFT训练格式
"""

import json
import os
from typing import List, Dict, Any

def convert_messages_to_sft_format(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    将多轮对话转换为单轮SFT格式
    
    Args:
        messages: 原始对话消息列表
        
    Returns:
        转换后的SFT格式数据列表
    """
    sft_samples = []
    
    # 找到第一个user消息的位置
    start_idx = 0
    for i, msg in enumerate(messages):
        if msg.get('role') == 'user':
            start_idx = i
            break
    
    # 从第一个user消息开始，提取user-assistant对话对
    for i in range(start_idx, len(messages) - 1, 2):
        if i + 1 < len(messages):
            user_msg = messages[i]
            assistant_msg = messages[i + 1]
            
            # 确保是user-assistant问答对
            if user_msg.get('role') == 'user' and assistant_msg.get('role') == 'assistant':
                # 构建ChatGLM格式的对话
                round_num = (i - start_idx) // 2 + 1
                prompt = f"[Round {round_num}]\n问：{user_msg['content']}\n答："
                response = assistant_msg['content']
                
                sft_samples.append({
                    "prompt": prompt,
                    "response": response
                })
    
    return sft_samples

def convert_messages_to_multiturn_sft_format(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """
    将多轮对话转换为完整的多轮SFT格式（可选方案）
    
    Args:
        messages: 原始对话消息列表
        
    Returns:
        转换后的完整对话SFT格式
    """
    # 跳过system消息，构建完整对话
    conversation = ""
    round_num = 1
    
    # 找到第一个user消息的位置
    start_idx = 0
    for i, msg in enumerate(messages):
        if msg.get('role') == 'user':
            start_idx = i
            break
    
    # 构建多轮对话
    for i in range(start_idx, len(messages) - 1, 2):
        if i + 1 < len(messages):
            user_msg = messages[i]
            assistant_msg = messages[i + 1]
            
            if user_msg.get('role') == 'user' and assistant_msg.get('role') == 'assistant':
                conversation += f"[Round {round_num}]\n问：{user_msg['content']}\n答：{assistant_msg['content']}\n\n"
                round_num += 1
    
    if conversation:
        # 将最后一轮拆分为prompt和response
        parts = conversation.strip().split('\n\n')
        if parts:
            last_round = parts[-1]
            if "答：" in last_round:
                prompt_part = last_round.split("答：")[0] + "答："
                response_part = last_round.split("答：")[1]
                
                # 前面的轮次作为历史
                history = '\n\n'.join(parts[:-1])
                if history:
                    prompt = history + '\n\n' + prompt_part
                else:
                    prompt = prompt_part
                
                return {
                    "prompt": prompt,
                    "response": response_part
                }
    
    return None

def process_dataset(input_file: str, output_file: str, use_multiturn: bool = False):
    """
    处理数据集文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        use_multiturn: 是否使用多轮对话格式
    """
    print(f"正在处理文件: {input_file}")
    print(f"多轮对话模式: {'是' if use_multiturn else '否'}")
    
    sft_data = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        total_samples = len(data)
        print(f"总样本数: {total_samples}")
        
        for i, item in enumerate(data):
            if i % 1000 == 0:
                print(f"处理进度: {i}/{total_samples}")
                
            messages = item.get('messages', [])
            if messages:
                if use_multiturn:
                    # 多轮对话模式：每个样本生成一个完整对话
                    sft_sample = convert_messages_to_multiturn_sft_format(messages)
                    if sft_sample:
                        sft_data.append(sft_sample)
                else:
                    # 单轮对话模式：每轮生成一个样本
                    sft_samples = convert_messages_to_sft_format(messages)
                    sft_data.extend(sft_samples)
        
        # 保存转换后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)
            
        print(f"转换完成! 输出文件: {output_file}")
        print(f"转换后样本数: {len(sft_data)}")
        
        # 显示样例
        if sft_data:
            print("\n样例数据:")
            print(f"Prompt: {sft_data[0]['prompt'][:200]}...")
            print(f"Response: {sft_data[0]['response'][:100]}...")
            
    except Exception as e:
        print(f"处理文件时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    # 确保输出目录存在
    os.makedirs('data', exist_ok=True)
    
    # 转换训练数据
    train_input = "data/PsyDTCorpus_train_mulit_turn_packing.json"
    train_output = "data/sft_train_data.json"
    
    if os.path.exists(train_input):
        # 使用单轮对话模式（每轮生成一个训练样本）
        process_dataset(train_input, train_output, use_multiturn=False)
        
        # 可选：生成多轮对话版本
        train_output_multiturn = "data/sft_train_data_multiturn.json"
        process_dataset(train_input, train_output_multiturn, use_multiturn=True)
    else:
        print(f"训练文件不存在: {train_input}")
    
    # 转换测试数据
    test_input = "data/PsyDTCorpus_test_single_turn_split.json"
    test_output = "data/sft_test_data.json"
    
    if os.path.exists(test_input):
        process_dataset(test_input, test_output, use_multiturn=False)
    else:
        print(f"测试文件不存在: {test_input}")

if __name__ == "__main__":
    main() 