#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGLM2-6b LoRA微调模型推理脚本
彻底绕过transformers版本兼容性问题
"""

import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import argparse
import os
import warnings
import sys
warnings.filterwarnings("ignore")

class UltimateSoulChatInferencer:
    """心理疏导对话推理器"""
    
    def __init__(self, base_model_path: str, lora_weights_path: str = None, merged_model_path: str = None):
        """初始化推理器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        if merged_model_path and os.path.exists(merged_model_path):
            # 加载合并后的模型
            print(f"加载合并后的模型: {merged_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                merged_model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            self.model = AutoModel.from_pretrained(
                merged_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # 加载基础模型 + LoRA权重
            print(f"加载基础模型: {base_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            self.model = AutoModel.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            if lora_weights_path and os.path.exists(lora_weights_path):
                print(f"加载LoRA权重: {lora_weights_path}")
                self.model = PeftModel.from_pretrained(self.model, lora_weights_path)
                print("合并LoRA权重到模型...")
                self.model = self.model.merge_and_unload()
        
        self.model.eval()
        print("模型加载完成!")
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 尝试禁用problematic方法
        self._patch_model_methods()
        
        self.system_prompt = (
            "你是一位精通理情行为疗法（Rational Emotive Behavior Therapy，简称REBT）的心理咨询师，"
            "能够合理地采用理情行为疗法给来访者提供专业地指导和支持，缓解来访者的负面情绪和行为反应，"
            "帮助他们实现个人成长和心理健康。\n\n"
            "理情行为治疗主要包括以下几个阶段：\n"
            "（1）检查非理性信念和自我挫败式思维：咨询师帮助来访者探查隐藏在情绪困扰后面的原因，"
            "并反省其内在自我对话。\n"
            "（2）与非理性信念辩论：使用认知技术质疑非理性信念的不合理性，引导来访者愿意放弃它们。\n"
            "（3）得出合理信念，学会理性思维：协助来访者找到更合适的应对方式，并重复教导其合理性。\n"
            "（4）迁移应用治疗收获：鼓励来访者将治疗中的收获应用于日常生活，实现持续成长。"
        )

    
    def _patch_model_methods(self):
        """修补模型方法以避免兼容性问题"""
        try:
            # 如果模型有chat方法，临时禁用它
            if hasattr(self.model, 'chat'):
                self.model._original_chat = self.model.chat
                # 用我们的方法替换
                self.model.chat = self._safe_chat_wrapper
            print("已应用兼容性补丁")
        except Exception as e:
            print(f"应用补丁时出错（可忽略）: {e}")
    
    def _safe_chat_wrapper(self, tokenizer, query, history=None, max_length=2048, **kwargs):
        """安全的chat方法包装器"""
        return self.chat(query), []
    
    def generate_response_direct(self, user_input: str, max_new_tokens: int = 200):
        """
        直接生成回复 - 绕过所有可能的兼容性问题
        """
        try:
            # 构建prompt
            prompt = f"问：{user_input}\n答："
            
            # 编码输入
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = input_ids.to(self.device)
            
            # 创建attention mask
            attention_mask = torch.ones_like(input_ids)
            
            input_length = input_ids.shape[1]
            
            # 手动实现生成逻辑
            generated_ids = input_ids.clone()
            
            with torch.no_grad():
                for step in range(max_new_tokens):
                    # 获取下一个token的logits
                    outputs = self.model(
                        input_ids=generated_ids,
                        attention_mask=torch.ones_like(generated_ids),
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False
                    )
                    
                    # 获取最后一个位置的logits
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # 应用temperature
                    next_token_logits = next_token_logits / 0.7
                    
                    # Top-p采样
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > 0.8
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                    
                    # 采样下一个token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # 添加到序列
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    # 检查是否生成了结束token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            # 解码生成的部分
            new_tokens = generated_ids[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # 清理输出
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            print(f"直接生成失败: {e}")
            raise e
    
    def generate_response_simple(self, user_input: str, max_new_tokens: int = 150):
        """
        简化生成方法 - 最小化兼容性问题
        """
        try:
            # 更简单的prompt
            prompt = user_input
            
            # 编码
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
            input_ids = inputs["input_ids"].to(self.device)
            
            # 确保有attention_mask
            if "attention_mask" in inputs:
                attention_mask = inputs["attention_mask"].to(self.device)
            else:
                attention_mask = torch.ones_like(input_ids)
            
            input_length = input_ids.shape[1]
            
            with torch.no_grad():
                # 使用最基础的forward pass
                for _ in range(max_new_tokens):
                    try:
                        # 直接调用模型forward
                        outputs = self.model.forward(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        
                        # 获取logits
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                        next_token_logits = logits[0, -1, :]
                        
                        # 简单的贪婪解码
                        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        
                        # 检查结束条件
                        if next_token_id.item() == self.tokenizer.eos_token_id:
                            break
                        
                        # 添加新token
                        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
                        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=-1)
                        
                        # 限制长度
                        if input_ids.shape[1] > input_length + max_new_tokens:
                            break
                            
                    except Exception as inner_e:
                        print(f"内部生成错误: {inner_e}")
                        break
            
            # 解码新生成的部分
            new_tokens = input_ids[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return self._clean_response(response)
            
        except Exception as e:
            print(f"简化生成失败: {e}")
            raise e
    
    def _clean_response(self, response: str) -> str:
        """清理响应文本"""
        response = response.strip()
        
        # 移除可能的格式标记
        if "答：" in response:
            response = response.split("答：")[-1].strip()
        if "问：" in response:
            response = response.split("问：")[0].strip()
        
        # 移除重复的文本
        lines = response.split('\n')
        if len(lines) > 1:
            response = lines[0].strip()
        
        return response
    

    
    def chat(self, user_input: str):
        """主要对话方法"""
        print("正在生成回复...", end="", flush=True)
        
        prompt_with_context = f"{self.system_prompt}\n\n来访者：{user_input}\n心理咨询师："
        
        # 尝试多种生成方法
        methods = [
            ("直接生成", self.generate_response_direct),
            ("简化生成", self.generate_response_simple)
        ]
        
        for method_name, method in methods:
            try:
                response = method(prompt_with_context)
                if response and len(response.strip()) > 0:
                    print(f"\r使用了{method_name}")
                    return response
            except Exception as e:
                print(f"\r{method_name}失败: {e}")
                continue
        
        # 如果所有方法都失败了，抛出异常
        raise Exception("所有生成方法都失败了")
    
    def interactive_chat(self):
        """交互式对话"""
        print("\n🤖 欢迎使用心理疏导对话系统!")
        print("💝 我是你的AI心理健康助手，随时倾听你的心声")
        print("📝 输入'退出'、'exit'、'quit'或'bye'结束对话")
        print("🔄 输入'重启'重新开始对话")
        print("=" * 60)
        
        conversation_count = 0
        
        while True:
            try:
                user_input = input(f"\n💬 [{conversation_count + 1}] 你: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['退出', 'exit', 'quit', 'bye']:
                    print("\n🌟 感谢你的信任和分享。")
                    print("💪 记住，寻求帮助是勇敢的表现，你并不孤单。")
                    print("🏥 如果需要专业帮助，建议咨询专业心理健康机构。")
                    print("👋 再见，祝你一切安好！")
                    break
                
                if user_input == '重启':
                    conversation_count = 0
                    print("🔄 对话已重启")
                    continue
                
                # 生成回复
                try:
                    response = self.chat(user_input)
                    print(f"🤖 助手: {response}")
                    conversation_count += 1
                    
                    # 定期提醒
                    if conversation_count % 5 == 0:
                        print("\n💡 温馨提醒：我是AI助手，可以提供情感支持，但如果遇到严重心理困扰，请及时寻求专业心理咨询师的帮助。")
                        
                except Exception as e:
                    print(f"❌ 生成回复失败: {e}")
                    print("🔧 请尝试重新输入或检查网络连接")
                
            except KeyboardInterrupt:
                print("\n\n⚡ 对话被中断。")
                print("🌈 希望我们的交流对你有所帮助。保重！")
                break
            except Exception as e:
                print(f"\n❌ 发生未知错误: {e}")
                print("🔄 让我们继续尝试...")
    
    def batch_test(self):
        """批量测试功能"""
        test_cases = [
            "我最近压力很大，总是失眠",
            "感觉很孤独，没有人真正理解我",
            "工作上遇到了很大的挫折",
            "和家人的关系很紧张",
            "对未来感到非常迷茫和焦虑",
            "最近总是胡思乱想，无法集中注意力",
            "刚刚分手，心情非常低落",
            "学习压力太大，想要放弃",
            "觉得自己一无是处，很自卑",
            "最近情绪波动很大，控制不了自己"
        ]
        
        print("\n🧪 开始批量测试...")
        print("=" * 60)
        
        success_count = 0
        
        for i, question in enumerate(test_cases, 1):
            print(f"\n📋 测试 {i}/{len(test_cases)}")
            print(f"❓ 问题: {question}")
            print("-" * 40)
            
            try:
                response = self.chat(question)
                if response and len(response.strip()) > 10:
                    print(f"✅ 回复: {response}")
                    success_count += 1
                else:
                    print(f"⚠️  回复过短: {response}")
            except Exception as e:
                print(f"❌ 生成失败: {e}")
            
            print("=" * 40)
        
        print(f"\n📊 测试完成: {success_count}/{len(test_cases)} 成功")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AuraChat推理脚本")
    parser.add_argument("--base_model", type=str, default="models/chatglm2-6b", 
                       help="基础模型路径")
    parser.add_argument("--lora_weights", type=str, default="output/lora_weights", 
                       help="LoRA权重路径（可选）")
    parser.add_argument("--merged_model", type=str, default="output/merged_model", 
                       help="合并后模型路径（可选）")
    parser.add_argument("--mode", type=str, default="interactive", 
                       choices=["interactive", "test", "single"], 
                       help="运行模式")
    parser.add_argument("--question", type=str, default="", 
                       help="单次模式下的问题")
    
    args = parser.parse_args()
    
    print("🚀 启动AuraChat...")
    print(f"🎯 运行模式: {args.mode}")
    
    try:
        # 初始化推理器
        inferencer = UltimateSoulChatInferencer(
            base_model_path=args.base_model,
            lora_weights_path=args.lora_weights if os.path.exists(args.lora_weights or "") else None,
            merged_model_path=args.merged_model if os.path.exists(args.merged_model or "") else None
        )
        
        if args.mode == "interactive":
            inferencer.interactive_chat()
        elif args.mode == "test":
            inferencer.batch_test()
        elif args.mode == "single":
            question = args.question or input("请输入你的问题: ").strip()
            if question:
                print(f"\n❓ 问题: {question}")
                response = inferencer.chat(question)
                print(f"🤖 回复: {response}")
            else:
                print("❌ 未提供问题")
        
    except Exception as e:
        print(f"💥 启动失败: {e}")
        print("🔧 请检查模型路径和环境配置")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 