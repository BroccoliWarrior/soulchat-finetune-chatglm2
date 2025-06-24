import gradio as gr
from infer import UltimateSoulChatInferencer

# 初始化推理器
inferencer = UltimateSoulChatInferencer(
    base_model_path="models/chatglm2-6b",
    lora_weights_path="output/lora_weights"
)

# 聊天函数
def chat_fn(message, history):
    response = inferencer.chat(message)
    return "", history + [[message, response]]

# 自定义气泡样式和头像
custom_css = """
.gradio-container {
    background-color: #fefefe;
    font-family: 'Helvetica Neue', sans-serif;
}
.chatbot .message.user {
    background-color: #d8eafd !important;
    border-radius: 10px 10px 0px 10px;
    color: #333;
}
.chatbot .message.bot {
    background-color: #ffeef4 !important;
    border-radius: 10px 10px 10px 0px;
    color: #333;
}
.chatbot .avatar.user {
    background-image: url('https://cdn-icons-png.flaticon.com/512/847/847969.png');
    background-size: cover;
}
.chatbot .avatar.bot {
    background-image: url('https://cdn-icons-png.flaticon.com/512/4333/4333609.png');
    background-size: cover;
}
"""

# 构建 Gradio UI
with gr.Blocks(css=custom_css, title="心语AI") as demo:
    gr.Markdown("## 🌸 心语 AI - 你的温柔倾诉对象")
    
    chatbot = gr.Chatbot(
        label="AI 心理助手",
        height=800,
        avatar_images=("image/user.png", "image/robot.png")
    )

    with gr.Row():
        txt = gr.Textbox(placeholder="和我聊聊你的烦恼吧...", show_label=False, scale=8)
        submit_btn = gr.Button("发送", scale=1)

    def respond(message, history):
        return chat_fn(message, history)

    submit_btn.click(respond, [txt, chatbot], [txt, chatbot])
    txt.submit(respond, [txt, chatbot], [txt, chatbot])

# 使用 share=True 避免 localhost 报错，指定端口避免冲突
demo.launch(share=True)
