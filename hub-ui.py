

import gradio as gr
import random

from tools.hub_client import request_svc, request_infer


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

with gr.Blocks() as hub:

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 鹦鹉独立版")
            with gr.Row():
                speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1, scale=4)
                seed = gr.Number(value=0, label="随机推理种子", scale=4)
                seed_button = gr.Button(value="\U0001F3B2", scale=2)

            ref_file = gr.Audio(sources='upload', type='filepath', label='选择参考音频文件，注意采样率不低于16khz')
            asr_text = gr.Textbox(label="输入ASR文本", lines=1, value="")
            tts_text = gr.Textbox(label="输入合成文本", lines=1, value="")

            generate_button = gr.Button("生成音频", variant="primary")
            cosy_audio = gr.Audio(label="合成后的语音", autoplay=True, streaming=False, format="wav")
            seed_button.click(generate_seed, outputs=[seed]).then(
                request_infer, inputs=[tts_text, seed, ref_file, asr_text, speed], outputs=[cosy_audio]
            )
            generate_button.click(
                request_infer, inputs=[tts_text, seed, ref_file, asr_text, speed], outputs=[cosy_audio]
            )
            from tools.funasr import auto_asr
            ref_file.change(auto_asr, ref_file, asr_text)

        with gr.Column():
            gr.Markdown("## 百灵独立版")

            with gr.Row():
                src_audio = gr.Audio(sources='upload', type='filepath', label='选择输入音频文件，注意采样率不低于16khz')
            with gr.Row():
                ref_audio = gr.Audio(sources='upload', type='filepath', label='选择参考音频文件，注意采样率不低于16khz')
            steps = gr.Number(value=50, label="扩散步数", minimum=1, maximum=100, step=1)

            generate_button = gr.Button("生成音频", variant="primary")
            svc_audio = gr.Audio(label="合成后的语音", autoplay=True, streaming=False, format="wav")
            generate_button.click(
                request_svc, inputs=[src_audio, ref_audio, steps], outputs=[svc_audio]
            )

hub.launch(server_name='0.0.0.0', server_port=5650)
