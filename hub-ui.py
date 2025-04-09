import gradio as gr
import random
import os
import logging
from typing import List, Optional, Dict, Any

logging.getLogger().setLevel(logging.INFO)

from tools.hub_client import request_infer, request_svc
from tools.spark_client import request_spark_tts
from tools.rvc_client import refresh_resources, convert_audio, change_voice
from tools.funasr import auto_asr
from gradio.processing_utils import PUBLIC_HOSTNAME_WHITELIST

# 添加本地主机到白名单
PUBLIC_HOSTNAME_WHITELIST.extend(["ttd-stage", "localhost", "127.0.0.1"])


def generate_seed() -> Dict[str, Any]:
    """生成随机种子"""
    return {"__type__": "update", "value": random.randint(1, 100000000)}


def set_download_url(file: Dict[str, Any]) -> Optional[str]:
    os.rename(file.get("path"), f"{file.get('path')}.wav")
    return f"{file.get('url')}.wav"

def handle_init_rvc_resources():
    """初始化RVC资源"""
    models, indices = refresh_resources()
    return models, indices, indices

def find_matching_index(model_name: str, indices: List[str] = None) -> str:
    """查找匹配的索引"""
    if not model_name or not indices:
        return ""
    base_name = os.path.splitext(model_name)[0]
    return next((index for index in indices if base_name in index), "")


def infer(
    text: str, seed: int, ref_file: str, asr: str, instruct: str, 
    speed: str = "1", mode: int = 0, model: str = "cosy"
) -> Optional[bytes]:
    """推理生成音频"""
    try:
        # 获取音频数据
        return request_infer(text, seed, ref_file, asr, instruct, speed, mode) if model == "cosy" else request_spark_tts(text, ref_file, asr)
    except Exception as e:
        logging.error(f"音频生成失败: {e}")
        return None


with gr.Blocks(title="八百鹦") as hub:
    indices_state = gr.State([])
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 鹦鹉/八哥推理")
            with gr.Row():
                model = gr.Radio([("鹦鹉", "cosy"), ("八哥", "spark")], label="选择模型", value="cosy", scale=3)
                mode = gr.JSON(0, visible=False) # 默认Zero-Shot
                # mode = gr.Dropdown(
                #     choices=(["3s极速复刻", "0"], ["跨语种复刻", "1"], ["自然语言处理", "2"]), 
                #     label='选择推理模式', value="0", scale=3
                # )                
                speed = gr.Number(value=1, label="速度调节", minimum=0.5, maximum=2.0, step=0.1, scale=3, min_width=80)
                seed = gr.Number(value=0, label="随机推理种子", scale=3, min_width=80)
                seed_button = gr.Button(value="\U0001F3B2", scale=1, min_width=40)

            ref_file = gr.Audio(sources='upload', type='filepath', label='选择参考音频文件，注意采样率不低于16khz')
            asr_text = gr.Textbox(label="输入ASR文本", lines=1, value="")
            instruct_text = gr.Text(label="Instruct文本参考", lines=2, value="<strong> </strong> ， <laughter> </laughter>  \n \
[noise] [breath] [laughter] [cough] [clucking] [accent] [quick_breath] \n \
[hissing] [sigh] [vocalized-noise] [lipsmack] [mn]")
            tts_text = gr.Textbox(label="输入合成文本", lines=1, value="")
            with gr.Row():
                generate_button = gr.Button("生成音频", variant="primary")
                download_button = gr.DownloadButton(label="\U0001F4BE 下载", scale=0)
            cosy_audio = gr.Audio(label="合成后的语音", autoplay=True, streaming=False, interactive=False, format="wav", show_download_button=False)
            seed_button.click(generate_seed, outputs=[seed]).then(
                infer, inputs=[tts_text, seed, ref_file, asr_text, instruct_text, speed, mode, model], outputs=[cosy_audio]
            )
            generate_button.click(
                infer, inputs=[tts_text, seed, ref_file, asr_text, instruct_text, speed, mode, model], outputs=[cosy_audio]
            )
            cosy_audio.change(
                set_download_url, cosy_audio, download_button, preprocess=False
            )
            from tools.funasr import auto_asr
            ref_file.change(auto_asr, ref_file, asr_text)

        with gr.Column():
            gr.Markdown("## 百灵")

            with gr.Row():
                src_audio = gr.Audio(sources='upload', type='filepath', label='选择输入音频文件，注意采样率不低于16khz',scale=7)
                gr.Button("导入推理结果", variant="secondary",scale=3).click(
                    lambda x: x,
                    inputs=[download_button],
                    outputs=[src_audio]
                )

            with gr.Row():
                ref_audio = gr.Audio(sources='upload', type='filepath', label='选择参考音频文件，注意采样率不低于16khz',scale=7)
                svc_load_rvc_audio_button = gr.Button("导入RVC结果", variant="secondary",scale=3)

            with gr.Row():
                f0_adjust = gr.Checkbox(label="手动F0调整", value=False, scale=2, min_width=80)
                pitch_shift = gr.Slider(label='音调变换', minimum=-24, maximum=24, step=1, value=0, scale=4, min_width=80)
                length_adjust = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="语速调整", scale=4, min_width=80)
                steps = gr.Number(value=50, label="扩散步数", minimum=1, maximum=100, step=1, scale=2, min_width=80)

            with gr.Row():
                generate_button = gr.Button("生成音频", variant="primary")
                svc_download_button = gr.DownloadButton(label="\U0001F4BE 下载", scale=0)

            svc_audio = gr.Audio(label="合成后的语音", autoplay=True, streaming=False, interactive=False, format="wav", show_download_button=False)
            generate_button.click(
                request_svc, inputs=[src_audio, ref_audio, steps, f0_adjust, pitch_shift, length_adjust], outputs=[svc_audio]
            )
            svc_audio.change(
                set_download_url, svc_audio, svc_download_button, preprocess=False
            )

        with gr.Column():
            gr.Markdown("## RVC")

            # RVC模型选择
            with gr.Row():
                # 刷新按钮
                rvc_refresh_button = gr.Button("刷新RVC音色库", variant="secondary")
                rvc_models = gr.Dropdown(label="音色模型", choices=[], interactive=True)
                rvc_indices = gr.Dropdown(label="特征索引", choices=[], interactive=True)

                rvc_refresh_button.click(
                    handle_init_rvc_resources,
                    outputs=[rvc_models, rvc_indices, indices_state]
                )
                rvc_models.change(
                    find_matching_index,
                    inputs=[rvc_models, indices_state],
                    outputs=rvc_indices
                ).then(
                    lambda x: change_voice(x) and None,  # 调用 change_voice 但忽略返回值
                    inputs=[rvc_models],
                )

            with gr.Row():
                # RVC输入源
                rvc_src_audio = gr.Audio(sources='upload', type='filepath', label='选择输入音频文件，注意采样率不低于16khz', scale=7, interactive=True)
                gr.Button("导入推理结果", variant="secondary",scale=3).click(
                    lambda x: x,
                    inputs=[download_button],
                    outputs=[rvc_src_audio]
                )

            with gr.Row():
                rvc_generate_button = gr.Button("生成RVC输出", variant="primary")
                rvc_download_button = gr.DownloadButton(label="\U0001F4BE 下载", scale=0)

            with gr.Row():
                # RVC生成的音频
                rvc_audio = gr.Audio(label="RVC输出", type="filepath", interactive=False, autoplay=True, show_download_button=False)
                rvc_audio.change(set_download_url, rvc_audio, rvc_download_button, preprocess=False)
            
            #RVC 设置
            with gr.Row():
                # 音高偏移设置
                pitch_shift = gr.Slider(label="变调 音高偏移（半音数）", minimum=-12, maximum=12, value=0, step=1, interactive=True)
                # 响度标准化
                loudnorm = gr.Slider(label="loudnorm到指定的LUFS（0为不调整）", minimum=-40, maximum=-10, value=-26, step=1, interactive=True)
                # 重采样采样率
                resample_sr = gr.Slider(label="重采样采样率", minimum=0, maximum=48000, value=48000, step=100, interactive=True)
                # RMS混合率
                rms_mix_rate = gr.Slider(label="输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络", minimum=0, maximum=1, value=1, step=0.01, interactive=True)
                # 保护清辅音和咽音
                protect_option = gr.Slider(label="保护清辅音和咽音，防止出现artifact，启用会牺牲转换度", minimum=0, maximum=0.5, value=0.33, step=0.01, interactive=True)
                # 滤波器半径
                filter_radius = gr.Slider(label=">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音", minimum=0, maximum=7, value=3, step=1, interactive=True)
                # 检索特征占比
                index_rate = gr.Slider(label="检索特征占比", minimum=0, maximum=1, value=1, step=0.01, interactive=True)

            rvc_generate_button.click(
                lambda src_audio, pitch_shift, indices, index_rate, filter_radius, resample_sr, rms_mix_rate, protect, loudnorm: convert_audio(
                    input_audio_path=src_audio,
                    f0_up_key=pitch_shift,
                    file_index2=indices,
                    index_rate=index_rate,
                    filter_radius=filter_radius,
                    resample_sr=resample_sr,
                    rms_mix_rate=rms_mix_rate,
                    protect=protect,
                    loudnorm=loudnorm
                ),
                inputs=[rvc_src_audio, pitch_shift, rvc_indices, index_rate, filter_radius, resample_sr, rms_mix_rate, protect_option, loudnorm],
                outputs=rvc_audio
            )
            
            svc_load_rvc_audio_button.click(
                lambda x: x,
                inputs=[rvc_download_button],
                outputs=[ref_audio]
            )

        hub.load(
            lambda: handle_init_rvc_resources(),
            outputs=[rvc_models, rvc_indices, indices_state]
        )

hub.launch(server_name='0.0.0.0', server_port=5650)
