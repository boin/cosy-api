# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import io
import logging
import os
import tempfile
import random
import sys
import wave
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

logging.getLogger().setLevel(logging.INFO)
logging.getLogger("multipart").setLevel(logging.ERROR)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model_info(actor_uuid: str, voice_uuid: str) -> (bytes | None, str | None):
    model_root = "{}/data/links".format(ROOT_DIR)
    prompt_wav: Path = Path(model_root) / actor_uuid / "train" / f"{voice_uuid}.wav"
    if not prompt_wav.exists():
        raise FileNotFoundError(f"Prompt wav {prompt_wav} not found")
    return prompt_wav, voice_uuid.split("_")[-1]


def normalize_sound(audio, sr):
    """
    Normalize the loudness of an audio signal to -25 LUFS.
    Args:
        audio (numpy.ndarray): The input audio signal (float32 expected).
        sr (int): The sample rate of the audio signal.
    Returns:
        numpy.ndarray: The loudness-normalized audio signal (float32).
    """
    try:
        meter = pyln.Meter(sr)  # ITU-R BS.1770 meter
        loudness = meter.integrated_loudness(audio)
        logging.info(f"Original loudness: {loudness:.2f} LUFS, normalize to target: -23 LUFS")
        # Normalize to -23 LUFS
        normalized_audio = pyln.normalize.loudness(audio, loudness, -23.0)
        return normalized_audio
    except Exception as e:
        logging.warning(f"Failed to normalize audio {e}, return original audio")
        return audio


def generate_data(model_output, post_process=True):
    """
    Generate a WAV audio file from model output with loudness normalization.
    Args:
        model_output (list): A list of dictionaries containing audio chunks.
    Returns:
        io.BytesIO: A memory buffer containing the WAV audio file.
    """
    tts_speeches = []
    for i in model_output:
        tts_speeches.append(i["tts_speech"])
    tts_audio = torch.concat(tts_speeches, dim=1).numpy().flatten()
    if post_process:
        tts_audio = normalize_sound(tts_audio, sr=cosyvoice.sample_rate)
    # Convert back to int16
    int16_audio = (tts_audio * (2**15)).astype(np.int16)
    # Write to WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(cosyvoice.sample_rate)  # 24000 kHz sample rate
        wav_file.writeframes(int16_audio.tobytes())
    # Reset buffer position
    buffer.seek(0)
    return buffer


@app.post("/infer")
async def inference(
    text: str = Form(),
    rseed: str = Form(),
    voice_uuid: str = Form(),
    actor_uuid: str = Form(),
):
    # re-seed
    seed = int(rseed)
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        prompt_wav, prompt_text = get_model_info(actor_uuid, voice_uuid)
        prompt_speech_16k = load_wav(prompt_wav, 16000)
        model_output = cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k)

        return StreamingResponse(
            generate_data(model_output),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.post("/zero_shot_infer")
async def zero_shot(
    mode: str = Form("0"),
    text: str = Form(),
    ref_file: UploadFile = File(...),
    asr: str = Form(""),
    instruct: str = Form(""),
    rseed: str = Form(),
    speed: str = Form(),
    post_process: bool = Form("True"),
):
    if rseed != "":
        seed = int(rseed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    speed = float(speed)
    
    # 保存上传的文件到临时目录
    
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, ref_file.filename)
    
    try:
        # 保存上传的文件
        contents = await ref_file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(contents)
        
        # 处理音频文件
        prompt_speech_16k = load_wav(temp_file_path, 16000)
        match mode:
            case "0":
                logging.info(f"Zero-shot with asr {asr}")
                model_output = cosyvoice.inference_zero_shot(text, asr, prompt_speech_16k, speed=speed)
            case "1":
                logging.info(f"Cross-lingual mode")
                model_output = cosyvoice.inference_cross_lingual(text, prompt_speech_16k, speed=speed)
            case "2":
                logging.info(f"Instruct2 with instruct {instruct}")
                model_output = cosyvoice.inference_instruct2(text, instruct, prompt_speech_16k, speed=speed)
            case _:
                raise ValueError(f"mode {mode} (type {type(mode)}) is not supported")

        return StreamingResponse(
            generate_data(model_output, post_process),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    #cosyvoice = CosyVoice('data/pretrained_models/CosyVoice-300M-25Hz', load_jit=True, load_onnx=False)
    cosyvoice = CosyVoice2('data/pretrained_models/CosyVoice2-0.5B')
    uvicorn.run(app, host="0.0.0.0", port=args.port)
