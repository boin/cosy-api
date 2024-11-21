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
import random
import sys
import wave
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import torch
import uvicorn
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

logging.getLogger("matplotlib").setLevel(logging.WARNING)

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


def get_model_info(actor_uuid, voice_uuid) -> (bytes, str):
    model_root = "{}/data/links".format(ROOT_DIR)
    prompt_wav = Path(model_root) / actor_uuid / "train" / f"{voice_uuid}.wav"
    return prompt_wav, voice_uuid.split("_")[-1]


def normalize_sound(audio, sr=22050):
    """
    Normalize the loudness of an audio signal to -25 LUFS.
    Args:
        audio (numpy.ndarray): The input audio signal (float32 expected).
        sr (int): The sample rate of the audio signal.
    Returns:
        numpy.ndarray: The loudness-normalized audio signal (float32).
    """
    # Create a loudness meter
    meter = pyln.Meter(sr)  # ITU-R BS.1770 meter
    # Measure loudness
    loudness = meter.integrated_loudness(audio)
    # Normalize to -25 LUFS
    normalized_audio = pyln.normalize.loudness(audio, loudness, -25.0)
    return normalized_audio


def generate_data(model_output):
    """
    Generate a WAV audio file from model output with loudness normalization.
    Args:
        model_output (list): A list of dictionaries containing audio chunks.
    Returns:
        io.BytesIO: A memory buffer containing the WAV audio file.
    """
    # Combine the TTS audio chunks
    tts_audio = b""
    for i in model_output:
        tts_audio += (i["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
    # Convert byte stream to numpy array (int16)
    audio_array = np.frombuffer(tts_audio, dtype=np.int16)
    # Normalize the audio
    # Convert to float32 (necessary for loudness normalization)
    float_audio = audio_array.astype(np.float32) / (2**15)
    normalized_audio = normalize_sound(float_audio, sr=22050)
    # Convert back to int16
    int16_audio = (normalized_audio * (2**15)).astype(np.int16)
    # Write to WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(22050)  # 22.05 kHz sample rate
        wav_file.writeframes(int16_audio.tobytes())
    # Reset buffer position
    buffer.seek(0)
    return buffer


"""
"rseed": 8343459,
"voice_uuid": "旁白_dildo,好奇,惊讶_001_535848_不知过去了多长时间，我的眼前，忽然出现了一抹光亮。",
"actor_uuid": "00291d13ba81803a546407fcfc34b365",
"text": "很快，我们回到了村子里。唐雅兰向我告辞，回了自己的家。我和老八以及天机子，则回到了师父的居所。歇了一夜。"
curl -X POST "https://8000.wodeip.org:8361/infer" \
     -d "text='%E5%BE%88%E5%BF%AB%EF%BC%8C%E6%88%91%E4%BB%AC%E5%9B%9E%E5%88%B0%E4%BA%86%E6%9D%91%E5%AD%90%E9%87%8C%E3%80%82%E5%94%90%E9%9B%85%E5%85%B0%E5%90%91%E6%88%91%E5%91%8A%E8%BE%9E%EF%BC%8C%E5%9B%9E%E4%BA%86%E8%87%AA%E5%B7%B1%E7%9A%84%E5%AE%B6%E3%80%82%E6%88%91%E5%92%8C%E8%80%81%E5%85%AB%E4%BB%A5%E5%8F%8A%E5%A4%A9%E6%9C%BA%E5%AD%90%EF%BC%8C%E5%88%99%E5%9B%9E%E5%88%B0%E4%BA%86%E5%B8%88%E7%88%B6%E7%9A%84%E5%B1%85%E6%89%80%E3%80%82%E6%AD%87%E4%BA%86%E4%B8%80%E5%A4%9C%E3%80%82'" \
     -d "voice_uuid=%E6%97%81%E7%99%BD_dildo,%E5%A5%BD%E5%A5%87,%E6%83%8A%E8%AE%B6_001_535848_%E4%B8%8D%E7%9F%A5%E8%BF%87%E5%8E%BB%E4%BA%86%E5%A4%9A%E9%95%BF%E6%97%B6%E9%97%B4%EF%BC%8C%E6%88%91%E7%9A%84%E7%9C%BC%E5%89%8D%EF%BC%8C%E5%BF%BD%E7%84%B6%E5%87%BA%E7%8E%B0%E4%BA%86%E4%B8%80%E6%8A%B9%E5%85%89%E4%BA%AE%E3%80%82" \
     -d "actor_uuid=df1b5ea67ae881bf5dd97b583e6e7d72" \
     -d "rseed=834345"
"""


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

    prompt_wav, prompt_text = get_model_info(actor_uuid, voice_uuid)
    prompt_speech_16k = load_wav(prompt_wav, 16000)

    model_output = cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k)
    return StreamingResponse(
        generate_data(model_output),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="iic/CosyVoice-300M-25Hz",
        help="local path or modelscope repo id",
    )
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
