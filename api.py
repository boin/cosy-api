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

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
    try:
        meter = pyln.Meter(sr)  # ITU-R BS.1770 meter
        loudness = meter.integrated_loudness(audio)
        logging.info(f"Original loudness: {loudness:.2f} LUFS, normalize to target: -23 LUFS")
        # Normalize to -23 LUFS
        normalized_audio = pyln.normalize.loudness(audio, loudness, -23.0)
        return normalized_audio
    except Exception as e:
        logging.exception(f"Failed to normalize audio {e}, return original audio")
        return audio


def generate_data(model_output):
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
    normalized_audio = normalize_sound(tts_audio, sr=22050)
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
    args = parser.parse_args()
    cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-25Hz', load_jit=True, load_onnx=False, load_trt=False)
    #cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_onnx=False, load_trt=False)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
