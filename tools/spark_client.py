import logging
import requests
from typing import Union, Optional, BinaryIO

logger = logging.getLogger(__name__)


def get_spark_api_endpoint(server_url: str = "ttd-edge:8000") -> str:
    """
    获取 Spark API 推理服务的端点 URL
    
    Args:
        server_url: 服务器地址，默认为 "ttd-edge:8000"
        
    Returns:
        str: 完整的推理服务 URL
    """
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"
    
    url = f"{server_url}/api/infer"
    logger.info(f"Spark API request endpoint: {url}")
    return url

def request_spark_tts(
    text: str,
    prompt_speech: Union[str, BinaryIO],
    prompt_text: Optional[str] = None,
    postprocess: bool = True,
    server_url: str = "ttd-edge:8000",
    output_path: Optional[str] = None,
    timeout: int = 120,
) -> Union[bytes, str]:
    """
    请求 Spark TTS 服务生成音频
    
    根据 OpenAPI 规范:
    - text: 要合成的文本 (必填)
    - prompt_speech: 提示语音文件，WAV 格式 (必填，用于语音克隆)
    - prompt_text: 提示文本 (可选)
    - postprocess: 是否进行后处理，默认为 True (可选)
    
    Args:
        text: 要合成的文本
        prompt_speech: 提示语音，可以是文件路径、文件对象或NumPy数组
        prompt_text: 提示文本，可选
        postprocess: 是否进行后处理，默认为True
        server_url: 服务器地址，默认为 "ttd-edge:8000"
        output_path: 可选的输出文件路径，如果提供则保存音频到文件并返回文件路径
        timeout: 请求超时时间（秒），默认为120秒
        
    Returns:
        Union[bytes, str]: 
            - 如果提供了output_path，则返回保存的文件路径
            - 否则返回音频二进制数据
        
    Raises:
        ValueError: 参数错误
        Exception: 请求失败或其他错误
    """
    if not text:
        raise ValueError("text parameter is required")
    
    try:
        # 获取API端点
        url = get_spark_api_endpoint(server_url)
        
        # 准备请求数据
        files = {}
        data = {
            'text': text,
            'postprocess': str(postprocess).lower()
        }
        
        if prompt_text:
            data['prompt_text'] = prompt_text
        
        # 处理prompt_speech参数
        if isinstance(prompt_speech, str):
            # 如果是文件路径，直接使用文件
            files['prompt_speech'] = ('prompt.wav', open(prompt_speech, 'rb'), 'audio/wav')
        elif hasattr(prompt_speech, 'read'):
            # 如果是文件对象，直接使用
            files['prompt_speech'] = ('prompt.wav', prompt_speech, 'audio/wav')
        else:
            raise TypeError("prompt_speech must be a file path, file object, or NumPy array")
        
        # 发送请求
        logger.info(f"Requesting Spark TTS with text: {text}")
        rsp = requests.post(
            url,
            data=data,
            files=files,
            timeout=timeout
        )
        rsp.raise_for_status()  # 如果请求失败，抛出异常
        
        # 获取二进制音频数据
        audio_binary = rsp.content
        
        # 如果提供了输出路径，则保存到文件
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(audio_binary)
            logger.info(f"Audio saved to {output_path}")
            return output_path
        
        # 返回音频二进制数据
        return audio_binary
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            # 尝试解析验证错误
            try:
                error_detail = e.response.json().get("detail", [])
                error_msg = "; ".join([f"{err.get('loc', [])} - {err.get('msg', '')}" for err in error_detail])
                message = f"Validation error: {error_msg}"
            except:
                message = f"Validation error: {str(e)}"
        else:
            message = f"HTTP error: {str(e)}"
        
        logger.exception(message)
        raise Exception(message)
    except Exception as e:
        message = f"Spark TTS request failed: {e}"
        logger.exception(message)
        raise Exception(message)