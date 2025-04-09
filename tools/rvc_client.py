"""
音频转换模块

该模块提供了调用 http://ttd-stage:7867/ 的 /infer_convert 接口的简单封装。

使用方法：
    from lib.emotion_converter import convert_audio, refresh_resources
    
    # 刷新资源并获取可用模型和索引
    models, indices = refresh_resources()
    
    # 音频转换
    output_file = convert_audio(
        input_audio_path="/path/to/input.wav",
        output_path="/path/to/output.wav",
        file_index=indices[0]  # 使用第一个可用的索引
    )
"""

import os
import logging
import tempfile
from typing import Optional, List, Tuple
from gradio_client import Client, handle_file

logger = logging.getLogger(__name__)

# 服务端点
SERVER_ENDPOINT = "http://ttd-stage:7867"


def get_client() -> Client:
    """
    获取 Gradio 客户端
    
    Returns:
        Client: Gradio 客户端实例
    """
    return Client(SERVER_ENDPOINT)


def refresh_resources():
    """
    刷新并获取可用的音色模型和特征索引
    
    Returns:
        原始API返回结果，用于直接更新Gradio组件
    """
    try:
        client = get_client()
        result = client.predict(api_name="/infer_refresh")
        
        # 记录原始返回值
        if result and isinstance(result, tuple) and len(result) > 1:
            logger.info(f"刷新RVC资源成功，模型数量: {len(result[0]['choices'])}, 索引数量: {len(result[1]['choices'])}")
        else:
            logger.warning(f"刷新RVC资源返回格式异常: {result}")
        
        # 直接返回原始结果
        return result
    except Exception as e:
        logger.error(f"刷新RVC资源失败: {e}")
        return None


def find_matching_index(model_name: str, indices: List[str] = None) -> str:
    """
    根据模型名称查找匹配的特征索引
    
    Args:
        model_name (str): 模型名称，例如 "枫原万叶.pth"
        indices (List[str], optional): 特征索引列表。如果不提供，将返回空字符串
        
    Returns:
        str: 匹配的特征索引路径，如果没有找到匹配的索引，则返回空字符串
    """
    # 去掉文件扩展名
    base_name = os.path.splitext(model_name)[0]
    
    try:
        # 如果没有提供索引列表，直接返回空字符串
        if not indices:
            logger.warning(f"未提供特征索引列表")
            return ""
        
        # 查找匹配的索引
        for index in indices:
            if base_name in index:
                logger.info(f"找到匹配的特征索引: {index}")
                return index
        
        logger.warning(f"未找到与模型 {model_name} 匹配的特征索引")
        return ""
    except Exception as e:
        logger.error(f"查找匹配的特征索引失败: {e}")
        return ""


def convert_audio(
    input_audio_path: str,
    output_path: Optional[str] = None,
    sid: float = 0,
    f0_up_key: float = 0,
    f0_method: str = "rmvpe",
    file_index: str = None,
    file_index2: str = None,
    index_rate: float = 1.0,
    filter_radius: float = 3.0,
    resample_sr: float = 48000,
    rms_mix_rate: float = 1.0,
    protect: float = 0.33,
    loudnorm: float = -26,
    model_name: Optional[str] = None
) -> str:
    """
    调用 http://ttd-stage:7867/ 的 /infer_convert 接口转换音频
    
    Args:
        input_audio_path (str): 输入音频文件路径
        output_path (str, optional): 输出文件路径。如果不提供，将使用输入文件名加后缀
        sid (float, optional): 说话人ID。默认为0
        f0_up_key (float, optional): 变调参数，半音数量。默认为0
        f0_method (str, optional): 音高提取算法，可选值：pm, harvest, crepe, rmvpe。默认为rmvpe
        file_index (str, optional): 特征检索库文件路径。默认为空字符串
        file_index2 (str, optional): 自动检测的特征索引路径。默认为空字符串
        index_rate (float, optional): 检索特征占比。默认为1.0
        filter_radius (float, optional): 中值滤波半径。默认为3.0
        resample_sr (float, optional): 重采样采样率。默认为48000
        rms_mix_rate (float, optional): 音量包络融合比例。默认为1.0
        protect (float, optional): 保护清辅音和呼吸声的强度。默认为0.33
        loudnorm (float, optional): 音量标准化LUFS值。默认为-26
        model_name (str, optional): 音色模型名称。如果提供，将自动查找匹配的特征索引
        
    Returns:
        str: 转换后的音频文件路径
        
    Raises:
        ValueError: 当参数无效时抛出异常
        Exception: 当转换失败时抛出异常
    """
    # 验证参数
    if not os.path.exists(input_audio_path):
        raise ValueError(f"输入音频文件不存在: {input_audio_path}")
    
    # 如果提供了模型名称，自动查找匹配的特征索引
    if model_name and not file_index:
        _, indices = refresh_resources()
        file_index = find_matching_index(model_name, indices)
        if not file_index:
            logger.warning(f"未找到与模型 {model_name} 匹配的特征索引，将使用默认索引")
    
    # 如果未提供输出文件路径，生成默认路径
    if not output_path:
        base_name = os.path.splitext(input_audio_path)[0]
        output_path = f"{base_name}_converted.wav"
    
    try:
        # 使用 Gradio 客户端调用接口
        client = get_client()
        
        logger.info(f"开始转换音频: {input_audio_path}")
        logger.info(f"使用特征索引: {file_index}")
        logger.info(f"使用模型: {model_name}")
        logger.info(f"其他参数: sid={sid}, f0_up_key={f0_up_key}, f0_method={f0_method}")
        
        # 检查输入文件是否可读
        try:
            with open(input_audio_path, 'rb') as f:
                file_size = os.path.getsize(input_audio_path)
                logger.info(f"输入文件大小: {file_size} 字节")
        except Exception as e:
            logger.error(f"无法读取输入文件: {e}")
            raise ValueError(f"无法读取输入文件: {e}")
        
        # 检查特征索引文件是否可读
        if file_index and os.path.exists(file_index):
            try:
                with open(file_index, 'rb') as f:
                    index_size = os.path.getsize(file_index)
                    logger.info(f"特征索引文件大小: {index_size} 字节")
            except Exception as e:
                logger.warning(f"无法读取特征索引文件: {e}")
        
        try:            
            # 调用接口
            result = client.predict(
                sid=sid,
                input_audio_path=handle_file(input_audio_path),
                f0_up_key=f0_up_key,
                f0_file=None,  # 不使用F0曲线文件
                f0_method=f0_method,
                file_index=None,
                file_index2=file_index,  # 直接传递索引文件路径
                index_rate=index_rate,
                filter_radius=filter_radius,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
                loudnorm=loudnorm,
                api_name="/infer_convert"
            )
        except Exception as e:
            logger.exception(f"调用RVC接口失败: {e}")
            raise Exception(f"音频转换失败: {e}")
        
        # 结果是一个元组，第二个元素是输出音频文件路径
        output_info = result[0]
        output_file = result[1]
        
        # 将输出文件复制到指定路径
        if output_file and os.path.exists(output_file):
            with open(output_file, 'rb') as src_file, open(output_path, 'wb') as dst_file:
                dst_file.write(src_file.read())
            logger.info(f"音频转换完成，输出文件: {output_path}")
            return output_path
        else:
            raise Exception(f"转换失败，未生成输出文件。服务返回信息: {output_info}")
    
    except Exception as e:
        logger.exception(f"音频转换失败: {e}")
        raise Exception(f"音频转换失败: {e}")


def change_voice(model_name: str, param_1: float = 0.33, param_2: float = 0.33) -> tuple:
    """
    调用 http://ttd-stage:7867/ 的 /infer_change_voice 接口切换音色模型
    
    Args:
        model_name: 模型名称，例如 "马里奥.pth"
        param_1: 保护清辅音和呼吸声参数1，默认为0.33
        param_2: 保护清辅音和呼吸声参数2，默认为0.33
        
    Returns:
        tuple: 接口返回的元组，包含5个元素
            [0] float: 说话人ID
            [1] float: 参数1
            [2] float: 参数2
            [3] str: 自动检测的索引路径1
            [4] str: 自动检测的索引路径2
            
    Raises:
        Exception: 当切换音色失败时抛出异常
    """
    try:
        # 使用 Gradio 客户端调用接口
        client = get_client()
        
        logger.info(f"开始切换音色模型: {model_name}")
        
        result = client.predict(
            sid=model_name,
            param_1=param_1,
            param_2=param_2,
            api_name="/infer_change_voice"
        )
        
        logger.info(f"成功切换音色模型: {model_name}")
        logger.info(f"自动检测的索引路径: {result[3]}")
        
        return result
    except Exception as e:
        logger.exception(f"切换音色模型失败: {e}")
        raise Exception(f"切换音色模型失败: {e}")