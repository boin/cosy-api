import logging

from gradio import Error
from hashlib import md5
import requests

logger = logging.getLogger(__name__)


def get_infer_end_point(name: str) -> str:
    if not name or type(name) is not str:
        raise ValueError(f"endpoint_name {name} should be a non-empty string.")
    endpoints_portmap = [360, 361, 362]
    index = (ord(name[-1]) - 1) % 3
    logger.info(f"SVC request get_api_end_point: {name} -> {endpoints_portmap[index]}")
    #return f"http://localhost:8000/zero_shot_infer"
    return f"http://ttd-worker:{endpoints_portmap[index]}/zero_shot_infer"


def request_infer(
    text: str,
    rseed: str,
    ref_file: str,
    asr: str,
    speed: str = "1",
) -> (int, str):
    if not (url := get_infer_end_point(asr)):
        return 1, f"endpoint_name {asr} not valid."

    # 请求的 payload
    request_payload = {
        "text": text,
        "ref_file": ref_file,
        "asr": asr,
        "rseed": rseed,
        "speed": speed
    }

    try:
        logger.info(f"requesting zero shot infer with payload: {request_payload} to {url}")
        # 2分钟内没有返回内容则视为失败，GPU Hang 或者 GPU Fail
        files = [("ref_file", ("ref_file", open(ref_file, "rb"), "application/octet-stream"))]
        response = requests.post(url, data=request_payload, files=files, timeout=120)
        response.raise_for_status()  # 如果请求失败，抛出异常

        # 将内容写入文件
        return response.content

    except Exception as e:
        message = f"Unknown error: {e}"
        logger.exception(message)
        raise Error(message)


def get_svc_end_point(name: str) -> str:
    endpoints_portmap = [7000, 7001, 7002]
    if not name or type(name) is not str:
        raise ValueError(f"endpoint_name {name} should be a non-empty string.")
    index = (ord(name[-1]) - 1) % 3
    logger.info(f"SVC request get_api_end_point: {name} -> {endpoints_portmap[index]}")
    #return f"http://localhost:7856/svc_file"
    return f"http://ttd-worker:{endpoints_portmap[index]}/svc_file"


def request_svc(
    src_file: str,
    ref_file: str,
    steps: int = 50,
) -> (int, str):
    if not (url := get_svc_end_point(md5(f"{src_file}-{ref_file}".encode()).hexdigest())):
        return 1, f"endpoint_name not valid."

    # 请求的 payload
    request_payload = {
        "steps": steps,
    }

    files = [
        ("ref_file", ("ref_file", open(ref_file, "rb"), "application/octet-stream")),
        ("src_file", ("src_file", open(src_file, "rb"), "application/octet-stream")),
    ]

    try:
        logger.info(
            f"requesting svc with payload: {request_payload} files: {[src_file, ref_file]} to {url}"
        )
        response = requests.post(url, params=request_payload, files=files, timeout=120)
        response.raise_for_status()  # 如果请求失败，抛出异常

        # 将内容写入文件
        return response.content

    except Exception as e:
        message = f"Unknown error: {e}"
        raise Error(message)
