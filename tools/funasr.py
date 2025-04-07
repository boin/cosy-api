import requests
import sys
import os


def auto_asr(audio_path):
    if not audio_path or os.path.exists(audio_path) is False:
        return ""
    res = asr_model(open(audio_path, "rb"))
    return res["result"][0]["clean_text"]


def asr_model(audio_data, host="ttd-server:8300"):
    """
    curl -X 'POST' \
     'http://ttd-server:8300/api/v1/asr' \
      -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@李弘彬_疑惑_0.wav;type=audio/wav' \
  -F 'keys=1' \
  -F 'lang=auto'
    """
    headers = {"accept": "application/json"}
    data = {"keys": "wav", "lang": "auto"}

    #     {
    #   "result": [
    #     {
    #       "key": "1",
    #       "text": "你真的不行😔",
    #       "raw_text": "<|zh|><|SAD|><|Speech|><|woitn|>你真的不行",
    #       "clean_text": "你真的不行"
    #     }
    #   ]
    # }

    response = requests.post(
        f"http://{host}/api/v1/asr",
        files={"files": audio_data},
        data=data,
        headers=headers,
    )
    return response.json()


if __name__ == "__main__":
    args = sys.argv[1:]
    audio_data = open(args[0], "rb")
    res = asr_model(audio_data, "192.168.100.16:8300")
    print(res["result"][0])
