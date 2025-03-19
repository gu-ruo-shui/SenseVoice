# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os
import re
from enum import Enum
from io import BytesIO
from typing import List

import torchaudio
from fastapi import FastAPI, File, Form
from fastapi.responses import HTMLResponse
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from typing_extensions import Annotated

from model import SenseVoiceSmall


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"


model_dir = "iic/SenseVoiceSmall"
# m, kwargs = SenseVoiceSmall.from_pretrained(
#     model=model_dir, device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"), batch_size=64
# )
# m.eval()

m = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)


regex = r"<\|.*\|>"

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.post("/api/v1/asr")
async def turn_audio_to_text(
    files: Annotated[List[bytes], File(description="wav or mp3 audios in 16KHz")],
    keys: Annotated[str, Form(description="name of each audio joined with comma")],
    lang: Annotated[Language, Form(description="language of audio content")] = "auto",
):
    audios = []
    audio_fs = 0
    for file in files:
        file_io = BytesIO(file)
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)
        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)
        file_io.close()
    if lang == "":
        lang = "auto"
    if keys == "":
        key = ["wav_file_tmp_name"]
    else:
        key = keys.split(",")
    res = m.generate(
        fs=audio_fs,
        language="auto",  # 自动检测语言，也可以指定"zn"（中文）、"en"（英语）等
        use_itn=True,  # 使用数字文本标准化
        batch_size_s=60,  # 批处理大小（秒）
        merge_vad=True,  # 合并语音活动检测
        merge_length_s=15,  # 合并长度（秒）
        key=key,
    )

    if len(res) == 0:
        return {"result": []}
    print(res[0].keys())
    for it in res[0]:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
        it["text"] = rich_transcription_postprocess(it["text"])
    return {"result": res[0]}
