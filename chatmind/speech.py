"""语音输出（edge-tts + pygame）与语音识别（SenseVoice）"""

import asyncio
import logging
import os
import time

import edge_tts
import pygame

logger = logging.getLogger("chatmind")

# pygame mixer 模块级初始化一次（多线程下反复init/quit易导致音频设备异常）
try:
    pygame.mixer.init()
except pygame.error as e:
    logger.warning(f"音频设备初始化失败，播报将降级为文本输出: {e}")


def is_speaking() -> bool:
    """当前是否正在播报"""
    try:
        return pygame.mixer.music.get_busy()
    except pygame.error:
        return False


def _play_audio(file_path: str):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    time.sleep(0.3)  # 等待音频设备释放


async def _tts_generate(text: str, voice: str, output_file: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)


def speak(text: str, voice: str = "zh-CN-XiaoyiNeural",
          tts_dir: str = "./output/tts", delete_after: bool = True):
    """播报一段话；TTS失败时降级为日志文本输出，不让系统静默卡死"""
    logger.info(f"播报: {text}")
    os.makedirs(tts_dir, exist_ok=True)
    tts_file = os.path.join(tts_dir, f"tts_{int(time.time() * 1000)}.mp3")
    try:
        asyncio.run(_tts_generate(text, voice, tts_file))
        _play_audio(tts_file)
    except Exception as e:
        logger.error(f"TTS生成/播放失败，已降级为文本输出: {e}")
    finally:
        if delete_after:
            try:
                if os.path.exists(tts_file):
                    os.remove(tts_file)
            except OSError:
                pass


def load_asr(model_dir: str = "iic/SenseVoiceSmall"):
    """加载SenseVoice ASR模型（启动时调用一次，耗时较长）"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from funasr import AutoModel  # 延迟导入：加载重，按需引入
    logger.info(f"加载ASR模型: {model_dir}")
    return AutoModel(model=model_dir, trust_remote_code=True)


def transcribe(asr_model, wav_path: str) -> str:
    """语音转文字"""
    res = asr_model.generate(input=wav_path, cache={}, language="auto", use_itn=False)
    text = res[0]["text"].split(">")[-1].strip()
    return text
