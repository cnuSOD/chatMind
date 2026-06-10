"""配置加载与日志初始化"""

import json
import logging
import os

CONFIG_FILE = "chatmind_config.json"

DEFAULT_CONFIG = {
    "llm": {
        "model": "qwen-turbo",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "tts": {"voice": "zh-CN-XiaoyiNeural"},
    "asr": {"model_dir": "iic/SenseVoiceSmall"},
    "audio": {
        "answer_pause_seconds": 3.5,
        "no_speech_threshold": 1.0,
        "min_answer_length": 3,
        "vad_mode": 3,
    },
    "assessment": {
        "stage_thresholds": {"1": 5, "2": 10},
        "max_follow_up": 2,
    },
    "safety": {
        "crisis_hotline": "12356",
        "crisis_keywords": ["想死", "自杀", "自残", "不想活", "活着没意思",
                            "活不下去", "了结", "解脱", "伤害自己"],
    },
    "privacy": {"delete_audio_after_use": True, "delete_tts_after_play": True},
    "persona": {"assistant_name": "小知"},
    "paths": {
        "output_dir": "./output",
        "tts_dir": "./output/tts",
        "report_dir": "./HAMD_Results/",
        "log_dir": "./logs",
        "questions_file": "hamd_questions_v3.json",
    },
}


def load_config(path: str = CONFIG_FILE) -> dict:
    """加载配置文件，缺失项回落到默认值（按节浅合并）"""
    config = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    try:
        with open(path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        for section, values in user_config.items():
            if section in config and isinstance(values, dict):
                config[section].update(values)
            else:
                config[section] = values
    except FileNotFoundError:
        print(f"[配置] 未找到{path}，使用默认配置")
    except (json.JSONDecodeError, OSError) as e:
        print(f"[配置] 读取{path}失败（{e}），使用默认配置")
    return config


def setup_logger(log_dir: str = "./logs") -> logging.Logger:
    """初始化日志：控制台 + 文件双通道"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "chatmind.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("chatmind")
