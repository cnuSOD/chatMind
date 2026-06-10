"""
chatMind - 语音心理陪伴与隐式抑郁评估系统

模块结构：
    config.py        配置加载与日志
    llm_client.py    千问（OpenAI兼容）客户端
    speech.py        TTS播报与ASR识别
    audio_capture.py 麦克风采集与VAD断句
    evaluator.py     HAMD评估器（隐式对话式提问 + 后台静默评分）
    dialogue.py      对话编排（普通陪伴 / 隐式评测 / 危机协议）

入口：仓库根目录 main.py
"""

__version__ = "4.0.0"
