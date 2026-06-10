"""
chatMind 主入口 —— 语音心理陪伴与隐式抑郁评估

运行前：
    set DASHSCOPE_API_KEY=你的_api_key
    python main.py

体验流程：
    - 日常陪伴：直接说话即可，像和朋友聊天
    - 想了解自己的状态：说"帮我做个评测 / 聊聊我最近的状态"，
      之后依然是自然聊天，没有"第几题"，分析在后台静默完成
    - 评测中想停：说"先这样吧 / 换个话题"即可
    - 报告：自动保存到 HAMD_Results/（不会在对话中播报分数）
"""

import os
import time

from chatmind.config import load_config, setup_logger
from chatmind.llm_client import init_llm, chat_api
from chatmind import speech
from chatmind.audio_capture import AudioCapture
from chatmind.evaluator import HAMDEvaluator
from chatmind.dialogue import DialogueManager


def main():
    config = load_config()
    logger = setup_logger(config["paths"]["log_dir"])
    init_llm(config["llm"])

    # --- ASR ---
    asr_model = speech.load_asr(config["asr"]["model_dir"])

    # --- 评估器（隐式对话式） ---
    evaluator = HAMDEvaluator(
        questions_file=config["paths"]["questions_file"],
        output_dir=config["paths"]["report_dir"],
        chat_fn=lambda messages, temperature=0.3, max_tokens=256: chat_api(
            messages, temperature=temperature, max_tokens=max_tokens, fallback=""),
        stage_thresholds={int(k): v for k, v in config["assessment"]["stage_thresholds"].items()},
        max_follow_up=config["assessment"]["max_follow_up"],
        crisis_hotline=config["safety"]["crisis_hotline"],
    )

    # --- 播报 ---
    def speak(text):
        speech.speak(text,
                     voice=config["tts"]["voice"],
                     tts_dir=config["paths"]["tts_dir"],
                     delete_after=config["privacy"]["delete_tts_after_play"])

    # --- 对话编排 ---
    dialogue = DialogueManager(config, evaluator, speak)

    # --- 一句话的处理管线：wav -> ASR -> 对话 -> （隐私清理） ---
    def on_utterance(wav_path):
        try:
            text = speech.transcribe(asr_model, wav_path)
            logger.info(f"ASR: {text}")
            if text:
                dialogue.process_text(text)
        except Exception as e:
            logger.exception(f"处理语音失败: {e}")
            try:
                speak("抱歉，我刚才走神了，你能再说一遍吗？")
            except Exception:
                pass
        finally:
            if config["privacy"]["delete_audio_after_use"]:
                try:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                except OSError:
                    pass

    # --- 采集：答题等待时停顿阈值放宽；播报/处理中丢弃语音 ---
    capture = AudioCapture(
        on_utterance=on_utterance,
        output_dir=config["paths"]["output_dir"],
        vad_mode=config["audio"]["vad_mode"],
        no_speech_threshold=config["audio"]["no_speech_threshold"],
        pause_threshold_fn=lambda: (config["audio"]["answer_pause_seconds"]
                                    if dialogue.waiting_for_answer
                                    else config["audio"]["no_speech_threshold"]),
        should_ignore_fn=lambda: speech.is_speaking() or dialogue.processing,
    )

    capture.start()
    speak(f"你好，我是{config['persona']['assistant_name']}。"
          "在这里你可以放心说任何想说的话，我会认真听。"
          "想了解自己最近的状态，跟我说一声就行。")

    logger.info("系统就绪，按 Ctrl+C 退出")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("正在退出...")
        capture.stop()


if __name__ == "__main__":
    main()
