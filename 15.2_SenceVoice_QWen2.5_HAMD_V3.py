"""
语音对话系统 + HAMD V3 心理评估
集成三阶段智能评估、LLM动态提问、情绪感知
"""

import cv2
import pyaudio
import wave
import threading
import numpy as np
import time
from queue import Queue
import webrtcvad
import os
from funasr import AutoModel
import pygame
from openai import OpenAI
import edge_tts
import asyncio
import re
from pypinyin import pinyin, Style

# 导入 HAMD V3 评估器
from hamd_evaluator_v3 import HAMDEvaluatorV3

# --- 配置huggingFace国内镜像 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 参数设置
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
CHUNK = 1024
VAD_MODE = 3
OUTPUT_DIR = "./output"
NO_SPEECH_THRESHOLD = 1
folder_path = "./Test_QWen2_VL/"
audio_file_count = 0

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(folder_path, exist_ok=True)

# 队列用于音频和视频同步缓存
audio_queue = Queue()
video_queue = Queue()

# 全局变量
last_active_time = time.time()
recording_active = True
segments_to_save = []
saved_intervals = []
last_vad_end_time = 0

# --- 唤醒词配置 ---
set_KWS = "ni hao xiao qian"
flag_KWS = 0
flag_KWS_used = 1  # 1=不使用唤醒词，0=使用唤醒词

# --- TTS音色配置 ---
DEFAULT_TTS_VOICE = "zh-CN-XiaoyiNeural"
VOICE_LOCK_ENABLED = True

# --- HAMD V3 评估相关变量 ---
hamd_evaluator = None
hamd_evaluation_active = False
hamd_trigger_keywords = ["抑郁评估", "心理测试", "抑郁测试", "心理评估", "开始评估", "hamd", "抑郁量表"]
hamd_waiting_for_answer = False  # 正在等待用户回答HAMD问题
hamd_processing_lock = False  # HAMD正在处理中（TTS生成、播放等）
ANSWER_PAUSE_SECONDS = 3.5  # 等待用户回答时的停顿时间（秒）
MIN_ANSWER_LENGTH = 3  # 最短有效回答长度（字符数）

# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)


def extract_chinese_and_convert_to_pinyin(input_string):
    """提取汉字并转换为拼音"""
    chinese_characters = re.findall(r'[\u4e00-\u9fa5]', input_string)
    chinese_text = ''.join(chinese_characters)
    pinyin_result = pinyin(chinese_text, style=Style.NORMAL)
    pinyin_text = ' '.join([item[0] for item in pinyin_result])
    return pinyin_text


# 音频录制线程
def audio_recorder():
    global audio_queue, recording_active, last_active_time, segments_to_save, last_vad_end_time
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=AUDIO_CHANNELS,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    audio_buffer = []
    print("音频录制已开始")
    
    while recording_active:
        data = stream.read(CHUNK)
        audio_buffer.append(data)
        
        # 每 0.5 秒检测一次 VAD
        if len(audio_buffer) * CHUNK / AUDIO_RATE >= 0.5:
            raw_audio = b''.join(audio_buffer)
            vad_result = check_vad_activity(raw_audio)
            
            if vad_result:
                print("检测到语音活动")
                last_active_time = time.time()
                segments_to_save.append((raw_audio, time.time()))
            else:
                print("静音中...")
            
            audio_buffer = []
        
        # 动态阈值：评估模式下等待时间更长
        threshold = ANSWER_PAUSE_SECONDS if hamd_waiting_for_answer else NO_SPEECH_THRESHOLD
        if time.time() - last_active_time > threshold:
            if segments_to_save and segments_to_save[-1][1] > last_vad_end_time:
                save_audio_video()
                last_active_time = time.time()
    
    stream.stop_stream()
    stream.close()
    p.terminate()


def check_vad_activity(audio_data):
    """检测 VAD 活动"""
    num, rate = 0, 0.5
    step = int(AUDIO_RATE * 0.02)
    flag_rate = round(rate * len(audio_data) // step)

    for i in range(0, len(audio_data), step):
        chunk = audio_data[i:i + step]
        if len(chunk) == step:
            if vad.is_speech(chunk, sample_rate=AUDIO_RATE):
                num += 1

    return num > flag_rate


def save_audio_video():
    """保存音频并启动推理"""
    pygame.mixer.init()

    global segments_to_save, video_queue, last_vad_end_time, saved_intervals
    global audio_file_count

    audio_file_count += 1
    audio_output_path = f"{OUTPUT_DIR}/audio_{audio_file_count}.wav"

    if not segments_to_save:
        return
    
    # 如正在播报，则忽略本次语音段
    if pygame.mixer.music.get_busy():
        print("正在播报，忽略本次语音段")
        segments_to_save.clear()
        return
    
    # 如果HAMD正在处理中，阻止新的推理
    if hamd_processing_lock:
        print("HAMD正在处理中，忽略本次语音段")
        segments_to_save.clear()
        return
    
    # 检查时间重叠
    start_time = segments_to_save[0][1]
    end_time = segments_to_save[-1][1]
    
    if saved_intervals and saved_intervals[-1][1] >= start_time:
        print("当前片段与之前片段重叠，跳过保存")
        segments_to_save.clear()
        return
    
    # 保存音频
    audio_frames = [seg[0] for seg in segments_to_save]
    wf = wave.open(audio_output_path, 'wb')
    wf.setnchannels(AUDIO_CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(AUDIO_RATE)
    wf.writeframes(b''.join(audio_frames))
    wf.close()
    print(f"音频保存至 {audio_output_path}")

    # 使用线程执行推理
    inference_thread = threading.Thread(target=Inference, args=(audio_output_path,))
    inference_thread.start()
    
    # 记录保存的区间
    saved_intervals.append((start_time, end_time))
    
    # 清空缓冲区
    segments_to_save.clear()


def play_audio(file_path):
    """播放音频"""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        print("播放完成！")
        time.sleep(0.5)  # 额外等待，确保音频设备完全释放
    except Exception as e:
        print(f"播放失败: {e}")
    finally:
        pygame.mixer.quit()


async def amain(TEXT, VOICE, OUTPUT_FILE) -> None:
    """TTS生成"""
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)


# -------- SenceVoice 语音识别 --模型加载-----
model_dir = r"iic/SenseVoiceSmall"
model_senceVoice = AutoModel(model=model_dir, trust_remote_code=True)


# --------- 千问 DashScope（OpenAI 兼容）适配 ---------------
def get_qwen_client():
    base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
    return OpenAI(api_key=api_key, base_url=base_url)


def chat_api(messages, model=None, temperature=0.7, max_tokens=512):
    try:
        client = get_qwen_client()
        model_name = model or os.getenv("QWEN_CHAT_MODEL", "qwen-turbo")
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"调用千问失败: {e}")
        return "抱歉，我现在无法连接千问服务，请稍后再试。"


# 初始化HAMD V3评估器（云端千问）
try:
    hamd_evaluator = HAMDEvaluatorV3(
        questions_file="hamd_questions_v3.json",
        output_dir="./HAMD_Results/",
        chat_fn=lambda messages, temperature=0.3, max_tokens=256: chat_api(messages, temperature=temperature, max_tokens=max_tokens)
    )
    print("HAMD V3评估器初始化成功（三阶段智能评估）")
except Exception as e:
    print(f"HAMD V3评估器初始化失败: {e}")
    hamd_evaluator = None


class ChatMemory:
    """对话记忆管理"""
    def __init__(self, max_length=2048):
        self.history = []
        self.max_length = max_length

    def add_to_history(self, user_input, model_response):
        self.history.append(f"User: {user_input}")
        self.history.append(f"system: {model_response}")

    def get_context(self):
        context = "\n".join(self.history)
        if len(context) > self.max_length:
            context = context[-self.max_length:]
        return context


memory = ChatMemory(max_length=512)


def system_introduction(text):
    """系统播报"""
    global folder_path
    print("LLM output:", text)
    used_speaker = DEFAULT_TTS_VOICE if VOICE_LOCK_ENABLED else "zh-CN-YunxiNeural"
    timestamp = int(time.time() * 1000)
    tts_file = os.path.join(folder_path, f"sft_tmp_{timestamp}.mp3")
    asyncio.run(amain(text, used_speaker, tts_file))
    play_audio(tts_file)


def check_hamd_trigger(text):
    """检查是否触发HAMD评估"""
    global hamd_trigger_keywords
    text_lower = text.lower()
    
    # 规范化常见听写错误
    def _normalize(s: str) -> str:
        s = s.replace("心里", "心理").replace("抑于", "抑郁").replace("抑以", "抑郁")
        s = s.replace("评古", "评估").replace("测是", "测试")
        return s
    
    text_lower = _normalize(text_lower)
    
    # 检查触发词
    evaluation_words = ["评估", "测试", "量表", "hamd"]
    depression_words = ["抑郁", "心理", "情绪", "心情"]
    
    has_evaluation = any(word in text_lower for word in evaluation_words)
    has_depression = any(word in text_lower for word in depression_words)
    
    strong_triggers = ["心理测试", "抑郁测试", "开始抑郁评估", "开始心理评估"]
    for keyword in (hamd_trigger_keywords + strong_triggers):
        if keyword in text_lower:
            return True
    
    return has_evaluation and has_depression


def classify_hamd_intent(text, current_question_context=""):
    """识别评估控制意图"""
    schema = (
        "你是心理评估意图分类器。判断用户说的话属于哪类意图:\n"
        "- answer: 回答评估问题（默认选项，只要用户在回答问题就选这个）\n"
        "- repeat: 用户明确要求重复当前问题\n"
        "- previous: 用户明确要求返回上一题\n"
        "- next/skip: 用户明确要求跳过当前题\n"
        "- stop: 用户明确要求结束评估（如：停止、结束评估、不做了）\n"
        "- switch_topic: 用户明确要求暂停评估去聊别的\n\n"
        "重要规则：\n"
        "1. 如果用户的回答与问题内容相关，无论是肯定还是否定的回答，都应该是answer\n"
        "2. 只有当用户明确说出控制指令时，才选择其他意图\n"
        "3. 有疑问时，默认选择answer\n\n"
        "只返回JSON: {\"intent\":\"answer\"}"
    )
    
    user_prompt = f"当前问题：{current_question_context}\n用户说：{text}\n\n请判断意图。"
    
    messages = [
        {"role": "system", "content": schema},
        {"role": "user", "content": user_prompt},
    ]
    raw = chat_api(messages, max_tokens=128)
    
    try:
        import json as _json
        parsed = _json.loads(raw)
        if isinstance(parsed, dict) and "intent" in parsed:
            return parsed
    except Exception:
        pass
    
    # 回退：正则识别（更保守的匹配）
    t = text.strip().lower()
    
    # 只有当文本非常明确时才触发控制指令
    # 使用更严格的匹配条件（完整短语，避免部分匹配）
    
    if any(t == k or t.startswith(k) for k in ["重新开始", "重来", "从头开始", "restart"]):
        return {"intent": "restart"}
    if any(t == k or t.startswith(k) for k in ["继续评估", "接着来", "resume"]):
        return {"intent": "resume"}
    if any(t == k or t.startswith(k) for k in ["上一题", "回到上一题", "previous"]):
        return {"intent": "previous"}
    if any(t == k or t.startswith(k) for k in ["下一题", "跳过这题", "skip", "next"]):
        return {"intent": "next"}
    if any(t == k or t.startswith(k) for k in ["重复一遍", "再说一遍问题", "repeat"]):
        return {"intent": "repeat"}
    
    # stop意图需要非常明确的表达
    stop_phrases = ["停止评估", "结束评估", "不做了", "不想做了", "退出评估"]
    if any(phrase in t for phrase in stop_phrases):
        # 排除一些可能被误判的回答
        exclude_phrases = ["不太", "不感兴趣", "不喜欢", "不开心", "不想活", "不想说"]
        if not any(exclude in t for exclude in exclude_phrases):
            return {"intent": "stop"}
    
    if any(t == k or t.startswith(k) for k in ["开始评估", "开始心理评估", "start"]):
        return {"intent": "start"}
    if any(k in t for k in ["不想做评估", "我们聊点别的", "暂停评估", "先不评估"]):
        return {"intent": "switch_topic"}
    
    # 默认都视为回答问题
    return {"intent": "answer"}


def start_hamd_evaluation():
    """开始HAMD V3评估"""
    global hamd_evaluator, hamd_evaluation_active
    
    if not hamd_evaluator:
        text = "抱歉，心理评估系统暂时不可用。"
        system_introduction(text)
        return False
    
    hamd_evaluation_active = True
    evaluation_id = hamd_evaluator.start_evaluation()
    
    intro_text = "好的，我将为您进行心理评估。请您如实回答每个问题。"
    system_introduction(intro_text)
    
    # 开始第一个问题
    ask_next_hamd_question()
    return True


def ask_next_hamd_question(is_follow_up=False):
    """询问下一个HAMD问题（支持追问）"""
    global hamd_evaluator, hamd_waiting_for_answer, hamd_processing_lock
    
    hamd_processing_lock = True
    
    try:
        if not hamd_evaluator or hamd_evaluator.is_evaluation_complete():
            hamd_waiting_for_answer = False
            return False
        
        current_question = hamd_evaluator.get_current_question()
        if current_question:
            # 使用LLM动态生成问题
            question_prompt = hamd_evaluator.get_question_prompt(is_follow_up=is_follow_up)
            hamd_waiting_for_answer = True
            
            # 显示进度信息（仅控制台）
            progress = hamd_evaluator.get_progress()
            print(f"[HAMD进度] 阶段{progress['current_stage']}, "
                  f"第{progress['current_question_in_stage']}/{progress['total_questions_in_stage']}题, "
                  f"情绪状态:{progress['emotional_state']}, "
                  f"风险等级:{progress['risk_level']}")
            
            system_introduction(question_prompt)
            return True
        
        hamd_waiting_for_answer = False
        return False
    
    finally:
        hamd_processing_lock = False
        print("[调试] 问题播放完成，释放锁")


def process_hamd_answer(user_answer):
    """处理HAMD评估回答（V3版本支持追问）"""
    global hamd_evaluator, hamd_evaluation_active, hamd_waiting_for_answer, hamd_processing_lock
    
    hamd_processing_lock = True
    
    try:
        hamd_waiting_for_answer = False
        
        if not hamd_evaluator:
            return False
        
        # 处理当前回答
        result = hamd_evaluator.process_answer(user_answer)
        
        # 检查是否需要追问
        if result.get("need_follow_up", False):
            print(f"[HAMD] 需要追问（第{result.get('follow_up_count', 1)}次）")
            # 不生成共情回应，直接追问
            ask_next_hamd_question(is_follow_up=True)
            return True
        
        # 生成共情回应
        try:
            sys_prompt = (
                "你是心理评估助手。请对用户的回答给出一句简短的共情性回应，"
                "使用温暖、支持的语气，不超过20个字；"
                "不要提问；不要给建议；不要复述评分。"
            )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"用户刚才说：{user_answer}"},
            ]
            comfort = chat_api(messages, temperature=0.7, max_tokens=50)
            
            comfort = comfort.replace("\n", "").strip()
            if ("?" in comfort) or ("？" in comfort):
                comfort = "我理解，谢谢您的分享。"
            if len(comfort) > 25:
                comfort = comfort[:25]
            
            system_introduction(comfort)
        except Exception as e:
            print(f"生成共情回应失败: {e}")
            system_introduction("我理解，谢谢您的分享。")
        
        # 检查高风险警报
        if result.get("high_risk_alert", False):
            print("[警报] 检测到高风险情况！")
        
        # 检查是否完成
        if result.get("is_complete", False):
            complete_hamd_evaluation()
        else:
            # 继续下一题（会自动切换阶段）
            ask_next_hamd_question()
        
        return True
    
    finally:
        hamd_processing_lock = False
        print("[调试] HAMD处理完成，释放锁")


def generate_personalized_feedback(report):
    """使用大模型生成个性化的评估反馈"""
    try:
        # 提取关键信息
        total_score = report['total_score']
        max_score = report['max_score']
        level = report['severity_level']['level']
        risk_level = report['risk_assessment']['overall_risk_level']
        emotion_analysis = report.get('emotion_analysis', {})
        category_analysis = report.get('category_analysis', {})
        
        # 构建分项分析摘要
        category_summary = []
        for category, data in category_analysis.items():
            if data['percentage'] >= 50:  # 只提及得分较高的维度
                category_summary.append(f"{category}方面得分{data['percentage']:.0f}%")
        category_text = "、".join(category_summary) if category_summary else "各维度表现均衡"
        
        # 构建提示词
        sys_prompt = f"""你是温暖、专业的心理评估助手。用户刚刚完成了HAMD抑郁量表评估，你需要给出一段个性化的反馈。

评估结果：
- 总分：{total_score}/{max_score}分
- 严重程度：{level}
- 风险等级：{risk_level}
- 分项表现：{category_text}
- 整体情绪：{emotion_analysis.get('dominant_emotion', 'neutral')}

要求：
1. 用温暖、共情、支持的语气
2. 首先感谢用户的配合和坦诚
3. 根据严重程度给出适当的反馈：
   - 正常：鼓励保持，给予肯定
   - 轻度：理解困扰，给出生活建议
   - 中度：表达关心，建议寻求专业帮助
   - 重度：温和关怀，强调寻求医疗支持的重要性
4. 如果有高风险情况，温和地强调寻求帮助
5. 结尾提及报告已保存
6. 语气自然口语化，就像面对面交流
7. 控制在80-120字
8. 不要使用"您"，用"你"更亲切
9. 不要说具体分数

请生成一段温暖的反馈："""

        messages = [
            {"role": "system", "content": "你是温暖、专业的心理评估助手，擅长用共情的方式给出反馈。"},
            {"role": "user", "content": sys_prompt}
        ]
        
        result_text = chat_api(messages, temperature=0.8, max_tokens=256)
        
        # 清洗生成的文本
        result_text = result_text.strip().replace("\n\n", "\n")
        
        # 如果生成失败或太短，使用备用文本
        if len(result_text) < 30:
            return get_fallback_feedback(level)
        
        return result_text
        
    except Exception as e:
        print(f"[生成个性化反馈失败] {e}")
        return get_fallback_feedback(level)


def get_fallback_feedback(level):
    """备用反馈文本"""
    fallback_texts = {
        "正常": "感谢你的配合，评估已经完成。从测评结果来看，你目前的心理状态整体比较稳定，没有明显的抑郁症状。继续保持积极心态哦。详细的评估报告已经保存好了。",
        "轻度抑郁": "评估完成了。从结果来看，你可能正在经历一些轻度的情绪困扰。建议你多和亲友交流，保持规律作息，适当运动放松。详细的评估报告已经保存好了。",
        "中度抑郁": "评估已经完成。测评显示你目前可能承受着较大的心理压力。建议你尽快寻求专业的心理咨询或医疗帮助。详细的评估报告已经保存好了。",
        "重度抑郁": "评估已经完成。测评结果表明你当前的状况需要得到专业关注。请你尽快联系专业的心理医生或前往医院的心理科就诊。详细的评估报告已经保存好了。"
    }
    return fallback_texts.get(level, fallback_texts["正常"])


def complete_hamd_evaluation():
    """完成HAMD评估并生成报告"""
    global hamd_evaluator, hamd_evaluation_active
    
    hamd_evaluation_active = False
    
    if not hamd_evaluator:
        return
    
    # 生成评估报告
    report = hamd_evaluator.generate_report()
    
    # 语音播报结果
    total_score = report['total_score']
    severity = report['severity_level']
    level = severity['level']
    
    # 使用大模型生成个性化共情反馈
    result_text = generate_personalized_feedback(report)
    
    system_introduction(result_text)
    
    print(f"\n=== HAMD V3 评估完成 ===")
    print(f"评估ID: {report['evaluation_id']}")
    print(f"总分: {total_score}/{report['max_score']}")
    print(f"严重程度: {level}")
    print(f"风险等级: {report['risk_assessment']['overall_risk_level']}")
    print("=" * 50)


def Inference(TEMP_AUDIO_FILE=f"{OUTPUT_DIR}/audio_0.wav"):
    """
    语音推理主流程
    1. 使用SenceVoice做ASR
    2. 检测唤醒词
    3. HAMD评估或普通对话
    """
    global audio_file_count, set_KWS, flag_KWS, flag_KWS_used, hamd_evaluation_active
    
    # -------- SenceVoice 推理 ---------
    input_file = TEMP_AUDIO_FILE
    res = model_senceVoice.generate(
        input=input_file,
        cache={},
        language="auto",
        use_itn=False,
    )
    prompt = res[0]['text'].split(">")[-1]
    prompt_pinyin = extract_chinese_and_convert_to_pinyin(prompt)
    print(prompt, prompt_pinyin)

    # --- 判断是否启动KWS
    if not flag_KWS_used:
        flag_KWS = 1
    if not flag_KWS:
        if set_KWS in prompt_pinyin:
            flag_KWS = 1
    
    # --- KWS成功，或不设置KWS
    if flag_KWS:
        prompt_tmp = res[0]['text'].split(">")[-1]
        
        # --- 检查是否为HAMD评估模式 ---
        if hamd_evaluation_active:
            # 当前处于HAMD评估模式，识别意图
            print(f"HAMD评估模式 - 用户说: {prompt_tmp}")
            
            # 过滤过短的无效回答
            chinese_chars = re.findall(r'[\u4e00-\u9fa5]', prompt_tmp)
            if len(chinese_chars) < MIN_ANSWER_LENGTH and len(prompt_tmp.strip()) < MIN_ANSWER_LENGTH * 2:
                print(f"[调试] 回答过短，忽略")
                return
            
            # 获取当前问题上下文
            current_question = hamd_evaluator.get_current_question()
            question_context = ""
            if current_question:
                question_context = f"{current_question.get('title', '')} - {current_question.get('intent_description', '')}"
            
            intent_obj = classify_hamd_intent(prompt_tmp, question_context)
            intent = str(intent_obj.get("intent", "answer")).lower()
            print(f"[调试] 意图: {intent_obj}")
            
            if intent == "previous":
                if hamd_evaluator.previous_question():
                    system_introduction("好的，我们回到上一题。")
                    ask_next_hamd_question()
                else:
                    system_introduction("已经是第一题，无法再返回。")
            elif intent == "repeat":
                qp = hamd_evaluator.get_question_prompt()
                system_introduction(qp)
            elif intent in ("next", "skip"):
                if hamd_evaluator.next_question():
                    ask_next_hamd_question()
                else:
                    complete_hamd_evaluation()
            elif intent == "restart":
                hamd_evaluator.reset_evaluation()
                hamd_evaluator.start_evaluation()
                system_introduction("好的，我们重新开始评估。")
                ask_next_hamd_question()
            elif intent == "resume":
                qp = hamd_evaluator.get_question_prompt()
                if qp:
                    system_introduction(qp)
                else:
                    ask_next_hamd_question()
            elif intent == "stop":
                system_introduction("好的，已结束本次评估。")
                hamd_evaluation_active = False
                return
            elif intent == "switch_topic":
                system_introduction("好的，我们先暂停评估。")
                hamd_evaluation_active = False
                return
            else:
                # 按回答处理
                process_hamd_answer(prompt_tmp)
            
            return
        
        # --- 检查是否触发HAMD评估 ---
        if check_hamd_trigger(prompt_tmp):
            print("检测到心理评估触发词，开始HAMD V3评估")
            start_hamd_evaluation()
            return
        
        # --- 正常对话模式 ---
        context = memory.get_context()
        prompt = f"{context}\nUser:{prompt_tmp}\n"

        print("History:", context)
        print("ASR OUT:", prompt)
        
        # -------- 模型推理（云端千问） ------
        messages = [
            {"role": "system", "content": "你叫小千，是一个18岁的女大学生，性格活泼开朗，说话俏皮简洁，回答不超过50字。如果用户询问心理相关问题，可以建议他们说'心理评估'来开始专业测试。"},
            {"role": "user", "content": prompt},
        ]
        output_text = chat_api(messages, max_tokens=512)

        print("answer", output_text)

        # 更新记忆库
        memory.add_to_history(prompt_tmp, output_text)

        # TTS播报
        text = output_text
        used_speaker = DEFAULT_TTS_VOICE if VOICE_LOCK_ENABLED else "zh-CN-XiaoyiNeural"
        timestamp = int(time.time() * 1000)
        tts_file = os.path.join(folder_path, f"sft_{timestamp}.mp3")
        asyncio.run(amain(text, used_speaker, tts_file))
        play_audio(tts_file)
    else:
        text = "很抱歉，唤醒词错误，请说出正确的唤醒词哦"
        system_introduction(text)


# 主函数
if __name__ == "__main__":
    try:
        # 启动音频录制线程
        audio_thread = threading.Thread(target=audio_recorder)
        audio_thread.start()

        # 启动提示
        if flag_KWS_used:
            text = "您好，我是小千。我可以为您提供专业的心理评估服务。"
            if hamd_evaluator:
                text += "说'心理评估'就可以开始专业的抑郁量表测试了。"
            system_introduction(text)

        print("按 Ctrl+C 停止录制")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("录制停止中...")
        recording_active = False
        audio_thread.join()
        print("录制已停止")

