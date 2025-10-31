
import cv2
import pyaudio
import wave
import threading
import numpy as np
import time
from queue import Queue
import webrtcvad
import os
import threading
from qwen_vl_utils import process_vision_info
import torch
from funasr import AutoModel
import pygame
from openai import OpenAI
import edge_tts
import asyncio
from time import sleep
import langid
from langdetect import detect
import re
from pypinyin import pinyin, Style
from hamd_evaluator import HAMDEvaluator

# --- 配置huggingFace国内镜像 ---
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 参数设置
AUDIO_RATE = 16000        # 音频采样率
AUDIO_CHANNELS = 1        # 单声道
CHUNK = 1024              # 音频块大小
VAD_MODE = 3              # VAD 模式 (0-3, 数字越大越敏感)
OUTPUT_DIR = "./output"   # 输出目录
NO_SPEECH_THRESHOLD = 1   # 无效语音阈值，单位：秒
folder_path = "./Test_QWen2_VL/"
audio_file_count = 0
audio_file_count_tmp = 0

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
last_vad_end_time = 0  # 上次保存的 VAD 有效段结束时间


# --- 唤醒词变量配置 ---
set_KWS = "ni hao xiao qian"
# set_KWS = "shuo hua xiao qian"
# set_KWS = "zhan qi lai"
flag_KWS = 0

flag_KWS_used = 1

# --- 统一TTS音色（可锁定全局） ---
DEFAULT_TTS_VOICE = "zh-CN-XiaoyiNeural"
VOICE_LOCK_ENABLED = True  # True 则所有播报固定用 DEFAULT_TTS_VOICE

# --- HAMD评估相关变量 ---
hamd_evaluator = None
hamd_evaluation_active = False
hamd_trigger_keywords = ["抑郁评估", "心理测试", "抑郁测试", "心理评估", "开始评估", "hamd", "抑郁量表"]
hamd_waiting_for_answer = False  # 标记是否正在等待用户回答HAMD问题
hamd_processing_lock = False  # 标记HAMD正在处理中（TTS生成、播放等），阻止新推理
ANSWER_PAUSE_SECONDS = 3.5  # 等待用户回答时的停顿时间（秒）
MIN_ANSWER_LENGTH = 3  # HAMD评估模式下，最短有效回答长度（字符数）

# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)


def extract_chinese_and_convert_to_pinyin(input_string):
    """
    提取字符串中的汉字，并将其转换为拼音。
    
    :param input_string: 原始字符串
    :return: 转换后的拼音字符串
    """
    # 使用正则表达式提取所有汉字
    chinese_characters = re.findall(r'[\u4e00-\u9fa5]', input_string)
    # 将汉字列表合并为字符串
    chinese_text = ''.join(chinese_characters)
    
    # 转换为拼音
    pinyin_result = pinyin(chinese_text, style=Style.NORMAL)
    # 将拼音列表拼接为字符串
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
            # 拼接音频数据并检测 VAD
            raw_audio = b''.join(audio_buffer)
            vad_result = check_vad_activity(raw_audio)
            
            if vad_result:
                print("检测到语音活动")
                last_active_time = time.time()
                segments_to_save.append((raw_audio, time.time()))
            else:
                print("静音中...")
            
            audio_buffer = []  # 清空缓冲区
        
        # 检查无效语音时间（动态阈值：评估模式下等待时间更长）
        threshold = ANSWER_PAUSE_SECONDS if (hamd_waiting_for_answer) else NO_SPEECH_THRESHOLD
        if time.time() - last_active_time > threshold:
            # 检查是否需要保存
            if segments_to_save and segments_to_save[-1][1] > last_vad_end_time:
                save_audio_video()
                last_active_time = time.time()
            else:
                pass
                # print("无新增语音段，跳过保存")
    
    stream.stop_stream()
    stream.close()
    p.terminate()

# 视频录制线程
def video_recorder():
    global video_queue, recording_active
    
    cap = cv2.VideoCapture(0)  # 使用默认摄像头
    print("视频录制已开始")
    
    while recording_active:
        ret, frame = cap.read()
        if ret:
            video_queue.put((frame, time.time()))
            
            # 实时显示摄像头画面
            cv2.imshow("Real Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 Q 键退出
                break
        else:
            print("无法获取摄像头画面")
    
    cap.release()
    cv2.destroyAllWindows()

# 检测 VAD 活动
def check_vad_activity(audio_data):
    # 将音频数据分块检测
    num, rate = 0, 0.5
    step = int(AUDIO_RATE * 0.02)  # 20ms 块大小
    flag_rate = round(rate * len(audio_data) // step)

    for i in range(0, len(audio_data), step):
        chunk = audio_data[i:i + step]
        if len(chunk) == step:
            if vad.is_speech(chunk, sample_rate=AUDIO_RATE):
                num += 1

    if num > flag_rate:
        return True
    return False

# 保存音频和视频
def save_audio_video():
    pygame.mixer.init()

    global segments_to_save, video_queue, last_vad_end_time, saved_intervals
    global audio_file_count

    audio_file_count += 1
    audio_output_path = f"{OUTPUT_DIR}/audio_{audio_file_count}.wav"

    if not segments_to_save:
        return
    
    # 如正在播报，则忽略本次语音段，避免打断 TTS
    if pygame.mixer.music.get_busy():
        print("正在播报，忽略本次语音段")
        segments_to_save.clear()
        return
    
    # 如果HAMD正在处理中（TTS生成/播放等），阻止新的推理
    if hamd_processing_lock:
        print("HAMD正在处理中，忽略本次语音段")
        segments_to_save.clear()
        return
        
    # 获取有效段的时间范围
    start_time = segments_to_save[0][1]
    end_time = segments_to_save[-1][1]
    
    # 检查是否与之前的片段重叠
    if saved_intervals and saved_intervals[-1][1] >= start_time:
        print("当前片段与之前片段重叠，跳过保存")
        segments_to_save.clear()
        return
    
    # 保存音频
    audio_frames = [seg[0] for seg in segments_to_save]
    wf = wave.open(audio_output_path, 'wb')
    wf.setnchannels(AUDIO_CHANNELS)
    wf.setsampwidth(2)  # 16-bit PCM
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

# --- 播放音频 -
def play_audio(file_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)  # 等待音频播放结束
        print("播放完成！")
        # 播放完成后额外等待0.5秒，确保音频设备完全释放，避免捕获残余音频
        time.sleep(0.5)
    except Exception as e:
        print(f"播放失败: {e}")
    finally:
        pygame.mixer.quit()

async def amain(TEXT, VOICE, OUTPUT_FILE) -> None:
    """Main function"""
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)


# -------- SenceVoice 语音识别 --模型加载-----
model_dir = r"iic/SenseVoiceSmall"
model_senceVoice = AutoModel( model=model_dir, trust_remote_code=True, )

# -------- CAM++声纹识别 -- 已禁用（避免SSL错误）--------
# 如需启用声纹识别，请确保模型已下载到本地
# set_SV_enroll = r'.\SpeakerVerification_DIR\enroll_wav\\'
# sv_pipeline = pipeline(
#     task='speaker-verification',
#     model='damo/speech_campplus_sv_zh-cn_16k-common',
#     model_revision='v1.0.0'
# )

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
# ---------- 适配结束 -----------------------

# 初始化HAMD评估器（云端千问）
try:
    hamd_evaluator = HAMDEvaluator(
        questions_file="hamd_questions.json",
        model_name=None,
        output_dir="./HAMD_Results/",
        chat_fn=lambda messages, temperature=0.3, max_tokens=256: chat_api(messages, temperature=temperature, max_tokens=max_tokens)
    )
    print("HAMD评估器初始化成功（云端千问）")
except Exception as e:
    print(f"HAMD评估器初始化失败: {e}")
    hamd_evaluator = None

class ChatMemory:
    def __init__(self, max_length=2048):
        self.history = []
        self.max_length = max_length  # 最大输入长度

    def add_to_history(self, user_input, model_response):
        """
        添加用户输入和模型响应到历史记录。
        """
        self.history.append(f"User: {user_input}")
        self.history.append(f"system: {model_response}")

    def get_context(self):
        """
        获取拼接后的对话上下文。
        """
        context = "\n".join(self.history)
        # 截断上下文，使其不超过 max_length
        if len(context) > self.max_length:
            context = context[-self.max_length :]
        return context
    
# -------- memory 初始化 --------
memory = ChatMemory(max_length=512)

def system_introduction(text):
    global folder_path
    text = text
    print("LLM output:", text)
    # 统一使用锁定音色
    used_speaker = DEFAULT_TTS_VOICE if VOICE_LOCK_ENABLED else "zh-CN-YunxiNeural"
    # 使用时间戳确保文件名唯一
    timestamp = int(time.time() * 1000)
    tts_file = os.path.join(folder_path, f"sft_tmp_{timestamp}.mp3")
    asyncio.run(amain(text, used_speaker, tts_file))
    play_audio(tts_file)

def check_hamd_trigger(text):
    """检查是否触发HAMD评估"""
    global hamd_trigger_keywords
    text_lower = text.lower()
    # 规范化常见听写/错别字
    def _normalize(s: str) -> str:
        s = s.replace("心里", "心理").replace("抑于", "抑郁").replace("抑以", "抑郁").replace("一于", "抑郁").replace("抑于评估", "抑郁评估")
        s = s.replace("评古", "评估").replace("测是", "测试").replace("策试", "测试")
        return s
    text_lower = _normalize(text_lower)
    
    # 更严格的匹配：必须包含明确的评估相关词汇
    evaluation_words = ["评估", "测试", "量表", "hamd"]
    depression_words = ["抑郁", "心理", "情绪", "心情"]
    
    # 检查是否同时包含评估词和抑郁相关词，或者包含明确的触发词
    has_evaluation = any(word in text_lower for word in evaluation_words)
    has_depression = any(word in text_lower for word in depression_words)
    
    # 直接匹配明确的触发关键词（包含强触发词：心理测试/抑郁测试/开始抑郁评估 等）
    strong_triggers = ["心理测试", "抑郁测试", "开始抑郁评估", "开始心理评估", "开始测试"]
    for keyword in (hamd_trigger_keywords + strong_triggers):
        if keyword in text_lower:
            return True
    
    # 或者同时包含评估词和抑郁词
    return has_evaluation and has_depression

def classify_hamd_intent(text):
    """使用LLM识别评估控制意图。返回字典: {intent}
    intent 取值：answer | repeat | previous | next | skip | stop | start | restart | resume | switch_topic
    """
    schema = (
        "你是心理评估意图分类器。当前用户正在进行心理评估问答。判断用户说的话属于哪类意图:\n"
        "- answer: 回答评估问题（描述症状、感受、情况等，这是默认意图）\n"
        "- repeat: 要求重复当前问题\n"
        "- previous: 返回上一题\n"
        "- next/skip: 跳过当前题\n"
        "- stop: 明确要求结束/停止评估\n"
        "- switch_topic: 明确要求暂停评估去聊别的话题（必须有明确的切换意图）\n\n"
        "注意：只有当用户明确表示「不想做评估了」「我们聊点别的」时才是switch_topic。\n"
        "用户描述自己的症状、心情、状况等都属于answer，即使内容很长或很详细。\n\n"
        "只返回JSON: {\"intent\":\"answer\"} 或 {\"intent\":\"stop\"} 等"
    )
    messages = [
        {"role": "system", "content": schema},
        {"role": "user", "content": text},
    ]
    raw = chat_api(messages, max_tokens=128)
    try:
        import json as _json
        # 尝试直接解析
        parsed = _json.loads(raw)
        if isinstance(parsed, dict) and "intent" in parsed:
            return parsed
    except Exception:
        pass
    # 回退：正则与拼音辅助识别
    try:
        import re as _re
        t = text.strip().lower()
        # 重启/继续/上一题/下一题/跳过/重复/停止/开始（宽松）
        # 重启
        if any(k in t for k in ["重新开始评估", "重新开始", "重来", "从头开始", "重新来一次", "restart", "start over",
                                 "chong xin kai shi", "chong lai", "cong tou kai shi"]):
            return {"intent": "restart"}
        # 继续/恢复
        if any(k in t for k in ["继续评估", "继续", "接着来", "恢复评估", "resume", "继续刚才", "ji xu"]):
            return {"intent": "resume"}
        if any(k in t for k in ["上一题", "上一个问题", "上一个", "上一道", "上题", "previous"]):
            return {"intent": "previous"}
        if any(k in t for k in ["下一题", "下一个", "跳过", "skip", "next"]):
            return {"intent": "next"}
        if any(k in t for k in ["重复", "再说一遍", "再读一遍", "重复一下", "重新说一下这个问题", "repeat",
                                 "chong fu", "chong xin shuo yi xia"]):
            return {"intent": "repeat"}
        if any(k in t for k in ["停止", "结束评估", "先停一下", "提前结束测试", "提前结束评估", "结束测试", "stop", "end",
                                 "ting zhi", "jie shu ce shi"]):
            return {"intent": "stop"}
        if any(k in t for k in ["开始评估", "做个评估", "心理评估", "start evaluation", "start"]):
            return {"intent": "start"}
        # switch_topic需要非常明确的切换意图（避免误判用户的正常回答）
        if any(k in t for k in ["不想做评估", "我们聊点别的", "暂停评估", "先不做了", "换个话题", "聊聊天", 
                                 "bu xiang zuo ping gu", "zan ting ping gu"]):
            return {"intent": "switch_topic"}
    except Exception:
        pass
    # 默认视为回答（不返回answer_text，外部会直接使用原始输入）
    return {"intent": "answer"}

def start_hamd_evaluation():
    """开始HAMD评估"""
    global hamd_evaluator, hamd_evaluation_active
    
    if not hamd_evaluator:
        text = "抱歉，心理评估系统暂时不可用。"
        system_introduction(text)
        return False
    
    hamd_evaluation_active = True
    evaluation_id = hamd_evaluator.start_evaluation()
    
    intro_text = "好的，我将为您进行心理评估，用于了解您的心理健康状况。请如实回答每个问题，这将有助于更好地了解您的情况。"
    system_introduction(intro_text)
    
    # 开始第一个问题
    ask_next_hamd_question()
    return True

def ask_next_hamd_question():
    """询问下一个HAMD问题"""
    global hamd_evaluator, hamd_waiting_for_answer, hamd_processing_lock
    
    # 设置处理锁
    hamd_processing_lock = True
    
    try:
        if not hamd_evaluator or hamd_evaluator.is_evaluation_complete():
            hamd_waiting_for_answer = False
            return False
        
        current_question = hamd_evaluator.get_current_question()
        if current_question:
            question_prompt = hamd_evaluator.get_question_prompt()
            hamd_waiting_for_answer = True  # 设置标志：正在等待用户回答
            system_introduction(question_prompt)
            return True
        hamd_waiting_for_answer = False
        return False
    
    finally:
        # 释放处理锁
        hamd_processing_lock = False
        print("[调试] 问题播放完成，释放锁")

def process_hamd_answer(user_answer):
    """处理HAMD评估回答（支持两阶段逻辑）"""
    global hamd_evaluator, hamd_evaluation_active, hamd_waiting_for_answer, hamd_processing_lock
    
    # 设置处理锁，阻止新的推理
    hamd_processing_lock = True
    
    try:
        # 清除等待标志
        hamd_waiting_for_answer = False
        
        if not hamd_evaluator:
            return False
        
        # 处理当前回答
        result = hamd_evaluator.process_answer(user_answer)
        
        # 在评分之后，用大模型生成一句安慰/共情回应
        try:
            sys_prompt = (
                "你是心理评估助手。请对用户的回答给出一句简短的共情性回应，"
                "使用温暖、支持的语气，不超过25个字；"
                "不要提问；不要给建议；不要复述评分；不要提及分数；"
                "避免医学诊断或指导；只用中文。"
            )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"用户刚才说：{user_answer}"},
            ]
            comfort = chat_api(messages, temperature=0.7, max_tokens=50)
            
            # 清洗生成内容
            if comfort and isinstance(comfort, str):
                comfort = comfort.replace("\n", "").strip()
                # 如果包含问号或疑问句，使用默认回应
                if ("?" in comfort) or ("？" in comfort) or comfort.endswith("吗"):
                    comfort = "我理解，这很不容易。"
                # 限制长度
                if len(comfort) > 30:
                    comfort = comfort[:30]
            else:
                comfort = "我理解，这很不容易。"
            
            system_introduction(comfort)
        except Exception as e:
            print(f"生成共情回应失败: {e}")
            # 失败时使用默认回应
            try:
                system_introduction("我理解，这很不容易。")
            except Exception:
                pass
        
        # 检查是否需要阶段转换
        if result.get("stage_transition_needed", False):
            # 阶段1完成，询问是否继续阶段2
            transition_prompt = hamd_evaluator.get_stage_transition_prompt()
            system_introduction(transition_prompt)
            return True
        
        if result.get("is_complete", False):
            # 评估完成，生成报告
            complete_hamd_evaluation()
        else:
            # 继续下一题
            ask_next_hamd_question()
        
        return True
    
    finally:
        # 释放处理锁，允许新的推理
        hamd_processing_lock = False
        print("[调试] HAMD处理完成，释放锁")

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
    
    # 根据不同评估结果，生成更人性化的播报话语
    if level == "正常":
        result_text = f"感谢您的配合，评估已经完成了。从测评结果来看，您目前的心理状态整体比较稳定，没有明显的抑郁症状。"
        result_text += "继续保持积极乐观的心态，注意劳逸结合。如果将来遇到困扰，随时可以找我聊聊。"
    elif level == "轻度抑郁":
        result_text = f"评估完成了，谢谢您的真诚回答。从结果来看，您可能正在经历一些轻度的情绪困扰。"
        result_text += "这是可以理解的，生活中难免会有起起伏伏。建议您可以多和亲友交流，保持规律作息，适当运动放松。"
        result_text += "如果情况持续或加重，建议寻求专业心理咨询师的帮助。"
    elif level == "中度抑郁":
        result_text = f"评估已经完成，非常感谢您的信任和配合。测评显示您目前可能承受着较大的心理压力。"
        result_text += "我理解这段时间对您来说可能很不容易。建议您尽快寻求专业的心理咨询或医疗帮助，"
        result_text += "专业人士可以为您提供更有针对性的支持和治疗方案。同时，也请多关注自己的身心健康，不要独自承受。"
    else:  # 重度抑郁
        result_text = f"评估已经完成，感谢您愿意和我分享这些。测评结果表明您当前的状况需要得到专业关注。"
        result_text += "我非常理解您现在可能正经历着很大的困难，这不是您的错。"
        result_text += "请您尽快联系专业的心理医生或前往医院的心理科就诊，寻求专业的医疗帮助。"
        result_text += "您并不孤单，有很多人和资源可以帮助您。请照顾好自己，您值得被关心和帮助。"
    
    result_text += f"\n详细的评估报告已经保存好了，您可以随时查看。"
    
    system_introduction(result_text)
    
    print(f"\n=== HAMD评估完成 ===")
    print(f"评估ID: {report['evaluation_id']}")
    print(f"总分: {total_score}/{report['max_score']}")
    print(f"严重程度: {severity['level']}")
    print(f"建议: {'; '.join(report['recommendations'][:3])}")
    print("=" * 50)

def Inference(TEMP_AUDIO_FILE=f"{OUTPUT_DIR}/audio_0.wav"):
    '''
    1. 使用senceVoice做asr，转换为拼音，检测唤醒词
    2. 通过唤醒词检测后进行大模型推理
    3. 支持HAMD心理评估功能
    '''
    global audio_file_count
    global set_KWS
    global flag_KWS
    global flag_KWS_used
    global hamd_evaluation_active
    
    # -------- SenceVoice 推理 ---------
    input_file = (TEMP_AUDIO_FILE)
    res = model_senceVoice.generate(
        input=input_file,
        cache={},
        language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
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
            # 检查是否正在等待阶段决策
            if hamd_evaluator.waiting_stage_decision:
                # 用户正在回答是否继续阶段2的问题
                user_response = prompt_tmp.lower()
                
                # 扩展识别"同意继续"的表达
                agree_keywords = [
                    # 直接肯定
                    "是", "好", "可以", "行", "嗯", "对", "要", "愿意",
                    # 继续相关
                    "继续", "开始", "进行", "做", "测", "评估",
                    # 组合表达
                    "那开始", "那就", "来吧", "开始吧", "继续吧", "可以的", "没问题", "当然",
                    # 英文/拼音
                    "yes", "ok", "yeah", "sure",
                    "shi", "hao", "ke yi", "ji xu", "kai shi", "na kai shi"
                ]
                
                # 识别"拒绝"的表达
                reject_keywords = [
                    "不", "不用", "不要", "算了", "不做", "不测", "不继续",
                    "跳过", "结束", "停", "不愿意", "不想",
                    "no", "bu", "bu yong", "suan le"
                ]
                
                # 先检查是否明确拒绝
                is_reject = any(word in user_response for word in reject_keywords)
                # 再检查是否同意
                is_agree = any(word in user_response for word in agree_keywords)
                
                if is_agree and not is_reject:
                    # 用户同意继续阶段2
                    hamd_evaluator.enter_stage2()
                    hamd_evaluator.current_question_index = 0  # 重置索引到阶段2的第一题
                    system_introduction("好的，我们开始更详细的评估。")
                    ask_next_hamd_question()
                else:
                    # 用户拒绝或不确定，跳过阶段2
                    hamd_evaluator.skip_stage2()
                    system_introduction("好的，我们结束本次评估。")
                    complete_hamd_evaluation()
                return
            
            # 当前处于HAMD评估模式，使用LLM识别意图
            print(f"HAMD评估模式 - 用户说: {prompt_tmp}")
            
            # 过滤过短的无效回答（如环境杂音、简短确认词等）
            # 提取中文字符计算长度
            chinese_chars = re.findall(r'[\u4e00-\u9fa5]', prompt_tmp)
            if len(chinese_chars) < MIN_ANSWER_LENGTH and len(prompt_tmp.strip()) < MIN_ANSWER_LENGTH * 2:
                print(f"[调试] 回答过短（{len(chinese_chars)}个汉字，{len(prompt_tmp)}个字符），忽略")
                return
            
            intent_obj = classify_hamd_intent(prompt_tmp)
            intent = str(intent_obj.get("intent", "answer")).lower()
            print(f"[调试] 意图分类结果: {intent_obj}")
            
            if intent == "previous":
                if hamd_evaluator.previous_question():
                    system_introduction("好的，我们回到上一题。")
                    ask_next_hamd_question()
                else:
                    system_introduction("已经是第一题，无法再返回。")
            elif intent in ("repeat",):
                # 重复当前题
                qp = hamd_evaluator.get_question_prompt()
                system_introduction(qp)
            elif intent in ("next", "skip"):
                if hamd_evaluator.next_question():
                    ask_next_hamd_question()
                else:
                    complete_hamd_evaluation()
            elif intent == "restart":
                # 重新开始评估
                hamd_evaluator.reset_evaluation()
                system_introduction("好的，我们重新开始评估。")
                ask_next_hamd_question()
            elif intent == "resume":
                # 继续评估：如果已经在评估中，则重复当前题；若非活动状态，上层会重新启动
                qp = hamd_evaluator.get_question_prompt()
                if qp:
                    system_introduction(qp)
                else:
                    ask_next_hamd_question()
            elif intent == "stop":
                # 结束评估
                system_introduction("好的，已结束本次评估。若需要，随时可以让我重新开始。")
                hamd_evaluation_active = False
                return  # 结束评估后直接返回
            elif intent == "switch_topic":
                # 切换到日常对话
                system_introduction("好的，我们先暂停评估，有需要随时叫我继续。")
                hamd_evaluation_active = False
                return  # 切换话题后直接返回，不执行后续流程
            else:
                # 按回答处理 - 直接使用原始用户输入，不使用LLM可能改写的answer_text
                process_hamd_answer(prompt_tmp)
            # 评估流程处理完毕，直接返回
            return
        
        # --- 检查是否触发HAMD评估（显式关键词立即开始） ---
        if check_hamd_trigger(prompt_tmp):
            print("检测到心理评估触发词，开始HAMD评估")
            start_hamd_evaluation()
            return  # 开始评估后直接返回
        
        # --- 正常对话模式 ---
        # --- 读取历史对话 ---
        context = memory.get_context()
        
        prompt = f"{context}\nUser:{prompt_tmp}\n"

        print("History:", context)
        print("ASR OUT:", prompt)
        # ---------SenceVoice --end----------
        # -------- 模型推理阶段（云端千问） ------
        messages = [
            {"role": "system", "content": "你叫小千，是一个18岁的女大学生，性格活泼开朗，说话俏皮简洁，回答问题不会超过50字。如果用户询问心理健康相关问题，你可以建议他们说'心理评估'来开始专业的抑郁量表测试。"},
            {"role": "user", "content": prompt},
        ]
        output_text = chat_api(messages, max_tokens=512)

        print("answer", output_text)

        # -------- 更新记忆库 -----
        memory.add_to_history(prompt_tmp, output_text)

        # 输入文本
        text = output_text
        # 语种识别 -- langid
        language, confidence = langid.classify(text)
        # 语种识别 -- langdetect 
        # language = detect(text).split("-")[0]

        language_speaker = {
        "ja" : "ja-JP-NanamiNeural",            # ok
        "fr" : "fr-FR-DeniseNeural",            # ok
        "es" : "ca-ES-JoanaNeural",             # ok
        "de" : "de-DE-KatjaNeural",             # ok
        "zh" : "zh-CN-XiaoyiNeural",            # ok
        "en" : "en-US-AnaNeural",               # ok
        }

        # 全局音色锁优先；否则评估中固定中文女声；否则按语种
        if VOICE_LOCK_ENABLED:
            used_speaker = DEFAULT_TTS_VOICE
        elif 'hamd_evaluation_active' in globals() and hamd_evaluation_active:
            used_speaker = DEFAULT_TTS_VOICE
        else:
            if language not in language_speaker.keys():
                used_speaker = DEFAULT_TTS_VOICE
            else:
                used_speaker = language_speaker[language]
                print("检测到语种：", language, "使用音色：", language_speaker[language])

        # 使用时间戳确保文件名唯一
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
            text = "您已开启关键词唤醒。"
            if hamd_evaluator:
                text += "我还可以为您提供专业的心理健康评估服务。"
            system_introduction(text)
        elif hamd_evaluator:
            text = "您好，我是小千。我可以为您提供专业的心理健康评估服务。"
            system_introduction(text)

        print("按 Ctrl+C 停止录制")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("录制停止中...")
        recording_active = False
        audio_thread.join()
        print("录制已停止")