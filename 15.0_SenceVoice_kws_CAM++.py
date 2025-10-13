
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
from modelscope.pipelines import pipeline
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


# --- 唤醒词、声纹变量配置 ---
set_KWS = "ni hao xiao qian"
# set_KWS = "shuo hua xiao qian"
# set_KWS = "zhan qi lai"
flag_KWS = 0

flag_KWS_used = 1
flag_sv_used = 0

flag_sv_enroll = 0
thred_sv = 0.35

# --- 统一TTS音色（可锁定全局） ---
DEFAULT_TTS_VOICE = "zh-CN-XiaoyiNeural"
VOICE_LOCK_ENABLED = True  # True 则所有播报固定用 DEFAULT_TTS_VOICE

# --- HAMD评估相关变量 ---
hamd_evaluator = None
hamd_evaluation_active = False
hamd_trigger_keywords = ["抑郁评估", "心理测试", "抑郁测试", "心理评估", "开始评估", "hamd", "抑郁量表"]

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
        
        # 检查无效语音时间
        if time.time() - last_active_time > NO_SPEECH_THRESHOLD:
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

    # 全局变量，用于保存音频文件名计数
    global audio_file_count
    global flag_sv_enroll
    global set_SV_enroll

    if flag_sv_enroll:
        audio_output_path = f"{set_SV_enroll}/enroll_0.wav"
    else:
        audio_file_count += 1
        audio_output_path = f"{OUTPUT_DIR}/audio_{audio_file_count}.wav"
    # audio_output_path = f"{OUTPUT_DIR}/audio_0.wav"

    if not segments_to_save:
        return
    
    # 如正在播报，则忽略本次语音段，避免打断 TTS
    if pygame.mixer.music.get_busy():
        print("正在播报，忽略本次语音段")
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
    if flag_sv_enroll:
        audio_length = 0.5 * len(segments_to_save)
        if audio_length < 3:
            print("声纹注册语音需大于3秒，请重新注册")
            return 1

    wf = wave.open(audio_output_path, 'wb')
    wf.setnchannels(AUDIO_CHANNELS)
    wf.setsampwidth(2)  # 16-bit PCM
    wf.setframerate(AUDIO_RATE)
    wf.writeframes(b''.join(audio_frames))
    wf.close()
    print(f"音频保存至 {audio_output_path}")

    # Inference()

    if flag_sv_enroll:
        text = "声纹注册完成！现在只有你可以命令我啦！"
        print(text)
        flag_sv_enroll = 0
        system_introduction(text)
    else:
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
            time.sleep(1)  # 等待音频播放结束
        print("播放完成！")
    except Exception as e:
        print(f"播放失败: {e}")
    finally:
        pygame.mixer.quit()

async def amain(TEXT, VOICE, OUTPUT_FILE) -> None:
    """Main function"""
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)

import os

def is_folder_empty(folder_path):
    """
    检测指定文件夹内是否有文件。
    
    :param folder_path: 文件夹路径
    :return: 如果文件夹为空返回 True，否则返回 False
    """
    # 获取文件夹中的所有条目（文件或子文件夹）
    entries = os.listdir(folder_path)
    # 检查是否存在文件
    for entry in entries:
        # 获取完整路径
        full_path = os.path.join(folder_path, entry)
        # 如果是文件，返回 False
        if os.path.isfile(full_path):
            return False
    # 如果没有文件，返回 True
    return True


# -------- SenceVoice 语音识别 --模型加载-----
model_dir = r"iic/SenseVoiceSmall"
model_senceVoice = AutoModel( model=model_dir, trust_remote_code=True, )

# -------- CAM++声纹识别 -- 模型加载 --------
set_SV_enroll = r'.\SpeakerVerification_DIR\enroll_wav\\'
sv_pipeline = pipeline(
    task='speaker-verification',
    model='damo/speech_campplus_sv_zh-cn_16k-common',
    model_revision='v1.0.0'
)

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
    global audio_file_count
    global folder_path
    text = text
    print("LLM output:", text)
    # 统一使用锁定音色
    used_speaker = DEFAULT_TTS_VOICE if VOICE_LOCK_ENABLED else "zh-CN-YunxiNeural"
    asyncio.run(amain(text, used_speaker, os.path.join(folder_path,f"sft_tmp_{audio_file_count}.mp3")))
    play_audio(f'{folder_path}/sft_tmp_{audio_file_count}.mp3')

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
    """使用LLM识别评估控制意图。返回字典: {intent, answer_text?, target_index?}
    intent 取值：answer | repeat | previous | next | skip | stop | start | restart | resume | switch_topic | jump
    """
    schema = (
        "你是心理评估会话的意图分类器。判断用户说的话是以下哪类: "
        "answer(回答问题)、repeat(重复当前题)、previous(上一题)、"
        "next(下一题)、skip(跳过此题)、stop(结束评估)、start(开始评估)、"
        "restart(重新开始评估)、resume(继续进行/恢复评估)、"
        "switch_topic(切换到日常聊天)、jump(跳转到指定题号，需字段target_index为整数)。请只用JSON返回，如: "
        "{\"intent\":\"answer\",\"answer_text\":\"...\"} 或 {\"intent\":\"previous\"} 或 {\"intent\":\"jump\",\"target_index\":1}."
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
            # 规范化 jump 的 target_index 类型
            if str(parsed.get("intent", "")).lower() == "jump":
                try:
                    if "target_index" in parsed:
                        parsed["target_index"] = int(parsed["target_index"])
                except Exception:
                    pass
            return parsed
    except Exception:
        pass
    # 回退：正则与拼音辅助识别（回到第一个问题 / di yi ge wen ti / 跳到第3题 等）
    try:
        import re as _re
        t = text.strip().lower()
        # 1) 中文数字与阿拉伯数字
        cn_digits = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
        m = _re.search(r"(回到|跳到|到)?第([一二三四五六七八九十0-9]+)(个)?(问题|题)", t)
        if m:
            num_raw = m.group(2)
            target = None
            if num_raw.isdigit():
                target = int(num_raw)
            else:
                if num_raw == "十":
                    target = 10
                elif num_raw.startswith("十") and len(num_raw) == 2:
                    target = 10 + cn_digits.get(num_raw[1], 0)
                elif num_raw.endswith("十") and len(num_raw) == 2:
                    target = cn_digits.get(num_raw[0], 0) * 10
                else:
                    target = cn_digits.get(num_raw, None)
            if target:
                return {"intent": "jump", "target_index": target}
        # 2) 拼音序数（di yi/er/san ... ge wen ti/ti），也兼容“hui dao/tiao dao/dao”
        pinyin_map = {"yi":1,"er":2,"san":3,"si":4,"wu":5,"liu":6,"qi":7,"ba":8,"jiu":9,"shi":10}
        m2 = _re.search(r"(hui\s*dao|tiao\s*dao|dao)?\s*di\s*(yi|er|san|si|wu|liu|qi|ba|jiu|shi)\s*(ge)?\s*(wen\s*ti|ti)", t)
        if m2:
            target = pinyin_map.get(m2.group(2), None)
            if target:
                return {"intent": "jump", "target_index": target}
        # 3) 重启/继续/上一题/下一题/跳过/重复/停止/开始（宽松）
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
    except Exception:
        pass
    # 默认视为回答
    return {"intent": "answer", "answer_text": text}

def start_hamd_evaluation():
    """开始HAMD评估"""
    global hamd_evaluator, hamd_evaluation_active
    
    if not hamd_evaluator:
        text = "抱歉，心理评估系统暂时不可用。"
        system_introduction(text)
        return False
    
    hamd_evaluation_active = True
    evaluation_id = hamd_evaluator.start_evaluation()
    
    intro_text = "好的，我将为您进行哈密尔顿抑郁量表评估。这个评估多个问题，用于了解您的心理健康状况。请如实回答每个问题，这将有助于更好地了解您的情况。"
    system_introduction(intro_text)
    
    # 开始第一个问题
    ask_next_hamd_question()
    return True

def ask_next_hamd_question():
    """询问下一个HAMD问题"""
    global hamd_evaluator
    
    if not hamd_evaluator or hamd_evaluator.is_evaluation_complete():
        return False
    
    current_question = hamd_evaluator.get_current_question()
    if current_question:
        question_prompt = hamd_evaluator.get_question_prompt()
        system_introduction(question_prompt)
        return True
    return False

def process_hamd_answer(user_answer):
    """处理HAMD评估回答"""
    global hamd_evaluator, hamd_evaluation_active
    
    if not hamd_evaluator:
        return False
    
    # 处理当前回答
    result = hamd_evaluator.process_answer(user_answer)
    
    if result.get("is_complete", False):
        # 评估完成，生成报告
        complete_hamd_evaluation()
    else:
        # 继续下一题
        ask_next_hamd_question()
    
    return True

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
    
    result_text = f"您的哈密尔顿抑郁量表评估已完成。总分为{total_score}分，评估结果为：{severity['level']}。{severity['description']}"
    
    # 添加主要建议
    if report['recommendations']:
        result_text += f"主要建议：{report['recommendations'][0]}"
    
    result_text += "详细的评估报告已保存到文件中，您可以查看完整的分析结果。"
    
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
        - 首先检测声纹注册文件夹是否有注册文件，如果无，启动声纹注册
    2. 使用CAM++做声纹识别
        - 设置固定声纹注册语音目录，每次输入音频均进行声纹对比
    3. 以上两者均通过，则进行大模型推理
    4. 新增：检测HAMD评估触发词，进行心理评估
    '''
    global audio_file_count

    global set_SV_enroll
    global flag_sv_enroll
    global thred_sv
    global flag_sv_used

    global set_KWS
    global flag_KWS
    global flag_KWS_used
    
    global hamd_evaluation_active
    
    os.makedirs(set_SV_enroll, exist_ok=True)
    # --- 如果开启声纹识别，且声纹文件夹为空，则开始声纹注册。设定注册语音有效长度需大于3秒
    if flag_sv_used and is_folder_empty(set_SV_enroll):
        text = f"无声纹注册文件！请先注册声纹，需大于三秒哦~"
        print(text)
        system_introduction(text)
        flag_sv_enroll = 1
    
    else:
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
            # 关闭声纹验证：直接通过
            sv_result = "yes"
            if sv_result == "yes":
                prompt_tmp = res[0]['text'].split(">")[-1]
                
                # --- 检查是否为HAMD评估模式 ---
                if hamd_evaluation_active:
                    # 当前处于HAMD评估模式，使用LLM识别意图
                    print(f"HAMD评估模式 - 用户说: {prompt_tmp}")
                    intent_obj = classify_hamd_intent(prompt_tmp)
                    intent = str(intent_obj.get("intent", "answer")).lower()
                    handled_in_eval = True
                    # 精确跳题（到第X题）
                    if intent == "jump":
                        target_index = intent_obj.get("target_index")
                        if isinstance(target_index, int) and target_index >= 1:
                            if hamd_evaluator.jump_to_question(target_index):
                                system_introduction(f"好的，我们回到第{target_index}个问题。")
                                ask_next_hamd_question()
                            else:
                                system_introduction("跳转失败，请再说一次题号。")
                        else:
                            system_introduction("没有听清题号，请再说一次。")
                    elif intent == "previous":
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
                    elif intent == "switch_topic":
                        # 切换到日常对话
                        system_introduction("好的，我们先暂停评估，继续日常对话。")
                        hamd_evaluation_active = False
                        handled_in_eval = False
                    else:
                        # 按回答处理
                        answer_text = intent_obj.get("answer_text", prompt_tmp)
                        process_hamd_answer(answer_text)
                    if handled_in_eval:
                        return  # 仍在评估流中则直接返回
                    
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

                asyncio.run(amain(text, used_speaker, os.path.join(folder_path,f"sft_{audio_file_count}.mp3")))
                play_audio(f'{folder_path}/sft_{audio_file_count}.mp3')
            else:
                text = "很抱歉，声纹验证失败，我无法为您服务"
                print(text)
                # system_introduction(text)
        else:
            text = "很抱歉，唤醒词错误，请说出正确的唤醒词哦"
            system_introduction(text)

# 主函数
if __name__ == "__main__":

    try:
        # 启动音视频录制线程
        audio_thread = threading.Thread(target=audio_recorder)
        # video_thread = threading.Thread(target=video_recorder)
        audio_thread.start()
        # video_thread.start()

        flag_info = f'{flag_sv_used}-{flag_KWS_used}'
        dict_flag_info = {
            "1-1": "您已开启声纹识别和关键词唤醒，",
            "0-1":"您已开启关键词唤醒",
            "1-0":"您已开启声纹识别",
            "0-0":"",
        }
        if flag_sv_used or flag_KWS_used:
            text = dict_flag_info[flag_info]
            if hamd_evaluator:
                text += "我还可以为您提供专业的心理健康评估服务。"
            system_introduction(text)

        print("按 Ctrl+C 停止录制")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("录制停止中...")
        recording_active = False
        audio_thread.join()
        # video_thread.join()
        print("录制已停止")
