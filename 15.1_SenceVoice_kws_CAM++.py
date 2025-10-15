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
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
from funasr import AutoModel
import pygame
import edge_tts
import asyncio
from time import sleep
import langid
from langdetect import detect
import re
from pypinyin import pinyin, Style
from hamd_evaluator import HAMDEvaluator
from http import HTTPStatus
try:
    import dashscope
    from dashscope import Generation
except Exception:
    dashscope = None
    Generation = None

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
ANSWER_PAUSE_SECONDS = 2.5  # 回答阶段允许的停顿时长（HAMD模式）
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
# set_KWS = "ni hao xiao qian"
# set_KWS = "shuo hua xiao qian"
set_KWS = "zhan qi lai"
flag_KWS = 0

flag_KWS_used = 0
flag_sv_used = 0

thred_sv = 0.35  # 已不使用（兼容保留）

# HAMD 模式开关
hamd_mode = True

# -------- HAMD 全局状态 --------
hamd_evaluator = None
hamd_started = False
hamd_waiting_for_answer = False

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
        
        # 检查无效语音时间（根据模式动态阈值）
        if time.time() - last_active_time > get_silence_threshold():
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

    audio_file_count += 1
    audio_output_path = f"{OUTPUT_DIR}/audio_{audio_file_count}.wav"
    # audio_output_path = f"{OUTPUT_DIR}/audio_0.wav"

    if not segments_to_save:
        return
    
    # 停止当前播放的音频
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        print("检测到新的有效音，已停止当前音频播放")
        
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

# 已移除声纹识别依赖与调用
set_SV_enroll = r'.\SpeakerVerification_DIR\enroll_wav\\'  # 兼容遗留目录变量，不再使用

# --------- QWen2.5大语言模型 ---------------
# model_name = r"E:\2_PYTHON\Project\GPT\QWen\Qwen2.5-0.5B-Instruct"
model_name = r"Qwen/Qwen2.5-1.5B-Instruct"
# model_name = r'E:\2_PYTHON\Project\GPT\QWen\Qwen2.5-7B-Instruct-GPTQ-Int4'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# ---------- 模型加载结束 -----------------------

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
    used_speaker = "zh-CN-XiaoyiNeural"
    asyncio.run(amain(text, used_speaker, os.path.join(folder_path,f"sft_tmp_{audio_file_count}.mp3")))
    play_audio(f'{folder_path}/sft_tmp_{audio_file_count}.mp3')


def qwen_chat_adapter(messages, temperature=0.3, max_new_tokens=256):
    """
    优先使用 DashScope 云端千问；若不可用则回退到本地 Qwen。
    messages: List[{"role": str, "content": str}]
    """
    # 云端优先
    if dashscope is not None and Generation is not None:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if api_key:
            dashscope.api_key = api_key
            try:
                result = Generation.call(
                    model="qwen-plus",
                    messages=messages,
                    temperature=temperature,
                    result_format='message',
                    max_tokens=max_new_tokens,
                )
                if getattr(result, 'status_code', None) == HTTPStatus.OK:
                    return result.output.choices[0]['message']['content']
            except Exception as e:
                print("DashScope 调用异常，回退本地：", e)
        else:
            print("未设置 DASHSCOPE_API_KEY，使用本地模型。")

    # 本地回退
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text


def hamd_speak(text):
    global audio_file_count
    audio_file_count += 1
    used_speaker = "zh-CN-XiaoyiNeural"
    asyncio.run(amain(text, used_speaker, os.path.join(folder_path,f"hamd_{audio_file_count}.mp3")))
    play_audio(f'{folder_path}/hamd_{audio_file_count}.mp3')


def get_silence_threshold():
    """
    在 HAMD 提问-回答阶段：提问播放后，等待的用户回答允许更长停顿；
    在其他阶段或未开始时，使用基础阈值。
    """
    if hamd_mode and hamd_waiting_for_answer:
        return ANSWER_PAUSE_SECONDS
    return NO_SPEECH_THRESHOLD

def Inference(TEMP_AUDIO_FILE=f"{OUTPUT_DIR}/audio_0.wav"):
    '''
    1. 使用senceVoice做asr，转换为拼音，检测唤醒词
        - 首先检测声纹注册文件夹是否有注册文件，如果无，启动声纹注册
    2. 使用CAM++做声纹识别
        - 设置固定声纹注册语音目录，每次输入音频均进行声纹对比
    3. 以上两者均通过，则进行大模型推理
    '''
    global audio_file_count

    global set_SV_enroll
    global flag_sv_enroll
    global thred_sv
    global flag_sv_used

    global set_KWS
    global flag_KWS
    global flag_KWS_used
    
    # 在 HAMD 模式下：把每段有效语音作为一题的回答处理
    if hamd_mode:
        global hamd_evaluator, hamd_started, hamd_waiting_for_answer

        # 初始化并开始评估（只执行一次）
        if not hamd_started:
            hamd_evaluator = HAMDEvaluator(
                questions_file="hamd_questions.json",
                model_name=None,
                output_dir="./HAMD_Results/",
                chat_fn=qwen_chat_adapter
            )
            hamd_evaluator.start_evaluation()
            hamd_started = True
            hamd_waiting_for_answer = False
            # 播报开场与第一题
            intro = "现在开始进行哈密尔顿抑郁量表评估，我会逐一提问，请如实作答。"
            print(intro)
            hamd_speak(intro)
            current_q = hamd_evaluator.get_current_question()
            if current_q:
                speak_text = f"第{current_q['index']}题。" + hamd_evaluator.get_question_prompt()
                print("Q:", speak_text)
                hamd_speak(speak_text)
                hamd_waiting_for_answer = True
            return

        # 如果当前在等待回答，则将此音频识别为回答
        if hamd_waiting_for_answer:
            input_file = (TEMP_AUDIO_FILE)
            res = model_senceVoice.generate(
                input=input_file,
                cache={},
                language="auto",
                use_itn=False,
            )
            user_answer = res[0]['text'].split(">")[-1]
            print("User A:", user_answer)

            # 处理答案并评分
            result = hamd_evaluator.process_answer(user_answer)
            hamd_waiting_for_answer = False

            # 若完成则生成报告
            if result.get("is_complete"):
                report = hamd_evaluator.generate_report()
                summary = f"评估完成。总分{report['total_score']}/{report['max_score']}，严重程度：{report['severity_level']['level']}。"
                print(summary)
                hamd_speak(summary)
                # 简要分项播报
                try:
                    cats = report.get('category_analysis', {})
                    brief = []
                    for k, v in cats.items():
                        brief.append(f"{k}{v['score']}/{v['max_score']}")
                    if brief:
                        hamd_speak("分项：" + "，".join(brief))
                except Exception:
                    pass
                return

            # 未完成则播报下一题
            current_q = hamd_evaluator.get_current_question()
            if current_q:
                speak_text = f"第{current_q['index']}题。" + hamd_evaluator.get_question_prompt()
                print("Q:", speak_text)
                hamd_speak(speak_text)
                hamd_waiting_for_answer = True
            return

        # 如果不在等待回答，则重复当前题
        current_q = hamd_evaluator.get_current_question()
        if current_q:
            speak_text = f"第{current_q['index']}题。" + hamd_evaluator.get_question_prompt()
            print("Q:", speak_text)
            hamd_speak(speak_text)
            hamd_waiting_for_answer = True
        return

# 主函数
if __name__ == "__main__":

    try:
        # 启动音视频录制线程
        audio_thread = threading.Thread(target=audio_recorder)
        # video_thread = threading.Thread(target=video_recorder)
        audio_thread.start()
        # video_thread.start()

        if hamd_mode:
            system_introduction("将进行哈密尔顿抑郁量表语音评估，稍后请用简短自然语言回答问题。")
        else:
            flag_info = f'{flag_sv_used}-{flag_KWS_used}'
            dict_flag_info = {
                "1-1": "您已开启声纹识别和关键词唤醒，",
                "0-1":"您已开启关键词唤醒",
                "1-0":"您已开启声纹识别",
                "0-0":"",
            }
            if flag_sv_used or flag_KWS_used:
                text = dict_flag_info[flag_info]
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