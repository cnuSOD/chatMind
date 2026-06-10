"""麦克风采集与VAD断句

把原15.2脚本中基于全局变量的录音线程封装为类：
- 持续录音，每0.5秒VAD投票判断是否有人声
- 静音超过阈值即认为一句话说完，写wav并回调
- 通过注入的回调与外部解耦（是否暂停采集、动态停顿阈值、识别结果处理）
"""

import logging
import os
import threading
import time
import wave

import pyaudio
import webrtcvad

logger = logging.getLogger("chatmind")

AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
CHUNK = 1024


class AudioCapture:
    def __init__(self, on_utterance, output_dir="./output",
                 vad_mode=3, no_speech_threshold=1.0,
                 pause_threshold_fn=None, should_ignore_fn=None):
        """
        Args:
            on_utterance: 回调 fn(wav_path)，一句话保存后调用（在独立线程执行）
            output_dir: wav输出目录
            vad_mode: WebRTC VAD灵敏度 0-3
            no_speech_threshold: 默认静音断句阈值（秒）
            pause_threshold_fn: 可选，返回当前应使用的静音阈值（如答题时放宽）
            should_ignore_fn: 可选，返回True时丢弃当前语音段（如正在播报）
        """
        self.on_utterance = on_utterance
        self.output_dir = output_dir
        self.no_speech_threshold = no_speech_threshold
        self.pause_threshold_fn = pause_threshold_fn or (lambda: no_speech_threshold)
        self.should_ignore_fn = should_ignore_fn or (lambda: False)

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(vad_mode)

        self._running = False
        self._thread = None
        self._segments = []
        self._last_active_time = time.time()
        self._last_saved_end = 0.0
        self._file_count = 0

        os.makedirs(output_dir, exist_ok=True)

    # ---------- 公共接口 ----------

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        logger.info("音频采集已启动")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("音频采集已停止")

    # ---------- 内部实现 ----------

    def _is_speech(self, raw_audio: bytes) -> bool:
        """对0.5秒音频按20ms帧投票"""
        step = int(AUDIO_RATE * 0.02)
        votes, frames = 0, 0
        for i in range(0, len(raw_audio), step):
            frame = raw_audio[i:i + step]
            if len(frame) == step:
                frames += 1
                if self.vad.is_speech(frame, sample_rate=AUDIO_RATE):
                    votes += 1
        return frames > 0 and votes > frames * 0.5

    def _record_loop(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=AUDIO_CHANNELS,
                        rate=AUDIO_RATE, input=True, frames_per_buffer=CHUNK)
        buffer = []
        try:
            while self._running:
                buffer.append(stream.read(CHUNK))

                # 攒够0.5秒做一次VAD
                if len(buffer) * CHUNK / AUDIO_RATE >= 0.5:
                    raw = b"".join(buffer)
                    buffer = []
                    if self._is_speech(raw):
                        self._last_active_time = time.time()
                        self._segments.append((raw, time.time()))

                # 静音超过阈值 -> 认为一句话结束
                threshold = self.pause_threshold_fn()
                if time.time() - self._last_active_time > threshold:
                    if self._segments and self._segments[-1][1] > self._last_saved_end:
                        self._flush_utterance()
                        self._last_active_time = time.time()
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def _flush_utterance(self):
        segments, self._segments = self._segments, []
        if not segments:
            return
        if self.should_ignore_fn():
            logger.debug("系统忙（播报/处理中），丢弃本段语音")
            return

        end_time = segments[-1][1]
        if end_time <= self._last_saved_end:
            return
        self._last_saved_end = end_time

        self._file_count += 1
        wav_path = os.path.join(self.output_dir, f"audio_{self._file_count}.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_RATE)
            wf.writeframes(b"".join(seg[0] for seg in segments))
        logger.debug(f"语音段已保存: {wav_path}")

        threading.Thread(target=self.on_utterance, args=(wav_path,), daemon=True).start()
