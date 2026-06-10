"""千问 DashScope（OpenAI兼容）客户端封装"""

import logging
import os
import time

from openai import OpenAI

logger = logging.getLogger("chatmind")

_client = None
_config = {"model": "qwen-turbo",
           "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"}


def init_llm(llm_config: dict):
    """注入配置（main.py启动时调用一次）"""
    _config.update(llm_config)


def get_client() -> OpenAI:
    """获取客户端（模块级单例）"""
    global _client
    if _client is None:
        base_url = os.getenv("OPENAI_BASE_URL", _config["base_url"])
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("未设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def chat_api(messages, model=None, temperature=0.7, max_tokens=512, retries=2,
             fallback="抱歉，我现在有点走神，您能再说一遍吗？"):
    """调用千问，失败时简单重试；最终失败返回fallback

    内部判别任务（评分/意图等）应传 fallback=""，
    让调用方走自己的规则兜底，避免把兜底话术误当成模型输出。
    """
    model_name = model or os.getenv("QWEN_CHAT_MODEL", _config["model"])
    last_err = None
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            logger.warning(f"调用千问失败（第{attempt + 1}次）: {e}")
            time.sleep(0.5 * (attempt + 1))
    logger.error(f"调用千问最终失败: {last_err}")
    return fallback
