"""对话编排：普通陪伴 / 隐式评测 / 危机协议

隐式评测设计：
- 用户说"做个评测/聊聊我最近的状态"后，系统不进入任何可感知的"评估模式"，
  而是开启一场自然对话；每轮回应由评估器以"朋友接话"的方式生成
- 评分、清晰度、情绪分析、阶段门控全部在后台静默进行
- 全程不出现"评估/题目/打分/第几题"等字眼；报告静默保存
- 危机检测在所有模式下最高优先级
"""

import logging
import re

from .llm_client import chat_api

logger = logging.getLogger("chatmind")


class ChatMemory:
    """普通对话的轻量记忆"""

    def __init__(self, max_length=1024):
        self.history = []
        self.max_length = max_length

    def add(self, user_input, reply):
        self.history.append(f"用户: {user_input}")
        self.history.append(f"助手: {reply}")

    def context(self):
        text = "\n".join(self.history)
        return text[-self.max_length:] if len(text) > self.max_length else text


class DialogueManager:
    """对话状态机：idle（陪伴聊天） / assessing（隐式评测中）"""

    # 隐式评测触发（显式意愿仍由用户发起——这是知情的前提）
    TRIGGER_KEYWORDS = [
        "心理评估", "抑郁评估", "抑郁测试", "心理测试", "开始评估", "hamd", "抑郁量表",
        "做个评测", "做个测评", "帮我评测", "测一下", "聊聊我最近的状态", "看看我最近怎么样",
    ]

    # 评测中自然退出的说法
    EXIT_PHRASES = [
        "不聊了", "先这样", "先到这", "到此为止", "换个话题", "聊点别的",
        "不想聊这个", "停止", "不测了", "退出", "结束吧",
    ]

    REPEAT_PHRASES = ["再说一遍", "没听清", "重复一下", "你说什么"]

    SHORT_VALID_ANSWERS = {"没有", "有", "是", "不是", "是的", "不会", "会",
                           "还好", "没", "嗯", "对", "不", "偶尔", "经常"}

    def __init__(self, config, evaluator, speak_fn):
        """
        Args:
            config: 全局配置dict
            evaluator: HAMDEvaluator实例（chat_fn已注入）
            speak_fn: fn(text) 播报回调
        """
        self.config = config
        self.evaluator = evaluator
        self.speak = speak_fn
        self.memory = ChatMemory()

        self.mode = "idle"
        self.waiting_for_answer = False  # 供采集层动态调整停顿阈值
        self.processing = False          # 供采集层在处理期间丢弃语音
        self._last_question = ""
        self._crisis_announced = False

        safety = config["safety"]
        self.crisis_keywords = safety["crisis_keywords"]
        self.crisis_hotline = safety["crisis_hotline"]
        self.assistant_name = config["persona"]["assistant_name"]
        self.min_answer_length = config["audio"]["min_answer_length"]

        self.chat_system_prompt = (
            f"你是{self.assistant_name}，一位温暖、专业的心理健康陪伴助手。\n\n"
            "你的原则：\n"
            "1. 以倾听和共情为先：先接住用户的情绪，再自然回应；不评判、不说教、不灌鸡汤\n"
            "2. 回应简洁口语化，适合语音播报，一般不超过60字，不用列表和表情符号\n"
            "3. 不做医学诊断、不推荐药物、不替用户做重大人生决定\n"
            "4. 用户聊到持续的情绪困扰时，可以温和地提一句：愿意的话，我们可以聊聊你最近的整体状态\n"
            f"5. 用户提到伤害自己的念头时认真对待：表达关心，告知心理援助热线{self.crisis_hotline}，"
            "鼓励联系信任的人，不要转移话题\n"
            "6. 像一位可靠的朋友：自然、真诚、有温度，但保持专业边界"
        )

    # ==================== 主入口 ====================

    def process_text(self, text: str):
        """处理一句识别出的用户话语（采集线程回调）"""
        text = text.strip()
        if not text:
            return
        self.processing = True
        try:
            # 1. 危机检测：所有模式下最高优先级
            if self._check_crisis(text):
                logger.warning(f"检测到危机表达: {text}")
                self._respond_crisis(text)
                return

            # 2. 模式分发
            if self.mode == "assessing":
                self._assessment_turn(text)
            elif self._check_trigger(text):
                self._start_assessment(text)
            else:
                self._normal_chat(text)
        finally:
            self.processing = False

    # ==================== 危机协议 ====================

    def _check_crisis(self, text: str) -> bool:
        return any(k in text for k in self.crisis_keywords)

    def _respond_crisis(self, text: str):
        response = (
            "听到你这样说，我很担心你。你现在的感受很重要，你并不孤单。"
            f"如果此刻很难受，可以拨打全国心理援助热线{self.crisis_hotline}，"
            "那里全天有人愿意听你说。也可以联系你信任的家人朋友，让他们陪陪你。"
            "我们可以继续聊，我在这里。"
        )
        self._crisis_announced = True
        if self.mode == "assessing":
            # 评测中触发危机：标记最高风险，回应后继续以对话方式陪伴
            self.evaluator._update_emotional_context("critical", 9)
        self.memory.add(text, response)
        self.speak(response)

    # ==================== 隐式评测 ====================

    def _check_trigger(self, text: str) -> bool:
        normalized = (text.lower()
                      .replace("心里", "心理").replace("抑于", "抑郁").replace("抑以", "抑郁")
                      .replace("评古", "评估").replace("测是", "测试"))
        return any(k in normalized for k in self.TRIGGER_KEYWORDS)

    def _start_assessment(self, trigger_text: str):
        """开启隐式评测：透明告知一句 + 自然开场，之后就是普通聊天的体验"""
        self.mode = "assessing"
        self._crisis_announced = False
        self.evaluator.start_evaluation()
        logger.info("隐式评测开始")

        # 透明告知（知情前提，但不搞仪式感）：聊天形式 + 随时可停 + 不是诊断
        intro = (
            "好呀。那我们就像平时一样随便聊聊，你想到什么就说什么，"
            "想停的时候直接说就行。"
        )
        first_question = self.evaluator.get_question_prompt()
        self._last_question = first_question
        self.waiting_for_answer = True
        self.speak(f"{intro}{first_question}")

    def _assessment_turn(self, text: str):
        """评测中的一轮对话"""
        # 太短的无效片段直接忽略（但保留量表有效短回答）
        chinese = "".join(re.findall(r"[一-龥]", text))
        if (len(chinese) < self.min_answer_length
                and len(text) < self.min_answer_length * 2
                and chinese not in self.SHORT_VALID_ANSWERS):
            logger.debug(f"回答过短，忽略: {text}")
            return

        # 自然退出
        if any(p in text for p in self.EXIT_PHRASES):
            self._end_assessment(completed=False)
            return

        # 要求重复
        if any(p in text for p in self.REPEAT_PHRASES) and self._last_question:
            self.speak(self._last_question)
            return

        # 交给评估器静默分析（评分/清晰度/情绪一次完成）
        self.waiting_for_answer = False
        result = self.evaluator.process_answer(text)

        if "error" in result:
            self._end_assessment(completed=True)
            return

        # 高风险即时干预（每次评测只播一次）
        if result.get("high_risk_alert") and not self._crisis_announced:
            self._crisis_announced = True
            self.speak(
                "谢谢你愿意把这些告诉我，我很担心你。如果此刻很难受，"
                f"可以拨打全国心理援助热线{self.crisis_hotline}，那里全天有人。"
                "我们慢慢聊，我陪着你。"
            )

        if result.get("is_complete"):
            self._end_assessment(completed=True)
            return

        # 继续聊：追问 或 自然过渡到下一个话题（共情+提问一次生成）
        if result.get("need_follow_up"):
            question = self.evaluator.get_question_prompt(is_follow_up=True)
        else:
            question = self.evaluator.get_question_prompt(last_answer=text)
        self._last_question = question
        self.waiting_for_answer = True
        self.speak(question)

    def _end_assessment(self, completed: bool):
        """结束评测：报告静默保存，收尾话术保持对话感"""
        self.mode = "idle"
        self.waiting_for_answer = False

        report = None
        if self.evaluator.scores:
            report = self.evaluator.generate_report()  # 静默保存JSON
            logger.info(
                f"评测结束 completed={completed} "
                f"总分={report.get('total_score')} "
                f"严重程度={report.get('severity_level', {}).get('level')} "
                f"风险={report.get('risk_assessment', {}).get('overall_risk_level')}"
            )

        if not completed:
            self.speak("好的，那这个话题我们先放一放。谢谢你愿意跟我聊这些，想继续的时候随时叫我。")
            return

        closing = self._generate_closing(report) if report else \
            "好的，今天先聊到这儿。有什么想说的，随时来找我。"
        self.speak(closing)

    def _generate_closing(self, report: dict) -> str:
        """生成自然的收尾反馈：不报分数、不提'评估完成'，像朋友聊完的总结"""
        level = report.get("severity_level", {}).get("level", "正常")
        risk = report.get("risk_assessment", {}).get("overall_risk_level", "low")
        emotion = report.get("emotion_analysis", {}).get("dominant_emotion", "neutral")

        prompt = f"""你刚陪一位朋友聊完TA最近的生活状态（心情、睡眠、精力、兴趣等）。
现在要说一段自然的收尾的话。

你对TA整体状态的把握（仅供你组织语言，绝不能向TA透露这些标签）：
- 整体状态：{level}
- 需要关注程度：{risk}
- 聊天中的主要情绪：{emotion}

要求：
1. 80-120字，口语化，像朋友聊完天的真诚收尾
2. 先谢谢TA愿意聊这些
3. 用观察性的语言温和反馈（如"听下来你最近确实挺累的"），不要下结论、不要贴标签
4. 整体状态是中度或重度时，自然地建议"找专业的心理咨询师或医生聊聊会很有帮助"；
   正常或轻度时给予肯定和日常建议（休息、运动、和朋友联系）
5. 绝对禁止出现：评估、测试、量表、分数、报告、抑郁症等字眼
6. 最后可以说一句"之后想聊随时找我"

只输出这段话："""
        result = chat_api(
            [{"role": "system", "content": "你是温暖真诚的倾听者，说话自然口语化。"},
             {"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=256, fallback="")
        result = (result or "").strip().replace("\n", "")
        if len(result) < 30:
            result = ("谢谢你愿意跟我聊这么多，听下来你最近确实承受了不少。"
                      "记得好好休息，多和身边的人聊聊。之后想说话，随时来找我。")
        return result

    # ==================== 普通陪伴对话 ====================

    def _normal_chat(self, text: str):
        context = self.memory.context()
        messages = [
            {"role": "system", "content": self.chat_system_prompt},
            {"role": "user", "content": f"{context}\n用户: {text}" if context else text},
        ]
        reply = chat_api(messages, max_tokens=512)
        self.memory.add(text, reply)
        self.speak(reply)
