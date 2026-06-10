"""
HAMD 隐式对话式评估器
=======================================================

与旧版（根目录 hamd_evaluator_v3.py）的核心差异：
- **隐式提问**：问题以"朋友闲聊"的方式生成，全程不出现
  评估/测试/题目/量表/打分等字眼，用户体验是一场自然对话
- 评分、清晰度、情绪分析在后台静默进行（单次结构化LLM调用）
- 三阶段门控（初筛→深入→全面，按累计得分自适应缩短）
- 追问回答累积评分；自杀风险实时监控；高风险强制进入深入阶段
"""

import json
import time
import os
from typing import Dict, List, Optional, Callable
import re


class HAMDEvaluator:
    """HAMD 隐式对话式评估器"""
    
    def __init__(self,
                 questions_file: str = "hamd_questions_v3.json",
                 output_dir: str = "./HAMD_Results/",
                 chat_fn: Optional[Callable[[List[Dict], float, int], str]] = None,
                 stage_thresholds: Optional[Dict[int, int]] = None,
                 max_follow_up: int = 2,
                 crisis_hotline: str = "12356"):
        """
        初始化HAMD评估器

        Args:
            questions_file: 问题配置文件路径
            output_dir: 结果输出目录
            chat_fn: 外部LLM调用函数（用于云端API，如千问）
            stage_thresholds: 阶段门控阈值 {1: 进入阶段2的累计分, 2: 进入阶段3的累计分}
            max_follow_up: 每题最多追问次数
            crisis_hotline: 危机干预热线号码（用于报告建议）
        """
        self.questions_file = questions_file
        self.output_dir = output_dir
        self.current_question_index = 0
        self.scores = {}
        self.evaluation_start_time = None
        self.evaluation_id = None
        self.crisis_hotline = crisis_hotline

        # === 三阶段评估配置 ===
        self.current_stage = 1  # 当前阶段：1=初筛, 2=深入, 3=全面
        self.stage_thresholds = stage_thresholds or {
            1: 5,   # 阶段1分数 >= 5 进入阶段2（可调整）
            2: 10   # 阶段2累计分数 >= 10 进入阶段3（可调整）
        }
        self.stage_completed = {1: False, 2: False, 3: False}
        self.stage_skipped = {2: False, 3: False}  # 记录哪些阶段被跳过（保留以便未来扩展）
        
        # === 上下文记忆 ===
        self.conversation_history = []  # 完整对话历史
        self.emotional_context = {
            "detected_emotion": "neutral",  # 检测到的情绪：positive, neutral, negative, critical
            "emotion_intensity": 0,  # 情绪强度：0-10
            "risk_level": "low"  # 风险等级：low, medium, high, critical
        }
        
        # === 追问控制 ===
        self.follow_up_count = 0  # 当前问题的追问次数
        self.max_follow_up = max_follow_up  # 每个问题最多追问次数
        self.last_answer_clarity = 1.0  # 上次回答的清晰度：0-1

        # === 当前题目的累积回答（首答 + 追问补充，评分时合并使用） ===
        self.current_answer_parts = []
        self.last_asked_question = ""  # 实际向用户提出的问题（LLM动态生成的版本）
        
        # === 高风险监控 ===
        self.high_risk_detected = False  # 是否检测到高风险
        self.suicide_risk_score = 0  # 自杀风险评分
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载问题配置
        self.questions = self._load_questions()
        self._organize_questions_by_stage()
        
        # 外部聊天函数（云端API）
        self.chat_fn = chat_fn
        
        print("[HAMD-V3] 评估器初始化完成（三阶段 + LLM动态提问）")
        
    def _load_questions(self) -> List[Dict]:
        """加载问题配置"""
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            print(f"[HAMD-V3] 加载了 {len(questions)} 个问题")
            return questions
        except FileNotFoundError:
            raise FileNotFoundError(f"问题配置文件 {self.questions_file} 不存在")
        except json.JSONDecodeError:
            raise ValueError(f"问题配置文件 {self.questions_file} 格式错误")
    
    def _organize_questions_by_stage(self):
        """按阶段组织问题"""
        self.stage_questions = {1: [], 2: [], 3: []}
        for q in self.questions:
            stage = q.get('stage', 1)
            self.stage_questions[stage].append(q)
        
        # 按优先级排序（critical > high > medium > low）
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        for stage in [1, 2, 3]:
            self.stage_questions[stage].sort(
                key=lambda q: priority_order.get(q.get('priority', 'low'), 99)
            )
        
        print(f"[HAMD-V3] 问题分布: 阶段1={len(self.stage_questions[1])}题, "
              f"阶段2={len(self.stage_questions[2])}题, 阶段3={len(self.stage_questions[3])}题")
    
    def _call_llm(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 256) -> str:
        """统一的LLM调用接口"""
        if self.chat_fn is not None:
            return self.chat_fn(messages, temperature, max_tokens)
        else:
            # 如果没有配置LLM，返回默认响应
            return "LLM未配置"

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """从LLM输出中健壮地提取JSON对象（容忍代码块围栏、前后缀文字）"""
        if not text:
            return None
        # 去掉markdown代码块标记
        text = re.sub(r'```(?:json)?', '', text).strip()
        # 提取第一个 {...} 块
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    
    # ==================== 情绪感知模块 ====================

    def _rule_based_emotion_analysis(self, text: str) -> Dict:
        """基于规则的情绪分析（备用方案）"""
        negative_words = ["悲伤", "难过", "痛苦", "绝望", "无助", "孤独", "害怕", "焦虑", "紧张"]
        critical_words = ["想死", "自杀", "活着没意思", "解脱", "不想活"]
        
        text_lower = text.lower()
        
        # 检测严重情绪
        if any(word in text_lower for word in critical_words):
            return {"emotion": "critical", "intensity": 9, "keywords": critical_words[:3]}
        
        # 检测消极情绪
        negative_count = sum(1 for word in negative_words if word in text_lower)
        if negative_count >= 3:
            return {"emotion": "negative", "intensity": 7, "keywords": negative_words[:3]}
        elif negative_count >= 1:
            return {"emotion": "negative", "intensity": 5, "keywords": negative_words[:2]}
        
        return {"emotion": "neutral", "intensity": 2, "keywords": []}
    
    def get_adaptive_tone(self) -> str:
        """根据情绪上下文生成适应性语气描述"""
        emotion = self.emotional_context["detected_emotion"]
        intensity = self.emotional_context["emotion_intensity"]
        
        if emotion == "critical" or intensity >= 8:
            return "用极度温和、充满关怀的语气，让用户感受到被理解和支持"
        elif emotion == "negative" or intensity >= 5:
            return "用温暖、共情的语气，表达理解和关心"
        elif emotion == "neutral":
            return "用自然、亲切的语气"
        else:
            return "用轻松、友好的语气"
    
    # ==================== LLM动态提问模块 ====================
    
    def generate_dynamic_question(self, question_obj: Dict, is_follow_up: bool = False,
                                  last_answer: Optional[str] = None) -> str:
        """
        使用LLM动态生成问题

        Args:
            question_obj: 问题对象（包含intent_description、question_variants等）
            is_follow_up: 是否为追问
            last_answer: 用户上一题的回答；提供时生成"一句共情 + 自然过渡提问"，
                         一次调用、一次TTS播报完成，避免单独的共情回应调用

        Returns:
            生成的自然问题文本
        """
        # 构建上下文
        context_summary = self._get_context_summary()
        adaptive_tone = self.get_adaptive_tone()

        # 隐式模式共用的"反问卷"硬约束
        implicit_rules = """绝对禁止（违反任何一条都算失败）：
- 禁止出现"评估、测试、测评、题目、量表、问卷、打分、第几个问题、下一个问题"等字眼
- 禁止医生问诊腔（如"请问您是否存在……症状"）
- 禁止一次问两件事；整段话只能有结尾一个问句
- 禁止说教、建议、安慰套话（如"要加油哦"）"""

        if is_follow_up:
            # 追问模式：顺着对方的话往深处聊一步
            prompt = f"""你是一位温暖细腻的倾听者，正在和朋友自然地聊天。
对方刚才说的有点笼统，你想顺着TA的话，再自然地往深处聊一步，
了解清楚TA的「{question_obj['title']}」（{question_obj['intent_description']}）。

可参考的聊法（不要照搬）：
{chr(10).join(question_obj.get('follow_up_hints', ['能再多说说吗？'])[:2])}

最近的对话：
{context_summary}

对方当前情绪：{self.emotional_context['detected_emotion']}（强度{self.emotional_context['emotion_intensity']}/10）

要求：
1. {adaptive_tone}
2. 像朋友间的自然接话，15-25字
3. 不要重复之前的问法
{implicit_rules}

只输出这句话本身："""
        elif last_answer:
            # 过渡模式：先接住对方刚才的话，再自然聊到新话题
            prompt = f"""你是一位温暖细腻的倾听者，正在和朋友自然地聊天。

对方刚才说："{last_answer[:60]}"

你想自然地把话题引到TA的「{question_obj['title']}」上
（想了解：{question_obj['intent_description']}），
但对话必须像朋友闲聊，绝不能让对方觉得在被提问或被调查。

可参考的聊法（不要照搬）：
{chr(10).join(question_obj.get('question_variants', [question_obj['question']])[:3])}

最近的对话：
{context_summary}

对方当前情绪：{self.emotional_context['detected_emotion']}（强度{self.emotional_context['emotion_intensity']}/10）

要求：
1. {adaptive_tone}
2. 先用一句简短的话接住对方刚才说的（不超过12字，是回应不是评价），再自然带出新话题
3. 总长度30-45字，口语化，适合语音说出来
{implicit_rules}

只输出这段话本身："""
        else:
            # 开场/独立提问模式
            prompt = f"""你是一位温暖细腻的倾听者，正在和朋友自然地聊天。
你想自然地聊到TA的「{question_obj['title']}」
（想了解：{question_obj['intent_description']}），
但对话必须像朋友闲聊，绝不能让对方觉得在被提问或被调查。

可参考的聊法（不要照搬）：
{chr(10).join(question_obj.get('question_variants', [question_obj['question']])[:3])}

最近的对话：
{context_summary}

对方当前情绪：{self.emotional_context['detected_emotion']}（强度{self.emotional_context['emotion_intensity']}/10）

要求：
1. {adaptive_tone}
2. 口语化，20-35字，适合语音说出来
3. 如果有之前的对话，要自然衔接，不要突兀转折
{implicit_rules}

只输出这句话本身："""
        
        try:
            messages = [
                {"role": "system", "content": "你是温暖细腻的倾听者，说话像朋友间的自然聊天。只输出那句话本身，不要任何多余内容、引号或解释。"},
                {"role": "user", "content": prompt}
            ]
            
            question_text = self._call_llm(messages, temperature=0.8, max_tokens=128)
            
            # 清洗生成的问题
            question_text = question_text.strip()
            # 移除可能的引号
            question_text = question_text.strip('"').strip("'")
            # 移除可能的前缀
            question_text = re.sub(r'^(问题[：:]|提问[：:]|我想问[：:])', '', question_text)
            
            # 如果生成失败或长度异常，使用备用问法（共情过渡模式允许更长）
            max_len = 120 if last_answer else 80
            if len(question_text) < 5 or len(question_text) > max_len:
                question_text = question_obj.get('question_variants', [question_obj['question']])[0]

            return question_text
            
        except Exception as e:
            print(f"[LLM生成问题失败] {e}")
            # 使用预设的变体问法
            variants = question_obj.get('question_variants', [question_obj['question']])
            return variants[0] if variants else question_obj['question']
    
    def _get_context_summary(self, max_turns: int = 3) -> str:
        """获取最近的对话上下文摘要"""
        if not self.conversation_history:
            return "（首次提问，无上下文）"
        
        recent_history = self.conversation_history[-max_turns:]
        summary_parts = []
        
        for i, turn in enumerate(recent_history, 1):
            q_title = turn.get('question_title', '问题')
            answer_summary = turn.get('answer', '')[:30] + '...' if len(turn.get('answer', '')) > 30 else turn.get('answer', '')
            summary_parts.append(f"{q_title}: {answer_summary}")
        
        return "\n".join(summary_parts) if summary_parts else "（无上下文）"
    
    # ==================== 智能追问模块 ====================

    def _rule_based_clarity_evaluation(self, text: str) -> float:
        """基于规则的清晰度评估（备用方案）"""
        text = text.strip()
        
        # 太短
        if len(text) < 3:
            return 0.2
        
        # 模糊词汇
        vague_words = ["可能", "好像", "也许", "不太清楚", "说不好", "不知道"]
        vague_count = sum(1 for word in vague_words if word in text)
        
        if vague_count >= 2:
            return 0.4
        elif vague_count == 1:
            return 0.6
        
        # 包含具体描述
        specific_words = ["每天", "经常", "总是", "从不", "严重", "轻微", "大约", "小时", "天"]
        specific_count = sum(1 for word in specific_words if word in text)
        
        if specific_count >= 2:
            return 0.9
        elif specific_count >= 1:
            return 0.7
        
        return 0.6
    
    def should_follow_up(self) -> bool:
        """判断是否需要追问"""
        # 已达到最大追问次数
        if self.follow_up_count >= self.max_follow_up:
            return False
        
        # 高风险问题总是要确认清楚
        current_q = self.get_current_question()
        if current_q and current_q.get('priority') == 'critical':
            if self.last_answer_clarity < 0.8:
                return True
        
        # 一般问题：清晰度低于0.5才追问
        if self.last_answer_clarity < 0.5:
            return True
        
        return False
    
    # ==================== 评分模块 ====================

    def analyze_answer_combined(self, user_answer: str, question_obj: Dict) -> Dict:
        """
        单次LLM调用完成：评分 + 清晰度 + 情绪分析（替代原先3次串行调用）

        Args:
            user_answer: 当前题目的完整回答（首答 + 追问补充，已合并）
            question_obj: 当前问题对象

        Returns:
            {"score": int, "clarity": float, "emotion": str,
             "intensity": int, "rationale": str}
        """
        scoring_text = "\n".join([f"{score}分: {desc}" for score, desc in question_obj['scoring'].items()])
        asked_question = self.last_asked_question or question_obj['question']

        prompt = f"""你是精神科量表评定员，依据汉密尔顿抑郁量表（HAMD）条目，对受试者的回答进行结构化分析。

【当前条目】{question_obj['title']}
【评估目的】{question_obj['intent_description']}
【实际提问】{asked_question}
【评分标准】
{scoring_text}

【受试者回答】（含追问后的补充，按时间顺序用 / 分隔）
"{user_answer}"

【近期对话摘要】（仅用于理解语境，不直接参与本条目评分）
{self._get_context_summary()}

请完成三项独立判断，只输出一个JSON对象，不要输出任何其他文字、不要markdown代码块：
{{"score": 0, "clarity": 1.0, "emotion": "neutral", "intensity": 0, "rationale": "一句话评分依据"}}

字段说明：
- score：0-4整数，严格对照上面的评分标准；只依据回答中的事实信息（频率、持续时间、对功能的影响），不要因情绪化用词而夸大评分；回答与本条目无关或完全无法判断时给0
- clarity：0-1小数，衡量回答是否足以支撑评分。1.0=信息充分明确；0.7=较清晰但缺少频率/程度等细节；0.4=模糊含混；0.2=答非所问或无效
- emotion：positive/neutral/negative/critical 之一；critical 仅在出现自伤、自杀、强烈绝望表达时使用
- intensity：0-10整数，情绪强度
- rationale：一句话依据，禁止提问、禁止给建议、禁止复述全部回答"""

        try:
            messages = [
                {"role": "system", "content": "你是专业的心理测量评定员，只输出一个合法的JSON对象，禁止输出任何其他内容。"},
                {"role": "user", "content": prompt}
            ]
            response = self._call_llm(messages, temperature=0.2, max_tokens=256)
            result = self._extract_json(response)
            if result is None:
                raise ValueError(f"无法解析分析结果: {response[:80]}")

            # 字段清洗与边界约束
            score = max(0, min(4, int(result.get("score", 0))))
            clarity = max(0.0, min(1.0, float(result.get("clarity", 0.7))))
            emotion = result.get("emotion", "neutral")
            if emotion not in ("positive", "neutral", "negative", "critical"):
                emotion = "neutral"
            intensity = max(0, min(10, int(result.get("intensity", 0))))
            rationale = str(result.get("rationale", "")).strip()
            # 防御：评定员输出不应包含提问
            if "?" in rationale or "？" in rationale:
                rationale = "（评分依据格式异常，已过滤）"

            return {"score": score, "clarity": clarity, "emotion": emotion,
                    "intensity": intensity, "rationale": rationale}

        except Exception as e:
            print(f"[综合分析失败，使用规则备用方案] {e}")
            emotion_result = self._rule_based_emotion_analysis(user_answer)
            return {
                "score": self._rule_based_scoring(user_answer),
                "clarity": self._rule_based_clarity_evaluation(user_answer),
                "emotion": emotion_result["emotion"],
                "intensity": emotion_result["intensity"],
                "rationale": "（LLM不可用，规则评分）"
            }

    def _update_emotional_context(self, emotion: str, intensity: int):
        """根据本轮分析结果更新情绪上下文与风险等级"""
        self.emotional_context["detected_emotion"] = emotion
        self.emotional_context["emotion_intensity"] = intensity

        if emotion == "critical" or intensity >= 8:
            self.emotional_context["risk_level"] = "critical"
        elif intensity >= 6:
            self.emotional_context["risk_level"] = "high"
        elif intensity >= 4:
            self.emotional_context["risk_level"] = "medium"
        else:
            self.emotional_context["risk_level"] = "low"

        # 情绪层面的危机表达也计入高风险（不依赖自杀条目是否已问到）
        if emotion == "critical":
            self.high_risk_detected = True

    def _rule_based_scoring(self, user_answer: str) -> int:
        """基于规则的简单评分（备用方案）"""
        answer_lower = user_answer.lower()
        
        negative_keywords = {
            "严重": 3, "很严重": 4, "非常": 3, "极度": 4,
            "无法": 3, "不能": 2, "困难": 2, "痛苦": 3,
            "绝望": 4, "想死": 4, "自杀": 4, "活着没意思": 4,
            "每天": 2, "经常": 2, "总是": 3, "一直": 3,
        }
        
        positive_keywords = {
            "正常": -2, "还好": -1, "偶尔": -1, "轻微": -1,
            "没有": -2, "不会": -2, "很好": -2,
        }
        
        score = 0
        
        for keyword, weight in negative_keywords.items():
            if keyword in answer_lower:
                score += weight
                
        for keyword, weight in positive_keywords.items():
            if keyword in answer_lower:
                score += weight
        
        # 标准化得分到0-4范围
        if score <= -2:
            return 0
        elif score <= 0:
            return 1
        elif score <= 2:
            return 2
        elif score <= 4:
            return 3
        else:
            return 4
    
    # ==================== 评估流程控制 ====================
    
    def start_evaluation(self) -> str:
        """开始评估，返回评估ID"""
        self.evaluation_start_time = time.time()
        self.evaluation_id = f"HAMD_V3_{int(self.evaluation_start_time)}"
        self.current_question_index = 0
        self.scores = {}

        # 重置阶段相关状态
        self.current_stage = 1
        self.stage_completed = {1: False, 2: False, 3: False}
        self.stage_skipped = {2: False, 3: False}
        
        # 重置上下文
        self.conversation_history = []
        self.emotional_context = {
            "detected_emotion": "neutral",
            "emotion_intensity": 0,
            "risk_level": "low"
        }
        
        # 重置追问和风险
        self.follow_up_count = 0
        self.high_risk_detected = False
        self.suicide_risk_score = 0
        self.current_answer_parts = []
        self.last_asked_question = ""

        print(f"[HAMD-V3] 评估开始: ID={self.evaluation_id}, 阶段=1")
        
        return self.evaluation_id
    
    def _total_score_so_far(self) -> int:
        """已完成条目的累计得分"""
        return sum(item["score"] for item in self.scores.values())

    def _should_enter_next_stage(self) -> bool:
        """
        阶段门控：判断是否需要进入下一阶段（自适应缩短评估时长）
        - 阶段1得分 >= stage_thresholds[1] 才进入阶段2
        - 累计得分 >= stage_thresholds[2] 才进入阶段3
        - 安全例外：只要检测到高风险/危机情绪，强制进入阶段2（含自杀条目），
          避免初筛分低但有危机表达的用户被漏掉
        """
        threshold = self.stage_thresholds.get(self.current_stage)
        if threshold is None:
            return False

        total = self._total_score_so_far()
        if total >= threshold:
            return True

        # 安全兜底：阶段1结束时若存在高风险信号，仍进入阶段2完成自杀风险评估
        if self.current_stage == 1 and (
            self.high_risk_detected
            or self.emotional_context["risk_level"] in ("high", "critical")
        ):
            print("[HAMD-V3] 初筛分数未达阈值，但检测到高风险信号，强制进入阶段2")
            return True

        return False

    def get_current_question(self) -> Optional[Dict]:
        """获取当前问题（三阶段逻辑：按累计得分门控是否深入）"""
        # 获取当前阶段的问题列表
        current_stage_questions = self.stage_questions.get(self.current_stage, [])

        # 如果当前阶段的题目都已完成
        if self.current_question_index >= len(current_stage_questions):
            # 阶段完成逻辑
            if not self.stage_completed[self.current_stage]:
                self.stage_completed[self.current_stage] = True

                if self.current_stage < 3:
                    if self._should_enter_next_stage():
                        self.current_stage += 1
                        self.current_question_index = 0
                        print(f"[HAMD-V3] 累计得分{self._total_score_so_far()}分，进入阶段{self.current_stage}")
                        next_stage_questions = self.stage_questions.get(self.current_stage, [])
                        if next_stage_questions:
                            return next_stage_questions[0]
                    else:
                        # 未达阈值：跳过后续阶段，提前结束评估
                        for s in range(self.current_stage + 1, 4):
                            self.stage_skipped[s] = True
                        print(f"[HAMD-V3] 累计得分{self._total_score_so_far()}分未达阈值，"
                              f"跳过阶段{self.current_stage + 1}及之后，评估提前结束")

            return None

        # 返回当前问题
        return current_stage_questions[self.current_question_index]
    
    def get_question_prompt(self, is_follow_up: bool = False,
                            last_answer: Optional[str] = None) -> str:
        """生成当前问题的提问内容（LLM动态生成）

        Args:
            is_follow_up: 是否为追问
            last_answer: 用户上一题的回答，提供时生成带共情过渡的提问
        """
        current_q = self.get_current_question()
        if not current_q:
            return ""

        # 使用LLM动态生成问题
        question_text = self.generate_dynamic_question(current_q, is_follow_up, last_answer)

        # 记录实际提问文本，供评分提示词对齐"问的什么"与"答的什么"
        self.last_asked_question = question_text

        return question_text
    
    def process_answer(self, user_answer: str) -> Dict:
        """
        处理用户回答（核心流程）
        
        Returns:
            {
                "question_id": int,
                "score": int,
                "analysis": str,
                "need_follow_up": bool,
                "is_complete": bool,
                "high_risk_alert": bool
            }
        """
        current_q = self.get_current_question()
        if not current_q:
            return {"error": "评估已完成"}

        question_id = current_q['index']

        # 1. 累积本题回答（首答 + 追问补充），评分基于完整信息而非最后一句
        self.current_answer_parts.append(user_answer.strip())
        combined_answer = " / ".join(self.current_answer_parts)

        # 2. 单次LLM调用完成评分 + 清晰度 + 情绪分析
        analysis_result = self.analyze_answer_combined(combined_answer, current_q)
        clarity = analysis_result["clarity"]
        emotion_analysis = {
            "emotion": analysis_result["emotion"],
            "intensity": analysis_result["intensity"]
        }
        self.last_answer_clarity = clarity
        self._update_emotional_context(analysis_result["emotion"], analysis_result["intensity"])
        print(f"[综合分析] 评分={analysis_result['score']}, 清晰度={clarity:.2f}, "
              f"情绪={analysis_result['emotion']}({analysis_result['intensity']}/10)")

        # 3. 判断是否需要追问（追问时暂不提交评分，不推进题目）
        need_follow_up = self.should_follow_up()

        if need_follow_up:
            self.follow_up_count += 1
            print(f"[追问] 第{self.follow_up_count}次追问")

            self.conversation_history.append({
                "question_title": current_q['title'],
                "answer": user_answer,
                "clarity": clarity,
                "emotion": emotion_analysis,
                "is_follow_up_response": True
            })

            return {
                "question_id": question_id,
                "need_follow_up": True,
                "follow_up_count": self.follow_up_count,
                "is_complete": False,
                "high_risk_alert": self.high_risk_detected
            }

        # 4. 提交评分
        score = analysis_result["score"]
        analysis = f"评分：{score}分，理由：{analysis_result['rationale']}"
        print(f"[评分] 题{question_id}: {score}分")

        # 自杀条目高风险监控
        if question_id == 3:
            self.suicide_risk_score = score
            if score >= 2:
                self.high_risk_detected = True
                self.emotional_context["risk_level"] = "critical"

        # 5. 记录评分
        self.scores[question_id] = {
            "score": score,
            "analysis": analysis,
            "scoring_criteria": current_q['scoring'][str(score)]
        }

        # 记录到对话历史
        self.conversation_history.append({
            "question_title": current_q['title'],
            "question_id": question_id,
            "answer": combined_answer,
            "score": score,
            "emotion": emotion_analysis,
            "clarity": clarity
        })

        # 6. 移动到下一题，重置本题状态
        self.current_question_index += 1
        self.follow_up_count = 0
        self.current_answer_parts = []

        # 7. 推进阶段状态并判断是否结束
        #    （get_current_question 内部执行阶段门控：达标进入下一阶段，否则跳过后续阶段）
        has_next_question = self.get_current_question() is not None
        is_complete = (not has_next_question) or self.is_evaluation_complete()

        return {
            "question_id": question_id,
            "score": score,
            "analysis": analysis,
            "need_follow_up": False,
            "is_complete": is_complete,
            "high_risk_alert": self.high_risk_detected
        }
    def is_evaluation_complete(self) -> bool:
        """检查评估是否完成（阶段3的所有题目完成即为完成）"""
        # 阶段3完成，评估完成
        if self.current_stage == 3:
            current_stage_questions = self.stage_questions.get(3, [])
            if self.current_question_index >= len(current_stage_questions):
                return True
        
        # 阶段2/3被跳过，且当前阶段完成
        if self.stage_skipped.get(2) or self.stage_skipped.get(3):
            current_stage_questions = self.stage_questions.get(self.current_stage, [])
            if self.current_question_index >= len(current_stage_questions):
                return True
        
        return False
    
    # ==================== 报告生成 ====================
    
    def generate_report(self) -> Dict:
        """生成评估报告"""
        if not self.scores:
            return {"error": "没有评估数据"}
        
        # 计算总分
        total_score = sum(item["score"] for item in self.scores.values())
        
        # 生成严重程度评估
        severity_level = self._get_severity_level(total_score)
        
        # 分项分析
        category_analysis = self._analyze_categories()
        
        # 生成建议
        recommendations = self._generate_recommendations(total_score, category_analysis)
        
        # 情绪分析汇总
        emotion_summary = self._summarize_emotions()
        
        # 是否因初筛/深入阶段分数未达阈值而提前结束
        early_terminated = self.stage_skipped.get(2) or self.stage_skipped.get(3)
        if early_terminated:
            severity_level["note"] = (
                "本次评估因前期阶段得分低于阈值而提前结束，"
                "总分基于已作答条目，结果仅供初筛参考。"
            )

        report = {
            "evaluation_id": self.evaluation_id,
            "version": "V3 (三阶段智能评估)",
            "total_score": total_score,
            "max_score": len(self.questions) * 4,
            "answered_questions": len(self.scores),
            "total_questions": len(self.questions),
            "early_terminated": bool(early_terminated),
            "severity_level": severity_level,
            "detailed_scores": self.scores,
            "category_analysis": category_analysis,
            "recommendations": recommendations,
            "structured_suggestions": self._generate_structured_suggestions(total_score),
            "conversation_turns": len(self.conversation_history),
            "emotion_analysis": emotion_summary,
            "risk_assessment": {
                "suicide_risk_score": self.suicide_risk_score,
                "overall_risk_level": self.emotional_context["risk_level"],
                "high_risk_detected": self.high_risk_detected
            },
            "stages_completed": {
                "stage_1": self.stage_completed[1],
                "stage_2": self.stage_completed[2],
                "stage_3": self.stage_completed[3]
            },
            "evaluation_time": {
                "start_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.evaluation_start_time)),
                "duration_minutes": round((time.time() - self.evaluation_start_time) / 60, 1)
            }
        }
        
        # 保存报告
        self._save_report(report)
        
        return report
    
    def _get_severity_level(self, total_score: int) -> Dict:
        """根据总分判断严重程度"""
        if total_score <= 7:
            level = "正常"
            description = "无明显抑郁症状"
            color = "green"
        elif total_score <= 17:
            level = "轻度抑郁"
            description = "有轻度抑郁症状，建议关注心理健康"
            color = "yellow"
        elif total_score <= 23:
            level = "中度抑郁"
            description = "有中度抑郁症状，建议寻求专业帮助"
            color = "orange"
        else:
            level = "重度抑郁"
            description = "有重度抑郁症状，强烈建议立即寻求专业医疗帮助"
            color = "red"
        
        return {
            "level": level,
            "description": description,
            "color": color,
            "score_range": f"{total_score}/{len(self.questions) * 4}"
        }
    
    def _analyze_categories(self) -> Dict:
        """分项分析"""
        categories = {
            "核心症状": [1, 2, 6],
            "认知症状": [10, 15],
            "生理症状": [4, 5, 7, 8, 13],
            "心理症状": [3, 9, 11, 12, 14],
            "其他症状": [16]
        }
        
        analysis = {}
        for category, question_ids in categories.items():
            # 只统计实际作答的条目，避免提前结束时未答题被当作0分稀释比例
            answered_ids = [qid for qid in question_ids if qid in self.scores]
            if not answered_ids:
                continue
            scores = [self.scores[qid]["score"] for qid in answered_ids]
            total = sum(scores)
            max_possible = len(scores) * 4
            percentage = (total / max_possible) * 100 if max_possible > 0 else 0

            analysis[category] = {
                "score": total,
                "max_score": max_possible,
                "percentage": round(percentage, 1),
                "severity": "重度" if percentage >= 75 else "中度" if percentage >= 50 else "轻度" if percentage >= 25 else "正常"
            }
        
        return analysis
    
    def _summarize_emotions(self) -> Dict:
        """汇总评估过程中的情绪变化"""
        if not self.conversation_history:
            return {}
        
        emotions = [turn.get('emotion', {}) for turn in self.conversation_history if turn.get('emotion')]
        
        if not emotions:
            return {}
        
        avg_intensity = sum(e.get('intensity', 0) for e in emotions) / len(emotions) if emotions else 0
        
        emotion_counts = {}
        for e in emotions:
            emotion_type = e.get('emotion', 'neutral')
            emotion_counts[emotion_type] = emotion_counts.get(emotion_type, 0) + 1
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
        
        return {
            "average_intensity": round(avg_intensity, 2),
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_counts
        }
    
    def _generate_recommendations(self, total_score: int, category_analysis: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if total_score <= 7:
            recommendations.append("您的心理状态良好，建议保持健康的生活方式。")
            recommendations.append("继续维持规律作息、适量运动和良好的社交关系。")
        elif total_score <= 17:
            recommendations.append("建议关注自己的心理健康状态。")
            recommendations.append("可以尝试放松技巧，如深呼吸、冥想或适度运动。")
            recommendations.append("如果症状持续，建议咨询心理健康专业人士。")
        elif total_score <= 23:
            recommendations.append("建议尽快寻求专业心理咨询或治疗。")
            recommendations.append("可以考虑心理治疗，如认知行为疗法。")
            recommendations.append("保持与家人朋友的联系，寻求社会支持。")
        else:
            recommendations.append("强烈建议立即寻求专业医疗帮助。")
            recommendations.append("可能需要药物治疗结合心理治疗。")
            recommendations.append("请不要独自承担，及时联系医生或心理健康专家。")
        
        # 检查自杀风险
        if self.scores.get(3, {}).get("score", 0) >= 2:
            recommendations.insert(
                0, f"⚠️ 重要提醒：如有自杀念头，请立即拨打全国心理援助热线：{self.crisis_hotline}")

        return recommendations

    def _generate_structured_suggestions(self, total_score: int) -> List[Dict]:
        """生成结构化建议卡片（title/description/frequency），便于前端按卡片渲染"""
        if total_score <= 7:
            suggestions = [
                {"title": "保持节律", "description": "规律的作息和饮食是情绪稳定的基础", "frequency": "每天"},
                {"title": "适度运动", "description": "散步、慢跑或任何让身体动起来的活动", "frequency": "每周3次，每次30分钟"},
                {"title": "社交联结", "description": "和朋友家人保持联系，分享日常", "frequency": "经常"},
            ]
        elif total_score <= 17:
            suggestions = [
                {"title": "正念冥想", "description": "专注呼吸，帮助缓解焦虑和低落情绪", "frequency": "每天10分钟"},
                {"title": "情绪日记", "description": "记录情绪变化，了解自己的情绪模式", "frequency": "每天睡前"},
                {"title": "适度运动", "description": "运动是天然的情绪调节剂", "frequency": "每周3次，每次30分钟"},
            ]
        elif total_score <= 23:
            suggestions = [
                {"title": "寻求专业帮助", "description": "建议预约心理咨询师，认知行为疗法对中度抑郁有较好效果", "frequency": "尽快"},
                {"title": "社会支持", "description": "把感受告诉信任的家人朋友，不要独自承受", "frequency": "经常"},
                {"title": "温和活动", "description": "从小事开始，散步、晒太阳，不强求状态", "frequency": "每天"},
            ]
        else:
            suggestions = [
                {"title": "及时就医", "description": "建议尽快前往医院心理科或精神科就诊", "frequency": "立即"},
                {"title": "危机支持", "description": f"感到难以承受时，拨打心理援助热线{self.crisis_hotline}", "frequency": "24小时可用"},
                {"title": "不要独处", "description": "请让家人或朋友陪伴，告诉他们你的状况", "frequency": "现阶段"},
            ]
        return suggestions
    
    def _save_report(self, report: Dict):
        """保存评估报告"""
        filename = f"{self.output_dir}/HAMD_Report_{self.evaluation_id}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"[HAMD-V3] 评估报告已保存至: {filename}")
        except Exception as e:
            print(f"[保存报告失败] {e}")
    

