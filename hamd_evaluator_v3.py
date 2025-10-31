"""
哈密尔顿抑郁量表（HAMD）评估系统 - Phase 3 完整版
=======================================================

功能特性：
- 三阶段评估（初筛5题 → 深入6题 → 全面5题）
- LLM动态生成问题（灵活自然的提问方式）
- 智能追问机制（根据回答模糊程度决定是否追问）
- 情绪感知与风格调整（根据用户情绪动态调整提问语气）
- 高风险问题优先（自杀风险实时评估）
- 多轮对话上下文记忆（记住之前的对话内容）
"""

import json
import time
import os
from typing import Dict, List, Tuple, Optional, Callable
import re


class HAMDEvaluatorV3:
    """哈密尔顿抑郁量表评估器 - Phase 3 完整版"""
    
    def __init__(self, 
                 questions_file: str = "hamd_questions_v3.json", 
                 model_name: str = None, 
                 output_dir: str = "./HAMD_Results/",
                 chat_fn: Optional[Callable[[List[Dict], float, int], str]] = None):
        """
        初始化HAMD评估器
        
        Args:
            questions_file: 问题配置文件路径
            model_name: 大语言模型路径（可选）
            output_dir: 结果输出目录
            chat_fn: 外部LLM调用函数（用于云端API，如千问）
        """
        self.questions_file = questions_file
        self.output_dir = output_dir
        self.current_question_index = 0
        self.scores = {}
        self.detailed_answers = {}
        self.evaluation_start_time = None
        self.evaluation_id = None
        
        # === 三阶段评估配置 ===
        self.current_stage = 1  # 当前阶段：1=初筛, 2=深入, 3=全面
        self.stage_thresholds = {
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
        self.max_follow_up = 2  # 每个问题最多追问次数
        self.last_answer_clarity = 1.0  # 上次回答的清晰度：0-1
        
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
    
    # ==================== 情绪感知模块 ====================
    
    def analyze_user_emotion(self, user_answer: str) -> Dict:
        """
        分析用户回答中的情绪
        返回: {emotion: str, intensity: float, keywords: List[str]}
        """
        prompt = f"""你是情绪分析专家。分析以下用户回答中的情绪状态。

用户回答："{user_answer}"

请分析：
1. 情绪类型：positive（积极）/ neutral（中性）/ negative（消极）/ critical（严重消极）
2. 情绪强度：0-10分（0=无情绪，10=极端强烈）
3. 关键词：列出体现情绪的3个关键词

只返回JSON格式：
{{"emotion": "negative", "intensity": 7, "keywords": ["悲伤", "绝望", "痛苦"]}}
"""
        
        try:
            messages = [
                {"role": "system", "content": "你是专业的情绪分析AI，只返回JSON格式结果。"},
                {"role": "user", "content": prompt}
            ]
            response = self._call_llm(messages, temperature=0.3, max_tokens=128)
            
            # 解析JSON
            result = json.loads(response)
            
            # 更新情绪上下文
            self.emotional_context["detected_emotion"] = result.get("emotion", "neutral")
            self.emotional_context["emotion_intensity"] = result.get("intensity", 0)
            
            # 更新风险等级
            if result.get("emotion") == "critical" or result.get("intensity", 0) >= 8:
                self.emotional_context["risk_level"] = "critical"
            elif result.get("intensity", 0) >= 6:
                self.emotional_context["risk_level"] = "high"
            elif result.get("intensity", 0) >= 4:
                self.emotional_context["risk_level"] = "medium"
            else:
                self.emotional_context["risk_level"] = "low"
            
            return result
            
        except Exception as e:
            print(f"[情绪分析失败] {e}")
            # 简单的规则fallback
            return self._rule_based_emotion_analysis(user_answer)
    
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
    
    def generate_dynamic_question(self, question_obj: Dict, is_follow_up: bool = False) -> str:
        """
        使用LLM动态生成问题
        
        Args:
            question_obj: 问题对象（包含intent_description、question_variants等）
            is_follow_up: 是否为追问
        
        Returns:
            生成的自然问题文本
        """
        # 构建上下文
        context_summary = self._get_context_summary()
        adaptive_tone = self.get_adaptive_tone()
        
        if is_follow_up:
            # 追问模式
            prompt = f"""你是心理评估助手，正在进行心理评估的第{self.current_stage}阶段。

当前要评估的维度：{question_obj['title']}
评估目的：{question_obj['intent_description']}

用户刚才的回答不够清楚，需要追问以获取更准确的信息。

参考追问示例：
{chr(10).join(question_obj.get('follow_up_hints', ['能再详细说说吗？'])[:2])}

对话上下文：
{context_summary}

当前情绪状态：{self.emotional_context['detected_emotion']}（强度{self.emotional_context['emotion_intensity']}/10）

要求：
1. {adaptive_tone}
2. 问题要简洁，15-25字
3. 不要重复之前的问法
4. 针对性地追问模糊的部分
5. 只生成问题本身，不要其他内容

请生成一个自然的追问："""
        else:
            # 首次提问模式
            prompt = f"""你是心理评估助手，正在进行抑郁症评估的第{self.current_stage}阶段。

当前要评估的维度：{question_obj['title']}
评估目的：{question_obj['intent_description']}

参考问法示例（不要照搬）：
{chr(10).join(question_obj.get('question_variants', [question_obj['question']])[:3])}

对话上下文：
{context_summary}

当前情绪状态：{self.emotional_context['detected_emotion']}（强度{self.emotional_context['emotion_intensity']}/10）

要求：
1. {adaptive_tone}
2. 用日常对话的方式提问，不要生硬
3. 问题简洁，20-30字
4. 不要提及"评估"、"量表"等专业术语
5. 结合之前的对话自然过渡
6. 只生成问题本身，不要前缀和解释

请生成一个自然的提问："""
        
        try:
            messages = [
                {"role": "system", "content": "你是专业的心理评估助手，擅长用自然亲切的方式提问。只输出问题本身，不要任何多余内容。"},
                {"role": "user", "content": prompt}
            ]
            
            question_text = self._call_llm(messages, temperature=0.8, max_tokens=128)
            
            # 清洗生成的问题
            question_text = question_text.strip()
            # 移除可能的引号
            question_text = question_text.strip('"').strip("'")
            # 移除可能的前缀
            question_text = re.sub(r'^(问题[：:]|提问[：:]|我想问[：:])', '', question_text)
            
            # 如果生成失败或太短，使用备用问法
            if len(question_text) < 5 or len(question_text) > 80:
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
    
    def evaluate_answer_clarity(self, user_answer: str, question_obj: Dict) -> float:
        """
        评估用户回答的清晰度
        返回: 0-1的分数（1=非常清晰，0=完全不清晰）
        """
        prompt = f"""你是回答质量评估专家。评估用户回答是否清晰、完整。

问题意图：{question_obj['intent_description']}
用户回答："{user_answer}"

评估标准：
- 1.0分：回答清晰、具体，包含足够的细节信息
- 0.7分：回答较清晰，但细节不够
- 0.5分：回答模糊，信息不足
- 0.3分：回答含糊不清或答非所问
- 0.0分：完全无关或无效回答

只返回一个0-1之间的小数，例如：0.8
"""
        
        try:
            messages = [
                {"role": "system", "content": "你是专业的评估专家，只返回一个0-1之间的数字。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self._call_llm(messages, temperature=0.3, max_tokens=32)
            
            # 提取数字
            match = re.search(r'0\.\d+|[01]', response)
            if match:
                clarity_score = float(match.group())
                self.last_answer_clarity = clarity_score
                return clarity_score
            else:
                return 0.7  # 默认中等清晰度
                
        except Exception as e:
            print(f"[清晰度评估失败] {e}")
            return self._rule_based_clarity_evaluation(user_answer)
    
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
    
    def analyze_answer_with_llm(self, user_answer: str, question_obj: Dict) -> Tuple[int, str]:
        """使用LLM分析用户回答并评分"""
        # 构建评分参考文本
        scoring_text = "\n".join([f"{score}分: {desc}" for score, desc in question_obj['scoring'].items()])
        
        prompt = f"""你是专业的心理测量评分员，只负责评分，不与用户对话。

题目：{question_obj['question']}
评估目的：{question_obj['intent_description']}

评分标准：
{scoring_text}

用户回答：
"{user_answer}"

对话上下文（用于理解回答）：
{self._get_context_summary()}

严格要求：
1) 只输出一行，格式必须是：评分：X分，理由：[简要分析]
2) 绝对禁止提问、追问、反问
3) 绝对禁止给建议、安慰、鼓励
4) 绝对禁止复述用户回答
5) 如果用户回答不清楚或无关，给0分
6) 根据评分标准严格评分

示例输出：评分：2分，理由：用户提到经常感到悲伤且失去兴趣，符合中度抑郁表现。

现在请严格按照格式输出评分："""

        try:
            messages = [
                {"role": "system", "content": "你是专业的心理评估评分机器人。你只能输出格式：评分：X分，理由：[分析]。严禁输出任何其他内容。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self._call_llm(messages, temperature=0.3, max_tokens=256)
            
            # 清洗响应
            response = response.strip()
            lines = response.split('\n')
            eval_line = ""
            for line in lines:
                if "评分" in line and ("：" in line or ":" in line):
                    eval_line = line.strip()
                    break
            
            if not eval_line:
                eval_line = response
            
            # 强力过滤：如果包含问号，直接丢弃
            if "?" in eval_line or "？" in eval_line:
                print(f"[警告] LLM生成了提问内容，已过滤")
                return 0, "评分：0分，理由：LLM输出格式错误"
            
            # 解析评分
            score_match = re.search(r'评分[：:]\s*(\d+)', eval_line)
            if score_match:
                score = int(score_match.group(1))
                score = max(0, min(4, score))
            else:
                score_match = re.search(r'(\d+)分', eval_line)
                if score_match:
                    score = int(score_match.group(1))
                    score = max(0, min(4, score))
                else:
                    score = 0
            
            # 更新自杀风险评分
            if question_obj.get('index') == 3:  # 自杀意图问题
                self.suicide_risk_score = score
                if score >= 2:
                    self.high_risk_detected = True
                    self.emotional_context["risk_level"] = "critical"
            
            return score, eval_line
            
        except Exception as e:
            print(f"[LLM评分失败] {e}")
            return self._rule_based_scoring(user_answer), f"规则评分: {str(e)}"
    
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
        self.detailed_answers = {}
        
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
        
        print(f"[HAMD-V3] 评估开始: ID={self.evaluation_id}, 阶段=1")
        
        return self.evaluation_id
    
    def get_current_question(self) -> Optional[Dict]:
        """获取当前问题（支持三阶段逻辑，自动转换）"""
        # 获取当前阶段的问题列表
        current_stage_questions = self.stage_questions.get(self.current_stage, [])
        
        # 如果当前阶段的题目都已完成
        if self.current_question_index >= len(current_stage_questions):
            # 阶段完成逻辑
            if not self.stage_completed[self.current_stage]:
                self.stage_completed[self.current_stage] = True
                
                # 自动进入下一阶段（不再询问）
                if self.current_stage < 3:
                    self.current_stage += 1
                    self.current_question_index = 0
                    print(f"[HAMD-V3] 自动进入阶段{self.current_stage}")
                    # 获取下一阶段的第一个问题
                    next_stage_questions = self.stage_questions.get(self.current_stage, [])
                    if next_stage_questions:
                        return next_stage_questions[0]
            
            return None
        
        # 返回当前问题
        return current_stage_questions[self.current_question_index]
    
    def get_question_prompt(self, is_follow_up: bool = False) -> str:
        """生成当前问题的提问内容（LLM动态生成）"""
        current_q = self.get_current_question()
        if not current_q:
            return ""
        
        # 使用LLM动态生成问题
        question_text = self.generate_dynamic_question(current_q, is_follow_up)
        
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
                "stage_transition_needed": bool,
                "is_complete": bool,
                "high_risk_alert": bool
            }
        """
        current_q = self.get_current_question()
        if not current_q:
            return {"error": "评估已完成"}
        
        question_id = current_q['index']
        
        # 1. 情绪分析
        emotion_analysis = self.analyze_user_emotion(user_answer)
        print(f"[情绪分析] {emotion_analysis}")
        
        # 2. 评估清晰度
        clarity = self.evaluate_answer_clarity(user_answer, current_q)
        print(f"[清晰度] {clarity:.2f}")
        
        # 3. 判断是否需要追问
        need_follow_up = self.should_follow_up()
        
        if need_follow_up:
            # 需要追问，不评分，不推进
            self.follow_up_count += 1
            print(f"[追问] 第{self.follow_up_count}次追问")
            
            # 记录到对话历史但不评分
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
                "stage_transition_needed": False,
                "is_complete": False,
                "high_risk_alert": self.high_risk_detected
            }
        
        # 4. 评分
        score, analysis = self.analyze_answer_with_llm(user_answer, current_q)
        print(f"[评分] 题{question_id}: {score}分")
        
        # 5. 记录答案和评分
        self.detailed_answers[question_id] = {
            "question": current_q['question'],
            "title": current_q['title'],
            "answer": user_answer,
            "timestamp": time.time(),
            "emotion": emotion_analysis,
            "clarity": clarity
        }
        
        self.scores[question_id] = {
            "score": score,
            "analysis": analysis,
            "scoring_criteria": current_q['scoring'][str(score)]
        }
        
        # 记录到对话历史
        self.conversation_history.append({
            "question_title": current_q['title'],
            "question_id": question_id,
            "answer": user_answer,
            "score": score,
            "emotion": emotion_analysis,
            "clarity": clarity
        })
        
        # 6. 移动到下一题
        self.current_question_index += 1
        self.follow_up_count = 0  # 重置追问计数
        
        # 7. 判断评估是否完全结束
        is_complete = self.is_evaluation_complete()
        
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
        
        report = {
            "evaluation_id": self.evaluation_id,
            "version": "V3 (三阶段智能评估)",
            "total_score": total_score,
            "max_score": len(self.questions) * 4,
            "severity_level": severity_level,
            "detailed_scores": self.scores,
            "category_analysis": category_analysis,
            "recommendations": recommendations,
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
            scores = [self.scores.get(qid, {}).get("score", 0) for qid in question_ids]
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
            recommendations.insert(0, "⚠️ 重要提醒：如有自杀念头，请立即拨打心理危机干预热线：400-161-9995")
        
        return recommendations
    
    def _save_report(self, report: Dict):
        """保存评估报告"""
        filename = f"{self.output_dir}/HAMD_Report_{self.evaluation_id}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"[HAMD-V3] 评估报告已保存至: {filename}")
        except Exception as e:
            print(f"[保存报告失败] {e}")
    
    # ==================== 辅助功能 ====================
    
    def previous_question(self) -> bool:
        """回到上一题"""
        if self.current_question_index <= 0:
            return False
        
        current_stage_questions = self.stage_questions.get(self.current_stage, [])
        if self.current_question_index > 0:
            prev_q = current_stage_questions[self.current_question_index - 1]
            prev_id = prev_q['index']
            
            # 撤销记录
            if prev_id in self.scores:
                del self.scores[prev_id]
            if prev_id in self.detailed_answers:
                del self.detailed_answers[prev_id]
            
            self.current_question_index -= 1
            self.follow_up_count = 0
            return True
        
        return False
    
    def next_question(self) -> bool:
        """跳到下一题"""
        current_stage_questions = self.stage_questions.get(self.current_stage, [])
        if self.current_question_index >= len(current_stage_questions):
            return False
        
        self.current_question_index += 1
        self.follow_up_count = 0
        return self.current_question_index <= len(current_stage_questions)
    
    def reset_evaluation(self):
        """重置评估"""
        self.current_question_index = 0
        self.scores = {}
        self.detailed_answers = {}
        self.evaluation_start_time = None
        self.evaluation_id = None
        
        self.current_stage = 1
        self.stage_completed = {1: False, 2: False, 3: False}
        self.stage_skipped = {2: False, 3: False}
        
        self.conversation_history = []
        self.emotional_context = {
            "detected_emotion": "neutral",
            "emotion_intensity": 0,
            "risk_level": "low"
        }
        
        self.follow_up_count = 0
        self.high_risk_detected = False
        self.suicide_risk_score = 0
        
        print("[HAMD-V3] 评估已重置")
    
    def get_progress(self) -> Dict:
        """获取评估进度"""
        current_stage_questions = self.stage_questions.get(self.current_stage, [])
        total_questions = sum(len(self.stage_questions[s]) for s in [1, 2, 3])
        completed_questions = sum(1 for q in self.questions if q['index'] in self.scores)
        
        return {
            "current_stage": self.current_stage,
            "current_question_in_stage": self.current_question_index + 1,
            "total_questions_in_stage": len(current_stage_questions),
            "overall_completed": completed_questions,
            "overall_total": total_questions,
            "progress_percentage": round((completed_questions / total_questions) * 100, 1) if total_questions > 0 else 0,
            "emotional_state": self.emotional_context["detected_emotion"],
            "risk_level": self.emotional_context["risk_level"]
        }


# ==================== 测试用例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("HAMD评估器 V3 测试")
    print("=" * 60)
    
    # 创建评估器实例（无LLM模式，使用规则fallback）
    evaluator = HAMDEvaluatorV3()
    
    # 开始评估
    evaluation_id = evaluator.start_evaluation()
    
    # 模拟测试答案
    test_answers = {
        1: "最近心情确实不太好，经常感到悲伤",
        4: "睡眠不好，经常失眠",
        6: "对很多事情都失去了兴趣",
        9: "有时候会焦虑",
        13: "经常感到疲劳"
    }
    
    # 模拟问答流程（阶段1）
    for i in range(5):
        if evaluator.is_evaluation_complete():
            break
        
        current_q = evaluator.get_current_question()
        if not current_q:
            break
        
        question_id = current_q['index']
        print(f"\n[题{question_id}] {current_q['title']}")
        
        # 获取问题（如果有LLM会动态生成，这里使用原问题）
        question_text = current_q['question']
        print(f"问题: {question_text}")
        
        # 模拟用户回答
        answer = test_answers.get(question_id, "还可以吧")
        print(f"回答: {answer}")
        
        # 处理回答
        result = evaluator.process_answer(answer)
        print(f"结果: 评分={result.get('score', 'N/A')}分")
        
        if result.get('stage_transition_needed'):
            print(f"\n{'='*60}")
            print("阶段1完成，准备进入阶段2...")
            print(f"{'='*60}")
            break
    
    print(f"\n{'='*60}")
    print("测试完成")
    print(f"进度: {evaluator.get_progress()}")
    print(f"{'='*60}")

