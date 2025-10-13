# """
# 哈密尔顿抑郁量表（HAMD）评估系统
# 集成智能语音问答、情感分析和动态评分功能
# """

# import json
# import time
# import os
# from typing import Dict, List, Tuple, Optional
# import re
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import asyncio
# import edge_tts
# import pygame


# class HAMDEvaluator:
#     """哈密尔顿抑郁量表评估器"""
    
#     def __init__(self, questions_file: str = "hamd_questions.json", 
#                  model_name: str = None, output_dir: str = "./HAMD_Results/"):
#         """
#         初始化HAMD评估器
        
#         Args:
#             questions_file: 问题配置文件路径
#             model_name: 大语言模型路径
#             output_dir: 结果输出目录
#         """
#         self.questions_file = questions_file
#         self.output_dir = output_dir
#         self.current_question_index = 0
#         self.scores = {}
#         self.detailed_answers = {}
#         self.evaluation_start_time = None
#         self.evaluation_id = None
        
#         # 确保输出目录存在
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 加载问题配置
#         self.questions = self._load_questions()
        
#         # 初始化模型（如果提供）
#         if model_name:
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 torch_dtype="auto",
#                 device_map="auto"
#             )
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         else:
#             self.model = None
#             self.tokenizer = None
            
#         # 初始化音频
#         pygame.mixer.init()
        
#     def _load_questions(self) -> List[Dict]:
#         """加载问题配置"""
#         try:
#             with open(self.questions_file, 'r', encoding='utf-8') as f:
#                 return json.load(f)
#         except FileNotFoundError:
#             raise FileNotFoundError(f"问题配置文件 {self.questions_file} 不存在")
#         except json.JSONDecodeError:
#             raise ValueError(f"问题配置文件 {self.questions_file} 格式错误")
    
#     def start_evaluation(self) -> str:
#         """开始评估，返回评估ID"""
#         self.evaluation_start_time = time.time()
#         self.evaluation_id = f"HAMD_{int(self.evaluation_start_time)}"
#         self.current_question_index = 0
#         self.scores = {}
#         self.detailed_answers = {}
        
#         print(f"=== 哈密尔顿抑郁量表（HAMD-17）评估开始 ===")
#         print(f"评估ID: {self.evaluation_id}")
#         print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.evaluation_start_time))}")
#         print("=" * 50)
        
#         return self.evaluation_id
    
#     def get_current_question(self) -> Optional[Dict]:
#         """获取当前问题"""
#         if self.current_question_index >= len(self.questions):
#             return None
#         return self.questions[self.current_question_index]
    
#     def get_question_prompt(self) -> str:
#         """生成当前问题的提问内容"""
#         current_q = self.get_current_question()
#         if not current_q:
#             return ""
            
#         progress = f"({self.current_question_index + 1}/{len(self.questions)})"
#         prompt = f"现在进行第{self.current_question_index + 1}个问题{progress}，关于{current_q['title']}。\n"
#         prompt += f"{current_q['question']}\n"
#         prompt += "请您详细描述一下您的情况。"
        
#         return prompt
    
#     def analyze_answer_with_llm(self, user_answer: str) -> Tuple[int, str]:
#         """使用大语言模型分析用户回答并评分"""
#         current_q = self.get_current_question()
#         if not current_q or not self.model:
#             return 0, "模型未初始化或问题不存在"
        
#         # 构建评分参考文本
#         scoring_text = "\n".join([f"{score}分: {desc}" for score, desc in current_q['scoring'].items()])
        
#         # 构建提示词
#         prompt = f"""你是一个专业的心理测量智能体，任务是对用户进行哈密尔顿抑郁量表（HAMD-17）的语音问答评估。

# 你现在正在进行第 {current_q['index']} 题：{current_q['question']}

# 评分参考：
# {scoring_text}

# 用户回答如下：
# "{user_answer}"

# 请根据用户的回答内容，结合评分标准，给出0-4分的评分。请分析用户回答中的关键信息，包括：
# 1. 症状严重程度
# 2. 频率和持续时间
# 3. 对日常生活的影响程度
# 4. 情感表达和用词

# 请直接给出评分数字（0-4），然后用一句话说明评分理由。
# 格式：评分：X分，理由：[简要分析]"""

#         try:
#             messages = [
#                 {"role": "system", "content": "你是专业的心理评估专家，擅长分析和评分心理量表。"},
#                 {"role": "user", "content": prompt}
#             ]
            
#             text = self.tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
#             model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
#             generated_ids = self.model.generate(
#                 **model_inputs,
#                 max_new_tokens=256,
#                 temperature=0.3,
#                 do_sample=True
#             )
#             generated_ids = [
#                 output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#             ]
            
#             response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
#             # 解析评分
#             score_match = re.search(r'评分[：:]\s*(\d+)', response)
#             if score_match:
#                 score = int(score_match.group(1))
#                 score = max(0, min(4, score))  # 确保评分在0-4范围内
#             else:
#                 # 尝试其他格式
#                 score_match = re.search(r'(\d+)分', response)
#                 if score_match:
#                     score = int(score_match.group(1))
#                     score = max(0, min(4, score))
#                 else:
#                     score = 0
                    
#             return score, response
            
#         except Exception as e:
#             print(f"LLM分析出错: {e}")
#             return self._rule_based_scoring(user_answer), f"LLM分析失败，使用规则评分: {str(e)}"
    
#     def _rule_based_scoring(self, user_answer: str) -> int:
#         """基于规则的简单评分（备用方案）"""
#         answer_lower = user_answer.lower()
        
#         # 情感关键词权重
#         negative_keywords = {
#             "严重": 3, "很严重": 4, "非常": 3, "极度": 4,
#             "无法": 3, "不能": 2, "困难": 2, "痛苦": 3,
#             "绝望": 4, "想死": 4, "自杀": 4, "活着没意思": 4,
#             "每天": 2, "经常": 2, "总是": 3, "一直": 3,
#             "失眠": 2, "睡不着": 2, "没食欲": 2, "不想吃": 1
#         }
        
#         positive_keywords = {
#             "正常": -2, "还好": -1, "偶尔": -1, "轻微": -1,
#             "没有": -2, "不会": -2, "很好": -2, "挺好": -1
#         }
        
#         score = 0
        
#         # 计算负面情绪得分
#         for keyword, weight in negative_keywords.items():
#             if keyword in answer_lower:
#                 score += weight
                
#         # 计算正面情绪得分
#         for keyword, weight in positive_keywords.items():
#             if keyword in answer_lower:
#                 score += weight
                
#         # 标准化得分到0-4范围
#         if score <= -2:
#             return 0
#         elif score <= 0:
#             return 1
#         elif score <= 2:
#             return 2
#         elif score <= 4:
#             return 3
#         else:
#             return 4
    
#     def process_answer(self, user_answer: str) -> Dict:
#         """处理用户回答"""
#         current_q = self.get_current_question()
#         if not current_q:
#             return {"error": "评估已完成"}
        
#         # 记录详细回答
#         question_id = current_q['index']
#         self.detailed_answers[question_id] = {
#             "question": current_q['question'],
#             "title": current_q['title'],
#             "answer": user_answer,
#             "timestamp": time.time()
#         }
        
#         # 分析并评分
#         if self.model:
#             score, analysis = self.analyze_answer_with_llm(user_answer)
#         else:
#             score = self._rule_based_scoring(user_answer)
#             analysis = "基于规则评分"
        
#         # 记录评分
#         self.scores[question_id] = {
#             "score": score,
#             "analysis": analysis,
#             "scoring_criteria": current_q['scoring'][str(score)]
#         }
        
#         print(f"问题 {question_id}: {current_q['title']}")
#         print(f"用户回答: {user_answer}")
#         print(f"评分: {score}分 - {current_q['scoring'][str(score)]}")
#         print(f"分析: {analysis}")
#         print("-" * 50)
        
#         # 移动到下一题
#         self.current_question_index += 1
        
#         return {
#             "question_id": question_id,
#             "score": score,
#             "analysis": analysis,
#             "is_complete": self.current_question_index >= len(self.questions)
#         }
    
#     def generate_report(self) -> Dict:
#         """生成评估报告"""
#         if not self.scores:
#             return {"error": "没有评估数据"}
        
#         # 计算总分
#         total_score = sum(item["score"] for item in self.scores.values())
        
#         # 生成严重程度评估
#         severity_level = self._get_severity_level(total_score)
        
#         # 分项分析
#         category_analysis = self._analyze_categories()
        
#         # 生成建议
#         recommendations = self._generate_recommendations(total_score, category_analysis)
        
#         report = {
#             "evaluation_id": self.evaluation_id,
#             "total_score": total_score,
#             "max_score": len(self.questions) * 4,
#             "severity_level": severity_level,
#             "detailed_scores": self.scores,
#             "category_analysis": category_analysis,
#             "recommendations": recommendations,
#             "evaluation_time": {
#                 "start_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.evaluation_start_time)),
#                 "duration_minutes": round((time.time() - self.evaluation_start_time) / 60, 1)
#             }
#         }
        
#         # 保存报告
#         self._save_report(report)
        
#         return report
    
#     def _get_severity_level(self, total_score: int) -> Dict:
#         """根据总分判断严重程度"""
#         if total_score <= 7:
#             level = "正常"
#             description = "无明显抑郁症状"
#             color = "green"
#         elif total_score <= 17:
#             level = "轻度抑郁"
#             description = "有轻度抑郁症状，建议关注心理健康"
#             color = "yellow"
#         elif total_score <= 23:
#             level = "中度抑郁"
#             description = "有中度抑郁症状，建议寻求专业帮助"
#             color = "orange"
#         else:
#             level = "重度抑郁"
#             description = "有重度抑郁症状，强烈建议立即寻求专业医疗帮助"
#             color = "red"
        
#         return {
#             "level": level,
#             "description": description,
#             "color": color,
#             "score_range": f"{total_score}/{len(self.questions) * 4}"
#         }
    
#     def _analyze_categories(self) -> Dict:
#         """分项分析"""
#         categories = {
#             "核心症状": [1, 2, 6],  # 抑郁情绪、有罪感、兴趣丧失
#             "认知症状": [10, 15],    # 自我评价、感知能力
#             "生理症状": [4, 5, 7, 8, 13],  # 睡眠、食欲、体重、精神运动、无力感
#             "心理症状": [3, 9, 11, 12, 14],  # 自杀、焦虑、情感平淡、性兴趣、积极性
#             "其他症状": [16]         # 其他症状
#         }
        
#         analysis = {}
#         for category, question_ids in categories.items():
#             scores = [self.scores.get(qid, {}).get("score", 0) for qid in question_ids]
#             total = sum(scores)
#             max_possible = len(scores) * 4
#             percentage = (total / max_possible) * 100 if max_possible > 0 else 0
            
#             analysis[category] = {
#                 "score": total,
#                 "max_score": max_possible,
#                 "percentage": round(percentage, 1),
#                 "severity": "重度" if percentage >= 75 else "中度" if percentage >= 50 else "轻度" if percentage >= 25 else "正常"
#             }
        
#         return analysis
    
#     def _generate_recommendations(self, total_score: int, category_analysis: Dict) -> List[str]:
#         """生成建议"""
#         recommendations = []
        
#         if total_score <= 7:
#             recommendations.append("您的心理状态良好，建议保持健康的生活方式。")
#             recommendations.append("继续维持规律作息、适量运动和良好的社交关系。")
#         elif total_score <= 17:
#             recommendations.append("建议关注自己的心理健康状态。")
#             recommendations.append("可以尝试放松技巧，如深呼吸、冥想或适度运动。")
#             recommendations.append("如果症状持续，建议咨询心理健康专业人士。")
#         elif total_score <= 23:
#             recommendations.append("建议尽快寻求专业心理咨询或治疗。")
#             recommendations.append("可以考虑心理治疗，如认知行为疗法。")
#             recommendations.append("保持与家人朋友的联系，寻求社会支持。")
#         else:
#             recommendations.append("强烈建议立即寻求专业医疗帮助。")
#             recommendations.append("可能需要药物治疗结合心理治疗。")
#             recommendations.append("请不要独自承担，及时联系医生或心理健康专家。")
            
#             # 检查自杀风险
#             if self.scores.get(3, {}).get("score", 0) >= 2:
#                 recommendations.append("⚠️ 重要提醒：如有自杀念头，请立即拨打心理危机干预热线：400-161-9995")
        
#         # 根据分项分析添加针对性建议
#         for category, data in category_analysis.items():
#             if data["percentage"] >= 75:
#                 if category == "生理症状":
#                     recommendations.append(f"您在{category}方面得分较高，建议注意改善睡眠质量和饮食习惯。")
#                 elif category == "心理症状":
#                     recommendations.append(f"您在{category}方面得分较高，建议学习情绪管理技巧。")
        
#         return recommendations
    
#     def _save_report(self, report: Dict):
#         """保存评估报告"""
#         filename = f"{self.output_dir}/HAMD_Report_{self.evaluation_id}.json"
#         try:
#             with open(filename, 'w', encoding='utf-8') as f:
#                 json.dump(report, f, ensure_ascii=False, indent=2)
#             print(f"评估报告已保存至: {filename}")
#         except Exception as e:
#             print(f"保存报告失败: {e}")
    
#     async def speak_text(self, text: str, voice: str = "zh-CN-XiaoyiNeural"):
#         """语音播报"""
#         try:
#             output_file = f"{self.output_dir}/temp_speech.mp3"
#             communicate = edge_tts.Communicate(text, voice)
#             await communicate.save(output_file)
            
#             pygame.mixer.music.load(output_file)
#             pygame.mixer.music.play()
#             while pygame.mixer.music.get_busy():
#                 await asyncio.sleep(0.1)
                
#         except Exception as e:
#             print(f"语音播报失败: {e}")
    
#     def get_progress(self) -> Dict:
#         """获取评估进度"""
#         return {
#             "current_question": self.current_question_index + 1,
#             "total_questions": len(self.questions),
#             "progress_percentage": round((self.current_question_index / len(self.questions)) * 100, 1),
#             "completed_questions": self.current_question_index,
#             "remaining_questions": len(self.questions) - self.current_question_index
#         }
    
#     def is_evaluation_complete(self) -> bool:
#         """检查评估是否完成"""
#         return self.current_question_index >= len(self.questions)
    
#     def reset_evaluation(self):
#         """重置评估"""
#         self.current_question_index = 0
#         self.scores = {}
#         self.detailed_answers = {}
#         self.evaluation_start_time = None
#         self.evaluation_id = None
#         print("评估已重置")


# # 测试用例
# if __name__ == "__main__":
#     # 创建评估器实例
#     evaluator = HAMDEvaluator()
    
#     # 开始评估
#     evaluation_id = evaluator.start_evaluation()
    
#     # 模拟问答流程
#     test_answers = [
#         "我最近心情确实不太好，经常感到悲伤，对很多事情都失去了兴趣",
#         "有时候会觉得自己做错了什么，但不是很严重",
#         "没有自杀的念头",
#         "睡眠不太好，经常失眠",
#         "食欲还可以，没什么大问题",
#         "对工作和社交确实兴趣不大",
#         "体重没有明显变化",
#         "感觉做事比以前慢一些，但还可以完成",
#         "有时候会焦虑，但不严重",
#         "觉得自己不如别人",
#         "有时候感觉和别人有距离感",
#         "这方面还正常",
#         "经常感到疲劳",
#         "做事情的积极性不如以前",
#         "感知能力正常",
#         "没有其他特别的症状"
#     ]
    
#     # 处理所有回答
#     for i, answer in enumerate(test_answers):
#         if evaluator.is_evaluation_complete():
#             break
            
#         current_q = evaluator.get_current_question()
#         print(f"\n问题 {current_q['index']}: {current_q['question']}")
#         print(f"回答: {answer}")
        
#         result = evaluator.process_answer(answer)
#         print(f"评分结果: {result}")
    
#     # 生成报告
#     if evaluator.is_evaluation_complete():
#         report = evaluator.generate_report()
#         print(f"\n=== 评估报告 ===")
#         print(f"总分: {report['total_score']}/{report['max_score']}")
#         print(f"严重程度: {report['severity_level']['level']}")
#         print(f"描述: {report['severity_level']['description']}")



#第二个版本
# """
# 哈密尔顿抑郁量表（HAMD）评估系统
# 集成智能语音问答、情感分析和动态评分功能
# """

# import json
# import time
# import os
# from typing import Dict, List, Tuple, Optional, Callable
# import re
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import asyncio
# import edge_tts
# import pygame


# class HAMDEvaluator:
#     """哈密尔顿抑郁量表评估器"""
    
#     def __init__(self, questions_file: str = "hamd_questions.json", 
#                  model_name: str = None, output_dir: str = "./HAMD_Results/",
#                  chat_fn: Optional[Callable[[List[Dict], float, int], str]] = None):
#         """
#         初始化HAMD评估器
        
#         Args:
#             questions_file: 问题配置文件路径
#             model_name: 大语言模型路径
#             output_dir: 结果输出目录
#         """
#         self.questions_file = questions_file
#         self.output_dir = output_dir
#         self.current_question_index = 0
#         self.scores = {}
#         self.detailed_answers = {}
#         self.evaluation_start_time = None
#         self.evaluation_id = None
        
#         # 确保输出目录存在
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 加载问题配置
#         self.questions = self._load_questions()
        
#         # 初始化模型（如果提供）
#         if model_name:
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 torch_dtype="auto",
#                 device_map="auto"
#             )
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         else:
#             self.model = None
#             self.tokenizer = None
        
#         # 外部聊天函数（例如 DashScope 适配器）
#         self.chat_fn = chat_fn
            
#         # 初始化音频
#         pygame.mixer.init()
        
#     def _load_questions(self) -> List[Dict]:
#         """加载问题配置"""
#         try:
#             with open(self.questions_file, 'r', encoding='utf-8') as f:
#                 return json.load(f)
#         except FileNotFoundError:
#             raise FileNotFoundError(f"问题配置文件 {self.questions_file} 不存在")
#         except json.JSONDecodeError:
#             raise ValueError(f"问题配置文件 {self.questions_file} 格式错误")
    
#     def start_evaluation(self) -> str:
#         """开始评估，返回评估ID"""
#         self.evaluation_start_time = time.time()
#         self.evaluation_id = f"HAMD_{int(self.evaluation_start_time)}"
#         self.current_question_index = 0
#         self.scores = {}
#         self.detailed_answers = {}
        
#         print(f"=== 哈密尔顿抑郁量表（HAMD-17）评估开始 ===")
#         print(f"评估ID: {self.evaluation_id}")
#         print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.evaluation_start_time))}")
#         print("=" * 50)
        
#         return self.evaluation_id
    
#     def get_current_question(self) -> Optional[Dict]:
#         """获取当前问题"""
#         if self.current_question_index >= len(self.questions):
#             return None
#         return self.questions[self.current_question_index]
    
#     def get_question_prompt(self) -> str:
#         """生成当前问题的提问内容"""
#         current_q = self.get_current_question()
#         if not current_q:
#             return ""
            
#         progress = f"({self.current_question_index + 1}/{len(self.questions)})"
#         prompt = f"现在进行第{self.current_question_index + 1}个问题{progress}，关于{current_q['title']}。\n"
#         prompt += f"{current_q['question']}\n"
#         prompt += "请您详细描述一下您的情况。"
        
#         return prompt
    
#     def analyze_answer_with_llm(self, user_answer: str) -> Tuple[int, str]:
#         """使用大语言模型（本地或chat_fn）分析用户回答并评分"""
#         current_q = self.get_current_question()
#         if not current_q:
#             return 0, "问题不存在"
        
#         # 构建评分参考文本
#         scoring_text = "\n".join([f"{score}分: {desc}" for score, desc in current_q['scoring'].items()])
        
#         # 构建提示词
#         prompt = f"""你是一个专业的心理测量智能体，任务是对用户进行哈密尔顿抑郁量表（HAMD-17）的语音问答评估。

# 你现在正在进行第 {current_q['index']} 题：{current_q['question']}

# 评分参考：
# {scoring_text}

# 用户回答如下：
# "{user_answer}"

# 请根据用户的回答内容，结合评分标准，给出0-4分的评分。请分析用户回答中的关键信息，包括：
# 1. 症状严重程度
# 2. 频率和持续时间
# 3. 对日常生活的影响程度
# 4. 情感表达和用词

# 请直接给出评分数字（0-4），然后用一句话说明评分理由。
# 格式：评分：X分，理由：[简要分析]"""

#         try:
#             messages = [
#                 {"role": "system", "content": "你是专业的心理评估专家，擅长分析和评分心理量表。"},
#                 {"role": "user", "content": prompt}
#             ]
            
#             if self.chat_fn is not None:
#                 response = self.chat_fn(messages, 0.3, 256)
#             elif self.model is not None and self.tokenizer is not None:
#                 text = self.tokenizer.apply_chat_template(
#                     messages,
#                     tokenize=False,
#                     add_generation_prompt=True
#                 )
#                 model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
#                 generated_ids = self.model.generate(
#                     **model_inputs,
#                     max_new_tokens=256,
#                     temperature=0.3,
#                     do_sample=True
#                 )
#                 generated_ids = [
#                     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#                 ]
#                 response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#             else:
#                 return self._rule_based_scoring(user_answer), "未配置LLM，使用规则评分"
            
#             # 解析评分
#             score_match = re.search(r'评分[：:]\s*(\d+)', response)
#             if score_match:
#                 score = int(score_match.group(1))
#                 score = max(0, min(4, score))  # 确保评分在0-4范围内
#             else:
#                 # 尝试其他格式
#                 score_match = re.search(r'(\d+)分', response)
#                 if score_match:
#                     score = int(score_match.group(1))
#                     score = max(0, min(4, score))
#                 else:
#                     score = 0
                    
#             return score, response
            
#         except Exception as e:
#             print(f"LLM分析出错: {e}")
#             return self._rule_based_scoring(user_answer), f"LLM分析失败，使用规则评分: {str(e)}"
    
#     def _rule_based_scoring(self, user_answer: str) -> int:
#         """基于规则的简单评分（备用方案）"""
#         answer_lower = user_answer.lower()
        
#         # 情感关键词权重
#         negative_keywords = {
#             "严重": 3, "很严重": 4, "非常": 3, "极度": 4,
#             "无法": 3, "不能": 2, "困难": 2, "痛苦": 3,
#             "绝望": 4, "想死": 4, "自杀": 4, "活着没意思": 4,
#             "每天": 2, "经常": 2, "总是": 3, "一直": 3,
#             "失眠": 2, "睡不着": 2, "没食欲": 2, "不想吃": 1
#         }
        
#         positive_keywords = {
#             "正常": -2, "还好": -1, "偶尔": -1, "轻微": -1,
#             "没有": -2, "不会": -2, "很好": -2, "挺好": -1
#         }
        
#         score = 0
        
#         # 计算负面情绪得分
#         for keyword, weight in negative_keywords.items():
#             if keyword in answer_lower:
#                 score += weight
                
#         # 计算正面情绪得分
#         for keyword, weight in positive_keywords.items():
#             if keyword in answer_lower:
#                 score += weight
                
#         # 标准化得分到0-4范围
#         if score <= -2:
#             return 0
#         elif score <= 0:
#             return 1
#         elif score <= 2:
#             return 2
#         elif score <= 4:
#             return 3
#         else:
#             return 4
    
#     def process_answer(self, user_answer: str) -> Dict:
#         """处理用户回答"""
#         current_q = self.get_current_question()
#         if not current_q:
#             return {"error": "评估已完成"}
        
#         # 记录详细回答
#         question_id = current_q['index']
#         self.detailed_answers[question_id] = {
#             "question": current_q['question'],
#             "title": current_q['title'],
#             "answer": user_answer,
#             "timestamp": time.time()
#         }
        
#         # 分析并评分
#         if self.model or self.chat_fn:
#             score, analysis = self.analyze_answer_with_llm(user_answer)
#         else:
#             score = self._rule_based_scoring(user_answer)
#             analysis = "基于规则评分"
        
#         # 记录评分
#         self.scores[question_id] = {
#             "score": score,
#             "analysis": analysis,
#             "scoring_criteria": current_q['scoring'][str(score)]
#         }
        
#         print(f"问题 {question_id}: {current_q['title']}")
#         print(f"用户回答: {user_answer}")
#         print(f"评分: {score}分 - {current_q['scoring'][str(score)]}")
#         print(f"分析: {analysis}")
#         print("-" * 50)
        
#         # 移动到下一题
#         self.current_question_index += 1
        
#         return {
#             "question_id": question_id,
#             "score": score,
#             "analysis": analysis,
#             "is_complete": self.current_question_index >= len(self.questions)
#         }
    
#     def generate_report(self) -> Dict:
#         """生成评估报告"""
#         if not self.scores:
#             return {"error": "没有评估数据"}
        
#         # 计算总分
#         total_score = sum(item["score"] for item in self.scores.values())
        
#         # 生成严重程度评估
#         severity_level = self._get_severity_level(total_score)
        
#         # 分项分析
#         category_analysis = self._analyze_categories()
        
#         # 生成建议
#         recommendations = self._generate_recommendations(total_score, category_analysis)
        
#         report = {
#             "evaluation_id": self.evaluation_id,
#             "total_score": total_score,
#             "max_score": len(self.questions) * 4,
#             "severity_level": severity_level,
#             "detailed_scores": self.scores,
#             "category_analysis": category_analysis,
#             "recommendations": recommendations,
#             "evaluation_time": {
#                 "start_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.evaluation_start_time)),
#                 "duration_minutes": round((time.time() - self.evaluation_start_time) / 60, 1)
#             }
#         }
        
#         # 保存报告
#         self._save_report(report)
        
#         return report
    
#     def _get_severity_level(self, total_score: int) -> Dict:
#         """根据总分判断严重程度"""
#         if total_score <= 7:
#             level = "正常"
#             description = "无明显抑郁症状"
#             color = "green"
#         elif total_score <= 17:
#             level = "轻度抑郁"
#             description = "有轻度抑郁症状，建议关注心理健康"
#             color = "yellow"
#         elif total_score <= 23:
#             level = "中度抑郁"
#             description = "有中度抑郁症状，建议寻求专业帮助"
#             color = "orange"
#         else:
#             level = "重度抑郁"
#             description = "有重度抑郁症状，强烈建议立即寻求专业医疗帮助"
#             color = "red"
        
#         return {
#             "level": level,
#             "description": description,
#             "color": color,
#             "score_range": f"{total_score}/{len(self.questions) * 4}"
#         }
    
#     def _analyze_categories(self) -> Dict:
#         """分项分析"""
#         categories = {
#             "核心症状": [1, 2, 6],  # 抑郁情绪、有罪感、兴趣丧失
#             "认知症状": [10, 15],    # 自我评价、感知能力
#             "生理症状": [4, 5, 7, 8, 13],  # 睡眠、食欲、体重、精神运动、无力感
#             "心理症状": [3, 9, 11, 12, 14],  # 自杀、焦虑、情感平淡、性兴趣、积极性
#             "其他症状": [16]         # 其他症状
#         }
        
#         analysis = {}
#         for category, question_ids in categories.items():
#             scores = [self.scores.get(qid, {}).get("score", 0) for qid in question_ids]
#             total = sum(scores)
#             max_possible = len(scores) * 4
#             percentage = (total / max_possible) * 100 if max_possible > 0 else 0
            
#             analysis[category] = {
#                 "score": total,
#                 "max_score": max_possible,
#                 "percentage": round(percentage, 1),
#                 "severity": "重度" if percentage >= 75 else "中度" if percentage >= 50 else "轻度" if percentage >= 25 else "正常"
#             }
        
#         return analysis
    
#     def _generate_recommendations(self, total_score: int, category_analysis: Dict) -> List[str]:
#         """生成建议"""
#         recommendations = []
        
#         if total_score <= 7:
#             recommendations.append("您的心理状态良好，建议保持健康的生活方式。")
#             recommendations.append("继续维持规律作息、适量运动和良好的社交关系。")
#         elif total_score <= 17:
#             recommendations.append("建议关注自己的心理健康状态。")
#             recommendations.append("可以尝试放松技巧，如深呼吸、冥想或适度运动。")
#             recommendations.append("如果症状持续，建议咨询心理健康专业人士。")
#         elif total_score <= 23:
#             recommendations.append("建议尽快寻求专业心理咨询或治疗。")
#             recommendations.append("可以考虑心理治疗，如认知行为疗法。")
#             recommendations.append("保持与家人朋友的联系，寻求社会支持。")
#         else:
#             recommendations.append("强烈建议立即寻求专业医疗帮助。")
#             recommendations.append("可能需要药物治疗结合心理治疗。")
#             recommendations.append("请不要独自承担，及时联系医生或心理健康专家。")
            
#             # 检查自杀风险
#             if self.scores.get(3, {}).get("score", 0) >= 2:
#                 recommendations.append("⚠️ 重要提醒：如有自杀念头，请立即拨打心理危机干预热线：400-161-9995")
        
#         # 根据分项分析添加针对性建议
#         for category, data in category_analysis.items():
#             if data["percentage"] >= 75:
#                 if category == "生理症状":
#                     recommendations.append(f"您在{category}方面得分较高，建议注意改善睡眠质量和饮食习惯。")
#                 elif category == "心理症状":
#                     recommendations.append(f"您在{category}方面得分较高，建议学习情绪管理技巧。")
        
#         return recommendations
    
#     def _save_report(self, report: Dict):
#         """保存评估报告"""
#         filename = f"{self.output_dir}/HAMD_Report_{self.evaluation_id}.json"
#         try:
#             with open(filename, 'w', encoding='utf-8') as f:
#                 json.dump(report, f, ensure_ascii=False, indent=2)
#             print(f"评估报告已保存至: {filename}")
#         except Exception as e:
#             print(f"保存报告失败: {e}")
    
#     async def speak_text(self, text: str, voice: str = "zh-CN-XiaoyiNeural"):
#         """语音播报"""
#         try:
#             output_file = f"{self.output_dir}/temp_speech.mp3"
#             communicate = edge_tts.Communicate(text, voice)
#             await communicate.save(output_file)
            
#             pygame.mixer.music.load(output_file)
#             pygame.mixer.music.play()
#             while pygame.mixer.music.get_busy():
#                 await asyncio.sleep(0.1)
                
#         except Exception as e:
#             print(f"语音播报失败: {e}")
    
#     def get_progress(self) -> Dict:
#         """获取评估进度"""
#         return {
#             "current_question": self.current_question_index + 1,
#             "total_questions": len(self.questions),
#             "progress_percentage": round((self.current_question_index / len(self.questions)) * 100, 1),
#             "completed_questions": self.current_question_index,
#             "remaining_questions": len(self.questions) - self.current_question_index
#         }
    
#     def is_evaluation_complete(self) -> bool:
#         """检查评估是否完成"""
#         return self.current_question_index >= len(self.questions)
    
#     def reset_evaluation(self):
#         """重置评估"""
#         self.current_question_index = 0
#         self.scores = {}
#         self.detailed_answers = {}
#         self.evaluation_start_time = None
#         self.evaluation_id = None
#         print("评估已重置")


# # 测试用例
# if __name__ == "__main__":
#     # 创建评估器实例
#     evaluator = HAMDEvaluator()
    
#     # 开始评估
#     evaluation_id = evaluator.start_evaluation()
    
#     # 模拟问答流程
#     test_answers = [
#         "我最近心情确实不太好，经常感到悲伤，对很多事情都失去了兴趣",
#         "有时候会觉得自己做错了什么，但不是很严重",
#         "没有自杀的念头",
#         "睡眠不太好，经常失眠",
#         "食欲还可以，没什么大问题",
#         "对工作和社交确实兴趣不大",
#         "体重没有明显变化",
#         "感觉做事比以前慢一些，但还可以完成",
#         "有时候会焦虑，但不严重",
#         "觉得自己不如别人",
#         "有时候感觉和别人有距离感",
#         "这方面还正常",
#         "经常感到疲劳",
#         "做事情的积极性不如以前",
#         "感知能力正常",
#         "没有其他特别的症状"
#     ]
    
#     # 处理所有回答
#     for i, answer in enumerate(test_answers):
#         if evaluator.is_evaluation_complete():
#             break
            
#         current_q = evaluator.get_current_question()
#         print(f"\n问题 {current_q['index']}: {current_q['question']}")
#         print(f"回答: {answer}")
        
#         result = evaluator.process_answer(answer)
#         print(f"评分结果: {result}")
    
#     # 生成报告
#     if evaluator.is_evaluation_complete():
#         report = evaluator.generate_report()
#         print(f"\n=== 评估报告 ===")
#         print(f"总分: {report['total_score']}/{report['max_score']}")
#         print(f"严重程度: {report['severity_level']['level']}")
#         print(f"描述: {report['severity_level']['description']}")


"""
哈密尔顿抑郁量表（HAMD）评估系统
集成智能语音问答、情感分析和动态评分功能
"""

import json
import time
import os
from typing import Dict, List, Tuple, Optional, Callable
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
import edge_tts
import pygame


class HAMDEvaluator:
    """哈密尔顿抑郁量表评估器"""
    
    def __init__(self, questions_file: str = "hamd_questions.json", 
                 model_name: str = None, output_dir: str = "./HAMD_Results/",
                 chat_fn: Optional[Callable[[List[Dict], float, int], str]] = None):
        """
        初始化HAMD评估器
        
        Args:
            questions_file: 问题配置文件路径
            model_name: 大语言模型路径
            output_dir: 结果输出目录
        """
        self.questions_file = questions_file
        self.output_dir = output_dir
        self.current_question_index = 0
        self.scores = {}
        self.detailed_answers = {}
        self.evaluation_start_time = None
        self.evaluation_id = None
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载问题配置
        self.questions = self._load_questions()
        
        # 初始化模型（如果提供）
        if model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.model = None
            self.tokenizer = None
        
        # 外部聊天函数（例如 DashScope 适配器）
        self.chat_fn = chat_fn
            
        # 初始化音频
        pygame.mixer.init()
        
    def _load_questions(self) -> List[Dict]:
        """加载问题配置"""
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"问题配置文件 {self.questions_file} 不存在")
        except json.JSONDecodeError:
            raise ValueError(f"问题配置文件 {self.questions_file} 格式错误")
    
    def start_evaluation(self) -> str:
        """开始评估，返回评估ID"""
        self.evaluation_start_time = time.time()
        self.evaluation_id = f"HAMD_{int(self.evaluation_start_time)}"
        self.current_question_index = 0
        self.scores = {}
        self.detailed_answers = {}
        
        # 控制台保留简要日志
        print(f"[HAMD] Start, ID={self.evaluation_id}, Time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.evaluation_start_time))}")
        
        return self.evaluation_id
    
    def get_current_question(self) -> Optional[Dict]:
        """获取当前问题"""
        if self.current_question_index >= len(self.questions):
            return None
        return self.questions[self.current_question_index]
    
    def get_question_prompt(self) -> str:
        """生成当前问题的提问内容（不包含 i/n 进度文案）"""
        current_q = self.get_current_question()
        if not current_q:
            return ""
        
        # 多样化问法：随机挑选更自然的提示
        variants = [
            f"{current_q['question']}",
            f"想了解一下：{current_q['question']}",
            f"可以聊聊：{current_q['question']}",
            f"这方面的情况是：{current_q['question']}"
        ]
        try:
            import random as _random
            chosen = _random.choice(variants)
        except Exception:
            chosen = variants[0]
        prompt = chosen + "\n请结合最近一周的感受简单说说。"
        
        return prompt
    
    def analyze_answer_with_llm(self, user_answer: str) -> Tuple[int, str]:
        """使用大语言模型（本地或chat_fn）分析用户回答并评分"""
        current_q = self.get_current_question()
        if not current_q:
            return 0, "问题不存在"
        
        # 构建评分参考文本
        scoring_text = "\n".join([f"{score}分: {desc}" for score, desc in current_q['scoring'].items()])
        
        # 构建提示词
        prompt = f"""你是一个专业的心理测量智能体，任务是对用户进行哈密尔顿抑郁量表（HAMD-17）的语音问答评估。

你现在正在进行第 {current_q['index']} 题：{current_q['question']}

评分参考：
{scoring_text}

用户回答如下：
"{user_answer}"

请根据用户的回答内容，结合评分标准，给出0-4分的评分。请分析用户回答中的关键信息，包括：
1. 症状严重程度
2. 频率和持续时间
3. 对日常生活的影响程度
4. 情感表达和用词

请直接给出评分数字（0-4），然后用一句话说明评分理由。
格式：评分：X分，理由：[简要分析]"""

        try:
            messages = [
                {"role": "system", "content": "你是专业的心理评估专家，擅长分析和评分心理量表。"},
                {"role": "user", "content": prompt}
            ]
            
            if self.chat_fn is not None:
                response = self.chat_fn(messages, 0.3, 256)
            elif self.model is not None and self.tokenizer is not None:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    do_sample=True
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                return self._rule_based_scoring(user_answer), "未配置LLM，使用规则评分"
            
            # 解析评分
            score_match = re.search(r'评分[：:]\s*(\d+)', response)
            if score_match:
                score = int(score_match.group(1))
                score = max(0, min(4, score))  # 确保评分在0-4范围内
            else:
                # 尝试其他格式
                score_match = re.search(r'(\d+)分', response)
                if score_match:
                    score = int(score_match.group(1))
                    score = max(0, min(4, score))
                else:
                    score = 0
                    
            return score, response
            
        except Exception as e:
            print(f"LLM分析出错: {e}")
            return self._rule_based_scoring(user_answer), f"LLM分析失败，使用规则评分: {str(e)}"
    
    def _rule_based_scoring(self, user_answer: str) -> int:
        """基于规则的简单评分（备用方案）"""
        answer_lower = user_answer.lower()
        
        # 情感关键词权重
        negative_keywords = {
            "严重": 3, "很严重": 4, "非常": 3, "极度": 4,
            "无法": 3, "不能": 2, "困难": 2, "痛苦": 3,
            "绝望": 4, "想死": 4, "自杀": 4, "活着没意思": 4,
            "每天": 2, "经常": 2, "总是": 3, "一直": 3,
            "失眠": 2, "睡不着": 2, "没食欲": 2, "不想吃": 1
        }
        
        positive_keywords = {
            "正常": -2, "还好": -1, "偶尔": -1, "轻微": -1,
            "没有": -2, "不会": -2, "很好": -2, "挺好": -1
        }
        
        score = 0
        
        # 计算负面情绪得分
        for keyword, weight in negative_keywords.items():
            if keyword in answer_lower:
                score += weight
                
        # 计算正面情绪得分
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
    
    def process_answer(self, user_answer: str) -> Dict:
        """处理用户回答"""
        current_q = self.get_current_question()
        if not current_q:
            return {"error": "评估已完成"}
        
        # 记录详细回答
        question_id = current_q['index']
        self.detailed_answers[question_id] = {
            "question": current_q['question'],
            "title": current_q['title'],
            "answer": user_answer,
            "timestamp": time.time()
        }
        
        # 分析并评分
        if self.model or self.chat_fn:
            score, analysis = self.analyze_answer_with_llm(user_answer)
        else:
            score = self._rule_based_scoring(user_answer)
            analysis = "基于规则评分"
        
        # 记录评分
        self.scores[question_id] = {
            "score": score,
            "analysis": analysis,
            "scoring_criteria": current_q['scoring'][str(score)]
        }
        
        print(f"问题 {question_id}: {current_q['title']}")
        print(f"用户回答: {user_answer}")
        print(f"评分: {score}分 - {current_q['scoring'][str(score)]}")
        print(f"分析: {analysis}")
        print("-" * 50)
        
        # 移动到下一题
        self.current_question_index += 1
        
        return {
            "question_id": question_id,
            "score": score,
            "analysis": analysis,
            "is_complete": self.current_question_index >= len(self.questions)
        }
    
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
        
        report = {
            "evaluation_id": self.evaluation_id,
            "total_score": total_score,
            "max_score": len(self.questions) * 4,
            "severity_level": severity_level,
            "detailed_scores": self.scores,
            "category_analysis": category_analysis,
            "recommendations": recommendations,
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
            "核心症状": [1, 2, 6],  # 抑郁情绪、有罪感、兴趣丧失
            "认知症状": [10, 15],    # 自我评价、感知能力
            "生理症状": [4, 5, 7, 8, 13],  # 睡眠、食欲、体重、精神运动、无力感
            "心理症状": [3, 9, 11, 12, 14],  # 自杀、焦虑、情感平淡、性兴趣、积极性
            "其他症状": [16]         # 其他症状
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
                recommendations.append("⚠️ 重要提醒：如有自杀念头，请立即拨打心理危机干预热线：400-161-9995")
        
        # 根据分项分析添加针对性建议
        for category, data in category_analysis.items():
            if data["percentage"] >= 75:
                if category == "生理症状":
                    recommendations.append(f"您在{category}方面得分较高，建议注意改善睡眠质量和饮食习惯。")
                elif category == "心理症状":
                    recommendations.append(f"您在{category}方面得分较高，建议学习情绪管理技巧。")
        
        return recommendations
    
    def _save_report(self, report: Dict):
        """保存评估报告"""
        filename = f"{self.output_dir}/HAMD_Report_{self.evaluation_id}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"评估报告已保存至: {filename}")
        except Exception as e:
            print(f"保存报告失败: {e}")
    
    async def speak_text(self, text: str, voice: str = "zh-CN-XiaoyiNeural"):
        """语音播报"""
        try:
            output_file = f"{self.output_dir}/temp_speech.mp3"
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_file)
            
            pygame.mixer.music.load(output_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
                
        except Exception as e:
            print(f"语音播报失败: {e}")
    
    def get_progress(self) -> Dict:
        """获取评估进度"""
        return {
            "current_question": self.current_question_index + 1,
            "total_questions": len(self.questions),
            "progress_percentage": round((self.current_question_index / len(self.questions)) * 100, 1),
            "completed_questions": self.current_question_index,
            "remaining_questions": len(self.questions) - self.current_question_index
        }
    
    def is_evaluation_complete(self) -> bool:
        """检查评估是否完成"""
        return self.current_question_index >= len(self.questions)
    
    def previous_question(self) -> bool:
        """回到上一题，并撤销上一题已记录的答案与评分。"""
        if self.current_question_index <= 0:
            return False
        # 定位上一题并撤销记录
        prev_q = self.questions[self.current_question_index - 1]
        prev_id = prev_q['index']
        if prev_id in self.scores:
            del self.scores[prev_id]
        if prev_id in self.detailed_answers:
            del self.detailed_answers[prev_id]
        # 指针后退
        self.current_question_index -= 1
        return True

    def next_question(self) -> bool:
        """跳到下一题（不记录当前题答案）。"""
        if self.current_question_index >= len(self.questions):
            return False
        self.current_question_index += 1
        return self.current_question_index <= len(self.questions)

    def jump_to_question(self, target_index: int) -> bool:
        """跳转到指定题号（1-based），并清除该题及之后题目的记录。"""
        if not isinstance(target_index, int):
            return False
        if target_index < 1 or target_index > len(self.questions):
            return False
        # 清除从 target_index 开始的记录
        for q in self.questions:
            qid = q.get('index')
            if isinstance(qid, int) and qid >= target_index:
                if qid in self.scores:
                    del self.scores[qid]
                if qid in self.detailed_answers:
                    del self.detailed_answers[qid]
        # 设置指针到目标题（0-based）
        self.current_question_index = target_index - 1
        return True

    def reset_evaluation(self):
        """重置评估"""
        self.current_question_index = 0
        self.scores = {}
        self.detailed_answers = {}
        self.evaluation_start_time = None
        self.evaluation_id = None
        print("评估已重置")


# 测试用例
if __name__ == "__main__":
    # 创建评估器实例
    evaluator = HAMDEvaluator()
    
    # 开始评估
    evaluation_id = evaluator.start_evaluation()
    
    # 模拟问答流程
    test_answers = [
        "我最近心情确实不太好，经常感到悲伤，对很多事情都失去了兴趣",
        "有时候会觉得自己做错了什么，但不是很严重",
        "没有自杀的念头",
        "睡眠不太好，经常失眠",
        "食欲还可以，没什么大问题",
        "对工作和社交确实兴趣不大",
        "体重没有明显变化",
        "感觉做事比以前慢一些，但还可以完成",
        "有时候会焦虑，但不严重",
        "觉得自己不如别人",
        "有时候感觉和别人有距离感",
        "这方面还正常",
        "经常感到疲劳",
        "做事情的积极性不如以前",
        "感知能力正常",
        "没有其他特别的症状"
    ]
    
    # 处理所有回答
    for i, answer in enumerate(test_answers):
        if evaluator.is_evaluation_complete():
            break
            
        current_q = evaluator.get_current_question()
        print(f"\n问题 {current_q['index']}: {current_q['question']}")
        print(f"回答: {answer}")
        
        result = evaluator.process_answer(answer)
        print(f"评分结果: {result}")
    
    # 生成报告
    if evaluator.is_evaluation_complete():
        report = evaluator.generate_report()
        print(f"\n=== 评估报告 ===")
        print(f"总分: {report['total_score']}/{report['max_score']}")
        print(f"严重程度: {report['severity_level']['level']}")
        print(f"描述: {report['severity_level']['description']}")
