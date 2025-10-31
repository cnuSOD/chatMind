# HAMD V3 心理评估系统 - Phase 3 完整方案

## 📋 概述

这是哈密尔顿抑郁量表（HAMD）的第三代智能评估系统，实现了**三阶段动态评估**、**LLM自然提问**、**智能追问**、**情绪感知**和**高风险监控**等高级功能。

## 🎯 核心特性

### 1. 三阶段评估架构

```
阶段1：初步筛查（5题）
├── 抑郁情绪（题1）         ⭐ 核心指标
├── 睡眠障碍（题4）         
├── 兴趣丧失（题6）         ⭐ 核心指标
├── 焦虑（题9）             
└── 无力感（题13）          

↓ 自动进入阶段2

阶段2：深入评估（6题）
├── 自杀意图（题3）         🚨 高风险指标
├── 有罪感（题2）           
├── 精神运动迟缓（题8）     
├── 自我评价（题10）        
├── 食欲减退（题5）         
└── 情感平淡（题11）        

↓ 自动进入阶段3

阶段3：全面评估（5题）
├── 体重减轻（题7）         
├── 积极性（题14）          
├── 性兴趣减退（题12）      
├── 感知能力（题15）        
└── 其他症状（题16）        
```

### 2. LLM动态提问

**传统问法（生硬）：**
> "您最近几天有没有觉得心情低落、悲伤、绝望或失去兴趣？"

**V3动态生成（自然）：**
> "最近心情怎么样？有没有觉得特别难过或者提不起劲？"
> "这几天感觉开心吗？还是经常会感到心情低落？"
> "能说说您最近的心情状态吗？有什么感受？"

**实现原理：**
```python
# 根据问题意图 + 上下文 + 情绪状态 → LLM生成自然问法
question_prompt = evaluator.get_question_prompt()

# 每次提问都是新的，考虑：
# 1. 问题评估目的（intent_description）
# 2. 之前的对话内容（conversation_history）
# 3. 用户当前情绪（emotional_context）
# 4. 问题优先级和风险等级
```

### 3. 智能追问机制

**触发条件：**
- 回答清晰度 < 0.5（模糊回答）
- 高风险问题（自杀意图）且清晰度 < 0.8
- 未达到最大追问次数（默认2次）

**示例：**

```
问：最近心情怎么样？
答：还行吧。                          # 清晰度评分：0.3（模糊）

追问1：能具体说说是什么感觉吗？
答：就是有时候会不太开心。          # 清晰度评分：0.5（一般）

追问2：这种不开心的感觉经常出现吗？会影响到日常生活吗？
答：每天都会有，做什么都提不起劲。  # 清晰度评分：0.9（清晰）

→ 评分：2分（明确情绪低落，影响日常）
```

### 4. 情绪感知与风格调整

**情绪分析：**
```python
{
    "emotion": "negative",      # positive/neutral/negative/critical
    "intensity": 7,             # 0-10分
    "keywords": ["悲伤", "绝望", "痛苦"]
}
```

**自适应语气：**

| 情绪状态 | 强度 | 语气策略 | 示例 |
|---------|------|---------|------|
| critical | 8-10 | 极度温和、充满关怀 | "我理解您现在很不容易，咱们慢慢说..." |
| negative | 5-7 | 温暖、共情 | "听起来您最近确实挺难的，能再说说吗？" |
| neutral | 2-4 | 自然、亲切 | "好的，那我们继续下一个问题" |
| positive | 0-1 | 轻松、友好 | "听起来不错，继续保持哦" |

### 5. 高风险实时监控

**监控指标：**
- **自杀风险评分**：题3（自杀意图）≥ 2分 → 立即标记
- **情绪强度**：intensity ≥ 8 → 风险等级升级
- **关键词检测**：["想死", "自杀", "活着没意思", "解脱"]

**风险等级：**
```python
risk_level = {
    "low":      "正常，无风险",
    "medium":   "中等风险，需要关注",
    "high":     "高风险，建议专业帮助",
    "critical": "🚨极高风险，紧急干预"
}
```

**触发动作：**
1. 控制台警报：`[警报] 检测到高风险情况！`
2. 报告标记：`high_risk_detected: true`
3. 优先级提升：自杀相关问题优先询问
4. 建议生成：自动添加危机热线信息

### 6. 多轮对话上下文记忆

**记忆内容：**
```python
conversation_history = [
    {
        "question_title": "抑郁情绪",
        "question_id": 1,
        "answer": "最近心情确实不太好，经常感到悲伤",
        "score": 2,
        "emotion": {"emotion": "negative", "intensity": 6},
        "clarity": 0.8,
        "timestamp": 1234567890
    },
    # ... 更多对话记录
]
```

**上下文利用：**
1. **提问时**：结合之前的回答，自然过渡
   ```
   前面您提到经常感到悲伤，那睡眠方面怎么样呢？
   ```

2. **评分时**：理解上下文，避免孤立评分
   ```
   用户前面提到"对什么都没兴趣"，现在说"做事很累"，
   可能存在抑郁导致的疲劳感。
   ```

3. **追问时**：针对性追问，不重复
   ```
   您刚才说"有时候会不开心"，能说说大概多久会有一次吗？
   ```

## 📁 文件结构

```
ASR-LLM-TTS-master/
├── hamd_questions_v3.json          # 问题配置（三阶段）
├── hamd_evaluator_v3.py            # 评估器核心逻辑
├── 15.2_SenceVoice_QWen2.5_HAMD_V3.py  # 主程序集成
├── HAMD_V3_完整方案说明.md          # 本文档
└── HAMD_Results/                   # 评估报告输出目录
```

## 🔧 配置说明

### hamd_questions_v3.json

```json
{
    "index": 1,                    // 题目编号（1-16）
    "stage": 1,                    // 所属阶段（1/2/3）
    "priority": "high",            // 优先级：critical/high/medium/low
    "title": "抑郁情绪",           // 题目标题
    "intent_description": "评估用户最近的心境状态...",  // 评估目的
    "question": "您最近几天有没有觉得心情低落...",    // 原问题（备用）
    "question_variants": [         // 多种问法示例
        "最近心情怎么样？",
        "这几天感觉开心吗？",
        "..."
    ],
    "follow_up_hints": [           // 追问提示
        "能具体说说是什么时候开始的吗？",
        "..."
    ],
    "emotion_keywords": {          // 情绪关键词（辅助识别）
        "negative": ["悲伤", "难过", "..."],
        "positive": ["开心", "还好", "..."]
    },
    "scoring": {                   // 评分标准（0-4分）
        "0": "情绪良好，无抑郁表现",
        "1": "轻度不快，偶尔心情低落",
        "..."
    }
}
```

### 阶段转换配置

```python
# 在 hamd_evaluator_v3.py 中配置
# V3版本采用自动阶段转换，无需阈值判断
# 所有用户都会完成三阶段评估（除非主动停止）

# 如需自定义阶段行为，可以修改：
def get_current_question(self):
    # 阶段1完成后自动进入阶段2
    if self.current_question_index >= len(stage1_questions):
        self.current_stage = 2
        self.current_question_index = 0
        ...
```

### 追问控制

```python
# 在 hamd_evaluator_v3.py 中修改
self.max_follow_up = 2          # 每个问题最多追问2次
self.follow_up_clarity_threshold = 0.5  # 清晰度 < 0.5 触发追问

# 高风险问题特殊处理
if current_q.get('priority') == 'critical':
    if self.last_answer_clarity < 0.8:  # 提高清晰度要求
        return True
```

## 🚀 使用方法

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
export DASHSCOPE_API_KEY="your_api_key"     # 千问API密钥
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# 确保问题配置文件存在
ls hamd_questions_v3.json
```

### 2. 独立测试评估器

```python
from hamd_evaluator_v3 import HAMDEvaluatorV3

# 创建评估器（无LLM模式，使用规则fallback）
evaluator = HAMDEvaluatorV3()

# 开始评估
evaluation_id = evaluator.start_evaluation()

# 获取第一个问题
current_q = evaluator.get_current_question()
question_text = evaluator.get_question_prompt()
print(f"问题: {question_text}")

# 用户回答
user_answer = "最近心情确实不太好，经常感到悲伤"

# 处理回答
result = evaluator.process_answer(user_answer)

# 检查是否需要追问
if result.get('need_follow_up'):
    follow_up_question = evaluator.get_question_prompt(is_follow_up=True)
    print(f"追问: {follow_up_question}")
else:
    print(f"评分: {result['score']}分")

# ... 继续后续问题
```

### 3. 集成到主程序

```python
# 在 15.2_SenceVoice_QWen2.5_HAMD_V3.py 中

# 初始化评估器
hamd_evaluator = HAMDEvaluatorV3(
    questions_file="hamd_questions_v3.json",
    output_dir="./HAMD_Results/",
    chat_fn=chat_api  # 使用千问API
)

# 触发评估
if check_hamd_trigger(user_text):
    start_hamd_evaluation()

# 处理用户回答
if hamd_evaluation_active:
    process_hamd_answer(user_text)
```

### 4. 运行完整系统

```bash
# 运行主程序
python 15.2_SenceVoice_QWen2.5_HAMD_V3.py

# 系统启动后：
# 1. 说 "心理评估" 触发评估
# 2. 按提示回答问题
# 3. 系统会自动从阶段1进入阶段2、阶段3，无需确认
# 4. 支持命令：
#    - "重复" - 重复当前问题
#    - "上一题" - 返回上一题
#    - "跳过" - 跳过当前题
#    - "停止" - 结束评估
```

## 📊 评估流程示例

### 场景1：正常流程（低风险）

```
[系统] 最近心情怎么样？有没有觉得特别难过或者提不起劲？
[用户] 还可以吧，偶尔会有点低落
[系统] 我理解，这很正常。→ 评分：1分

[系统] 您晚上睡得怎么样？容易入睡吗？
[用户] 睡眠质量还不错，一般很快就能睡着
[系统] 听起来不错。→ 评分：0分

... 继续阶段1剩余3题 ...

[系统] [自动进入阶段2]
[系统] 您有没有想过伤害自己，或者觉得活着没意思？
[用户] 没有，从来没这样想过
[系统] 我理解。→ 评分：0分

... 继续评估 ...
```

### 场景2：需要追问（回答模糊）

```
[系统] 最近心情怎么样？
[用户] 不太好               # 清晰度：0.3
[系统] 能具体说说是什么感觉吗？    # 追问1
[用户] 就是有时候会难过
[系统] 这种难过的感觉经常出现吗？大概多久会有一次？  # 追问2
[用户] 几乎每天都会有，持续大半天
[系统] 我理解，这很不容易。→ 评分：3分
```

### 场景3：高风险检测

```
[系统] 您有没有想过伤害自己，或者觉得活着没意思？
[用户] 有时候会想，如果不在了可能会轻松一些
[系统] [情绪分析: critical, intensity=9, risk_level=critical]
       [警报] 检测到高风险情况！
[系统] 我非常理解您现在的感受。能再详细说说吗？  # 强制追问
[用户] 就是觉得很累，不想继续了
[系统] 我理解您现在很痛苦。→ 评分：3分

... 继续评估 ...

[系统] 评估完成。⚠️ 重要提醒：如有自杀念头，
       请立即拨打心理危机干预热线：400-161-9995
```

### 场景4：多阶段完整评估

```
=== 阶段1：初步筛查 ===
[进行5题]
总分：7分

[系统] [自动进入阶段2]

=== 阶段2：深入评估 ===
[进行6题，包括自杀风险评估]
阶段2总分：9分，累计：16分

[系统] [自动进入阶段3]

=== 阶段3：全面评估 ===
[进行5题]
阶段3总分：5分，累计：21分

[系统] 评估已经完成。测评显示您目前可能承受着较大的心理压力。
       建议您尽快寻求专业的心理咨询或医疗帮助。
       详细的评估报告已经保存好了。

[报告] 
- 总分：21/64
- 严重程度：中度抑郁
- 风险等级：high
- 情绪分析：dominant_emotion=negative, avg_intensity=6.2
- 完成阶段：1✓ 2✓ 3✓
```

## 📈 报告输出

### 报告文件结构

```json
{
  "evaluation_id": "HAMD_V3_1729123456",
  "version": "V3 (三阶段智能评估)",
  "total_score": 21,
  "max_score": 64,
  
  "severity_level": {
    "level": "中度抑郁",
    "description": "有中度抑郁症状，建议寻求专业帮助",
    "color": "orange",
    "score_range": "21/64"
  },
  
  "detailed_scores": {
    "1": {
      "score": 3,
      "analysis": "评分：3分，理由：用户提到持续情绪低落，影响日常功能",
      "scoring_criteria": "持续情绪低落，难以维持日常功能"
    },
    "...": "..."
  },
  
  "category_analysis": {
    "核心症状": {"score": 8, "max_score": 12, "percentage": 66.7, "severity": "中度"},
    "生理症状": {"score": 6, "max_score": 20, "percentage": 30.0, "severity": "轻度"},
    "...": "..."
  },
  
  "emotion_analysis": {
    "average_intensity": 6.2,
    "dominant_emotion": "negative",
    "emotion_distribution": {
      "negative": 12,
      "neutral": 3,
      "critical": 1
    }
  },
  
  "risk_assessment": {
    "suicide_risk_score": 2,
    "overall_risk_level": "high",
    "high_risk_detected": true
  },
  
  "stages_completed": {
    "stage_1": true,
    "stage_2": true,
    "stage_3": true
  },
  
  "recommendations": [
    "建议尽快寻求专业心理咨询或治疗。",
    "可以考虑心理治疗，如认知行为疗法。",
    "保持与家人朋友的联系，寻求社会支持。"
  ],
  
  "evaluation_time": {
    "start_time": "2024-10-20 14:30:00",
    "duration_minutes": 12.5
  }
}
```

## 🔍 技术细节

### 1. LLM Prompt 设计

**动态提问 Prompt:**
```python
prompt = f"""你是心理评估助手，正在进行评估的第{stage}阶段。

当前要评估的维度：{title}
评估目的：{intent_description}

参考问法示例（不要照搬）：
{question_variants}

对话上下文：
{context_summary}

当前情绪状态：{emotion}（强度{intensity}/10）

要求：
1. {adaptive_tone}（根据情绪调整语气）
2. 用日常对话的方式提问，不要生硬
3. 问题简洁，20-30字
4. 不要提及"评估"、"量表"等专业术语
5. 结合之前的对话自然过渡
6. 只生成问题本身，不要前缀和解释

请生成一个自然的提问："""
```

**评分 Prompt:**
```python
prompt = f"""你是专业的心理测量评分员。

题目：{question}
评估目的：{intent_description}

评分标准：
0分: {scoring_0}
1分: {scoring_1}
...

用户回答："{user_answer}"

对话上下文：{context}

严格要求：
1) 只输出：评分：X分，理由：[简要分析]
2) 绝对禁止提问、追问、反问
3) 绝对禁止给建议、安慰、鼓励
4) 根据评分标准严格评分

输出："""
```

### 2. 清晰度评估算法

```python
def evaluate_answer_clarity(user_answer, question_obj):
    # LLM评估（推荐）
    clarity_score = llm_evaluate_clarity(user_answer, question_obj)
    
    # 规则评估（备用）
    if llm_failed:
        length_score = min(len(user_answer) / 50, 1.0)
        vague_penalty = count_vague_words(user_answer) * 0.15
        specific_bonus = count_specific_words(user_answer) * 0.2
        
        clarity_score = length_score - vague_penalty + specific_bonus
        clarity_score = max(0, min(1, clarity_score))
    
    return clarity_score
```

### 3. 情绪检测机制

```python
def analyze_user_emotion(user_answer):
    # LLM情绪分析
    emotion_data = llm_analyze_emotion(user_answer)
    
    # 关键词增强
    critical_keywords = ["想死", "自杀", "活着没意思"]
    if any(kw in user_answer for kw in critical_keywords):
        emotion_data["emotion"] = "critical"
        emotion_data["intensity"] = max(emotion_data["intensity"], 9)
    
    # 更新全局情绪上下文
    update_emotional_context(emotion_data)
    
    # 动态调整风险等级
    if emotion_data["emotion"] == "critical":
        risk_level = "critical"
    elif emotion_data["intensity"] >= 7:
        risk_level = "high"
    ...
    
    return emotion_data
```

### 4. 阶段转换逻辑

```python
def check_stage_transition():
    # 阶段1完成 → 自动进入阶段2
    if current_stage == 1 and all_questions_answered:
        current_stage = 2
        current_question_index = 0
        print(f"[HAMD-V3] 自动进入阶段{current_stage}")
        return next_stage_first_question
    
    # 阶段2完成 → 自动进入阶段3
    if current_stage == 2 and all_questions_answered:
        current_stage = 3
        current_question_index = 0
        print(f"[HAMD-V3] 自动进入阶段{current_stage}")
        return next_stage_first_question
    
    # 阶段3完成 → 评估结束
    ...
```

## 🎨 自定义扩展

### 1. 添加新的情绪类型

```python
# 在 hamd_evaluator_v3.py 中扩展
EMOTION_TYPES = {
    "positive": "积极",
    "neutral": "中性",
    "negative": "消极",
    "critical": "严重消极",
    "anxious": "焦虑",      # 新增
    "apathetic": "冷漠"     # 新增
}

def analyze_user_emotion(user_answer):
    # 在 LLM prompt 中添加新的情绪类型
    emotion_types = list(EMOTION_TYPES.keys())
    ...
```

### 2. 调整阶段题目分配

```python
# 修改 hamd_questions_v3.json
# 将题目的 stage 字段改为目标阶段

# 例如：将"食欲减退"从阶段2移到阶段1
{
    "index": 5,
    "stage": 1,  # 改为1
    "title": "食欲减退",
    ...
}
```

### 3. 增加更多问法变体

```python
# 在 hamd_questions_v3.json 中扩展 question_variants
{
    "question_variants": [
        "最近心情怎么样？",
        "这几天感觉开心吗？",
        "您最近对平时喜欢的事情还有兴趣吗？",
        "能说说您最近的心情状态吗？",
        # 新增更多变体
        "心情有什么变化吗？",
        "最近情绪稳定吗？",
        "您觉得自己最近开心吗？"
    ]
}
```

### 4. 自定义共情回应策略

```python
# 在主程序中修改
def generate_empathy_response(user_answer, emotion):
    if emotion["emotion"] == "critical":
        # 严重情况：强调理解和支持
        return "我非常理解您现在的感受，您并不孤单。"
    elif emotion["emotion"] == "negative":
        # 消极情况：表达共情
        return "我理解，这确实很不容易。"
    elif emotion["emotion"] == "neutral":
        # 中性情况：简单确认
        return "好的，我明白了。"
    else:
        # 积极情况：正向强化
        return "听起来不错，继续保持。"
```

## ⚠️ 注意事项

### 1. 伦理和安全

- ⚠️ **本系统仅用于辅助筛查，不能替代专业诊断**
- ⚠️ **检测到高风险时，必须立即提供危机干预热线**
- ⚠️ **评估数据应严格保密，遵守隐私保护法规**
- ⚠️ **不要过度依赖LLM评分，建议人工复核高风险案例**

### 2. 技术限制

- LLM可能生成不恰当的问题（需要过滤和审查）
- 情绪检测不是100%准确（需要结合多种信号）
- 追问次数受限（避免用户疲劳）
- 网络延迟可能影响体验（建议本地缓存）

### 3. 性能优化

```python
# 1. 缓存LLM生成的问题（减少重复调用）
question_cache = {}

def get_question_with_cache(question_id, context_hash):
    cache_key = f"{question_id}_{context_hash}"
    if cache_key in question_cache:
        return question_cache[cache_key]
    
    question = generate_dynamic_question(...)
    question_cache[cache_key] = question
    return question

# 2. 异步处理（提高响应速度）
import asyncio

async def process_answer_async(user_answer):
    # 并行执行：情绪分析 + 清晰度评估 + LLM评分
    emotion, clarity, score = await asyncio.gather(
        analyze_emotion_async(user_answer),
        evaluate_clarity_async(user_answer),
        llm_score_async(user_answer)
    )
    ...

# 3. 降低LLM调用频率（使用规则优先）
def should_use_llm(question_priority):
    if question_priority == "critical":
        return True  # 高风险必须用LLM
    elif question_priority == "high":
        return random.random() < 0.8  # 80%概率用LLM
    else:
        return random.random() < 0.5  # 50%概率用LLM
```

## 📚 参考资料

- [Hamilton Depression Rating Scale (HAMD)](https://en.wikipedia.org/wiki/Hamilton_Rating_Scale_for_Depression)
- [心理测量学评分标准](https://www.example.com)
- [LLM Prompt Engineering Best Practices](https://www.example.com)
- [危机干预热线：400-161-9995](tel:400-161-9995)

## 🤝 贡献指南

欢迎贡献代码和改进建议！

```bash
# 1. Fork 本项目
# 2. 创建特性分支
git checkout -b feature/your-feature

# 3. 提交更改
git commit -m "Add: your feature description"

# 4. 推送到分支
git push origin feature/your-feature

# 5. 创建 Pull Request
```

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

---

**版本**: V3.0.0  
**更新时间**: 2024-10-20  
**作者**: HAMD-V3 开发团队

如有问题或建议，请提交 Issue 或联系开发团队。

