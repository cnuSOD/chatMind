# chatMind — 语音心理陪伴与隐式抑郁评估

一个本地运行的语音对话系统：平时是温暖的心理陪伴助手，需要时通过**一场自然的聊天**完成 HAMD（汉密尔顿抑郁量表）评估——没有"第一题、第二题"，没有问卷感，评分与分析全部在后台静默进行。

```
用户:  帮我做个评测吧
小知:  好呀。那我们就像平时一样随便聊聊，你想到什么就说什么，
       想停的时候直接说就行。最近过得怎么样？
用户:  还行吧，就是总觉得提不起劲……
小知:  嗯，提不起劲的感觉确实不好受。那晚上睡得还好吗？
       （…就这样自然聊下去，分析在后台完成…）
小知:  谢谢你愿意跟我聊这么多。听下来你最近确实挺累的，
       记得照顾好自己。之后想聊随时找我。
       （报告已静默保存到 HAMD_Results/，对话中不报分数）
```

## 快速开始

```bash
# 1. 环境
conda create -n chatAudio python=3.10
conda activate chatAudio

# 2. PyTorch（funasr 依赖；CPU版亦可，按需选择CUDA版本）
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# 3. 依赖（新系统所需的最小集合）
pip install edge-tts==6.1.17 funasr==1.1.12 webrtcvad==2.0.10 pygame==2.6.1 \
            PyAudio==0.2.14 openai transformers==4.45.2 accelerate==0.33.0

# 4. API Key（云端千问，OpenAI兼容接口）
set DASHSCOPE_API_KEY=你的_api_key        # Windows
# export DASHSCOPE_API_KEY=...            # Linux/Mac

# 5. 运行
python main.py
```

首次运行会自动下载 SenseVoice ASR 模型（已内置 hf-mirror 镜像）。

## 代码结构

```
main.py                  入口：装配各模块，启动采集循环
chatmind/
  config.py              配置加载（chatmind_config.json）与日志
  llm_client.py          千问客户端（单例、重试、fallback分离）
  speech.py              TTS播报（edge-tts，失败降级文本）与 ASR（SenseVoice）
  audio_capture.py       麦克风采集 + WebRTC VAD 断句（类封装，回调解耦）
  evaluator.py           HAMD评估器：隐式对话式提问 + 后台静默评分
  dialogue.py            对话编排：陪伴聊天 / 隐式评测 / 危机协议
chatmind_config.json     全部可调参数（模型、音色、阈值、热线、隐私开关）
hamd_questions_v3.json   16条目题库（意图描述/参考问法/追问提示/0-4评分细则）
HAMD_Results/            评估报告（JSON，静默保存）
logs/chatmind.log        运行日志
```

## 工作流程

### 日常陪伴

直接说话即可。助手人设为温暖专业的倾听者：共情优先、口语化短回应、不诊断不荐药。

### 隐式评测

1. **触发**（用户主动发起，这是知情的前提）：说"帮我做个评测 / 聊聊我最近的状态 / 心理评估"等；
2. **自然开场**：一句透明告知（"像平时一样随便聊聊，想停直接说"）+ 第一个生活化话题；
3. **对话即评估**：每轮回应由 LLM 以"朋友接话"方式生成——先接住上一句，再自然聊到下一个维度（心情、睡眠、兴趣、精力、食欲……16 个 HAMD 条目）。提示词硬约束：禁止出现评估/测试/题目/量表/打分等字眼，一段话只许一个问句；
4. **后台静默分析**：每轮回答经单次结构化 LLM 调用同时得到 评分(0-4)/清晰度/情绪/强度；回答含糊会自然追问（最多2次），追问的补充**累积**进当题评分；
5. **三阶段门控**：初筛 5 个话题累计分 <5 → 聊天自然收尾（约3分钟）；≥5 或出现高风险信号 → 继续深入 6 个话题（含自杀风险条目）；累计 ≥10 → 全面 5 个话题；
6. **退出**：随时说"先这样吧 / 换个话题"即可，已聊的部分仍会保存；
7. **收尾**：LLM 生成观察式反馈（"听下来你最近确实挺累的"），不报分数、不贴标签；中重度自然建议寻求专业帮助。报告 JSON 静默保存。

### 危机协议（所有模式全程生效）

检测到"想死/自杀/不想活"等表达 → 立即播报关怀话术 + 全国心理援助热线 **12356**，不转移话题、不中断对话。评测中每次只播报一次，随后继续陪伴。

### 隐私

录音 wav 用完即删，TTS 临时文件播完即删（`chatmind_config.json` 可关）；对话与报告只存本机。

## 配置要点（chatmind_config.json）

| 配置项 | 说明 |
|---|---|
| `llm.model` | 默认 `qwen-turbo`，可换 `qwen-plus` 等 |
| `tts.voice` | edge-tts 音色，默认 `zh-CN-XiaoyiNeural` |
| `audio.answer_pause_seconds` | 评测中等待回答的静音阈值（默认3.5s，给思考时间） |
| `assessment.stage_thresholds` | 阶段门控阈值 `{"1": 5, "2": 10}` |
| `safety.crisis_hotline` / `crisis_keywords` | 危机热线与触发词 |
| `privacy.*` | 录音/TTS文件用完即删开关 |

## 报告字段（HAMD_Results/*.json）

总分与严重程度（≤7 正常 / 8-17 轻度 / 18-23 中度 / ≥24 重度）、逐条目评分与依据、分项维度分析（核心/认知/生理/心理症状）、情绪轨迹、自杀风险评分、结构化建议卡片（title/description/frequency）、对话轮数与时长、是否提前结束（`early_terminated`，提前结束的总分仅供初筛参考）。

> ⚠️ 方法学说明：本实现为 16 条目、每条 0-4 分的 HAMD 变体，严重程度分界沿用 HAMD-17 惯例，严格的临床效度需另行标定；结果仅供参考，不构成医学诊断。

## 各条目评分细则（0-4分）

<details>
<summary>展开 16 条目评分标准</summary>

**阶段1 初筛**：1 抑郁情绪 / 4 睡眠障碍 / 6 兴趣丧失 / 9 焦虑 / 13 无力感
**阶段2 深入**：3 自杀意图（critical）/ 2 有罪感 / 8 精神运动性迟缓 / 10 自我评价 / 5 食欲减退 / 11 情感平淡
**阶段3 全面**：7 体重减轻 / 14 积极性 / 12 性兴趣减退 / 15 感知能力 / 16 其他症状

每条目的 0-4 分锚点描述见 `hamd_questions_v3.json` 的 `scoring` 字段（评分提示词直接引用该处，文档与代码单一来源）。自杀意图条目 ≥2 分即触发高风险协议。

</details>

## 旧版脚本与组件示例

| 文件 | 说明 |
|---|---|
| `0~14_*.py` | 各组件单独验证脚本（ASR/TTS/LLM/视觉等示例） |
| `cosyvoice/`、`webui.py` 等 | 上游 CosyVoice 本地TTS能力（新系统未使用） |
| `优化说明_代码与提示词工程.md` | 历次优化的完整变更记录 |

（抑郁评测旧版入口 `15.0/15.1/15.2_*.py` 及 `hamd_evaluator(_v3).py` 已删除——功能已并入 `chatmind/` 包，可从 git 历史找回。）

## 致谢

ASR: [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) · TTS: [edge-tts](https://github.com/rany2/edge-tts) · LLM: 通义千问 · 量表: Hamilton Depression Rating Scale
