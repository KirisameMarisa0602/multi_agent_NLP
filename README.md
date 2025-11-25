# 多智能体学术写作优化系统（Academic Multi-Agent WritingOptimizer）

> 本 README 在原说明基础上，**结合你目前的本地环境与本地 Qwen 模型配置**进行了完整重写：
> - 统一整合 `README.md` / Web 说明 / 项目结构文档；
> - 补充 **Python 3.11 + RTX 4060 + 本地 Qwen2/Qwen GPTQ 模型** 的推荐环境配置；
> - 给出从零重建 venv、安装依赖、加载本地模型、运行 CLI / Web 的一步到位指南。
>
> 若你只是想快速「能跑」，直接看：**[5. 环境与依赖] → [6. 安装与启动] → [7. 本地 Qwen 模型配置] → [8. 快速示例]**。

---
## 目录
1. 项目概览与目标
2. 核心特性速览
3. 代码结构与模块职责
4. 流程与总体架构示意
5. 环境与依赖（Python 3.11 + GPU + 本地模型）
6. 安装与启动（从零搭建 venv）
7. 本地 Qwen 模型配置（1.8B / 14B GPTQ）
8. CLI 使用模式：`demo` / `synthesize` / `eval` / `distill`
9. 多智能体协作机制（Agent A / Agent B）
10. 工具调用与回退策略（Search / Python REPL / 文件读写）
11. 向量记忆与检索（FAISS / 简易回退）
12. 学术质量评估体系（9 维指标）
13. 数据管线：合成 → 蒸馏 → LoRA/QLoRA 微调
14. Web 图形界面与 API 概览
15. 报告与可视化（HTML / JSON / 文本）
16. 测试与质量保障
17. 常见问题 FAQ（含本地模型 & GPU 配置）
18. 扩展与维护建议
19. 路线图
20. 许可与引用

---
## 1. 项目概览与目标
本项目实现一个 **多智能体协作闭环的学术写作优化系统**：

- **Agent A（Optimizer / 学生模型）**：对原始草稿进行结构化、学术化、逻辑与表达优化，可使用本地 Qwen 学生模型（如 Qwen2 1.8B）。
- **Agent B（Reviewer / 教师模型）**：基于高质量 LLM（如 DeepSeek）进行严谨评审与多维评分，并给出下一轮改进建议。
- 多轮交替迭代后，输出：
  - 最终优化文本；
  - 每轮优化与审稿记录（diff、评分、工具调用、记忆检索等）；
  - 完整 HTML/JSON 报告，便于分析与可视化。

系统同时支持：

- **数据合成**：自动生成带 teacher signal 的高质量学术文本样本；
- **蒸馏样本构造**：从合成日志抽取指令-输出对；
- **LoRA / QLoRA 微调**：基于蒸馏数据对本地 Qwen 学生模型进行微调，提高其学术写作能力；
- **本地部署**：在完全离线或内网环境下，仅使用本地模型也可跑通完整流程（回退到 DummyLLM / 简易向量检索）。

适用场景：论文段落打磨、方法/结果/讨论优化、学术表达风格统一、教学演示、多智能体策略实验、蒸馏数据构建、小模型强化学术写作能力等。

---
## 2. 核心特性速览

- ✅ **双 Agent 多轮协作**：Agent A 负责改写优化，Agent B 负责评审与打分，迭代收敛到高质量版本。
- ✅ **学术质量 9 维指标体系**：规范性、证据完整性、创新度、流畅度、结构完整性等，可对比优化前后改进幅度。
- ✅ **灵活的 LLM 初始化链**：`LangChain ChatOpenAI → HTTPFallbackChat → DummyLLM`，在缺 API Key 或依赖时自动回退。
- ✅ **本地学生模型支持**：通过 `hf_student_llm.py` 加载本地 Qwen（推荐 Qwen2 1.8B Chat），可与远程教师模型混合使用（Hybrid 模式）。
- ✅ **工具层 ReAct 风格推理**：内置检索（SerpAPI）、Python REPL、文件读写工具，可按内容自动触发或关闭。
- ✅ **向量记忆与检索**：默认使用 FAISS + Embeddings，若 FAISS 或 Embeddings 不可用，则回退为基于词集重叠的简易检索。
- ✅ **长文本智能分段**：支持按句子边界 + 重叠进行切块，对段落级别优化再汇总。
- ✅ **数据闭环**：`synthesize`（合成）→ `distill`（蒸馏）→ `lora_distill.py`（微调）→ `hf_student_llm.py`（加载学生模型）。
- ✅ **丰富报告与可视化**：HTML 报告展示每轮文本变化、评分趋势、指标对比、工具调用日志。
- ✅ **Web 实时界面**：Flask + SocketIO 提供任务提交、进度展示、结果下载等能力。
- ✅ **健壮回退设计**：在缺少 LLM、SerpAPI、FAISS 等情况下一律降级为占位/简易策略，流程仍可正常演示。

---
## 3. 代码结构与模块职责

项目根目录关键文件/目录：

- `multi_agent_nlp_project.py`：
  - CLI 入口，包含 `demo` / `synthesize` / `eval` / `distill` 四大模式；
  - 负责主 LLM 初始化（DeepSeek/OpenAI/DummyLLM）、工具集构建、向量存储与 Memory 管理、多智能体协作流程；
  - 生成 HTML/JSON 报告。
- `hf_student_llm.py`：
  - 本地学生模型封装，使用 HuggingFace `AutoTokenizer` / `AutoModelForCausalLM` 加载任意兼容模型（推荐 Qwen2 1.8B Chat 本地目录）；
  - 支持可选 LoRA 适配器（PEFT）；
  - 在 `FORCE_STUDENT_STUB=1` 或缺少 torch/transformers 时自动回退到轻量 stub 实现。
- `metrics.py`：
  - 学术质量 9 维指标计算与综合评分逻辑；
  - 纯 Python 实现，无外部依赖，适配中英文。
- `demo_metrics.py`：
  - 指标体系使用示例脚本（单段/对比/可视化）。
- `lora_distill.py`：
  - LoRA / QLoRA 微调脚本；
  - 从蒸馏数据 JSONL 读取指令-输出对，使用 `transformers + peft + accelerate (+ bitsandbytes)` 进行训练；
  - 输出 LoRA 适配器目录供 `hf_student_llm.py` 加载。
- `web_interface/`：
  - `app.py`：Flask + SocketIO 后端，提供任务 API、进度推送、报告下载；
  - `start_web.py`：Web 服务启动脚本；
  - `index.html` + `static/js/app.js` + `static/css/styles.css`：前端界面与交互逻辑；
  - `uploads/`：上传文件暂存目录。
- `data/`：示例数据与训练数据目录。
  - `seeds.txt`：合成任务的种子句子；
  - `distill_pairs.jsonl`：示例蒸馏对；
  - `synth_*.jsonl`：示例合成日志文件。
- `tests/test_flow.py`：
  - 用于验证文本分段、需求解析等核心逻辑的单元测试。
- `requirements.txt`：
  - 统一的依赖清单（核心 + Web + 微调 + 评估），已适配 Python 3.11 与本地 Qwen 模型使用场景。

---
## 4. 流程与总体架构示意

```text
┌──────────────────────────────────────────────┐
│ multi_agent_nlp_project.py                  │
│  ├─ LLM 初始化链 (LangChain ChatOpenAI → HTTPFallbackChat → DummyLLM)
│  ├─ 工具集合 (Search / Python REPL / File IO)                         
│  ├─ 向量记忆 (FAISS 或 SimpleVectorStore)                            
│  ├─ DualAgentAcademicSystem                                          
│  │   ├─ Agent A Prompt + Chain                                       
│  │   ├─ Agent B Prompt + Chain                                       
│  │   ├─ 协作日志记录 (scores/diff/tools/memory)                     
│  │   ├─ evaluate / synthesize / distill                              
│  │   └─ HTML 报告生成                                                
│  ├─ 长文本分段 + 文件优化                                           
│  └─ CLI 入口 (demo / synthesize / eval / distill)                   
├──────────────────────────────────────────────┤
│ hf_student_llm.py (本地学生模型 + LoRA 适配)                          │
├──────────────────────────────────────────────┤
│ lora_distill.py (LoRA / QLoRA 微调脚本)                               │
├──────────────────────────────────────────────┤
│ metrics.py (9 维学术质量指标) + demo_metrics.py (演示)                │
├──────────────────────────────────────────────┤
│ web_interface/ (Web API + 前端界面 + 实时进度)                        │
└──────────────────────────────────────────────┘
```

---
## 5. 环境与依赖（Python 3.11 + RTX 4060 + 本地模型）

### 5.1 推荐基础环境

- 操作系统：Windows 10/11 x64
- Python：**3.11.x**（建议单独为本项目创建虚拟环境）
- 显卡：NVIDIA RTX 4060（或同级别，建议 >= 8GB 显存）
- CUDA：建议使用 PyTorch 官方 cu121 对应驱动（安装官方最新驱动即可，一般可自动适配）。

### 5.2 `requirements.txt`（已更新）

当前仓库中的 `requirements.txt` 已整合所有核心依赖，主要包括：

- **LangChain 相关**：`langchain*`, `google-search-results`
- **向量检索**：`faiss-cpu`, `numpy`
- **工具与通用依赖**：`tiktoken`, `python-dotenv`, `requests`, `safetensors`
- **HuggingFace & 微调**：`transformers==4.46.0`, `huggingface-hub<1.0`, `peft`, `datasets`, `accelerate`, `bitsandbytes`, `optimum`
- **Web 框架**：`Flask`, `Flask-SocketIO`, `Flask-CORS`, `python-socketio`, `python-engineio`, `Werkzeug`
- **测试**：`pytest`

> 注意：**`torch` / `torchvision` / `torchaudio` 不在 `requirements.txt` 中固定**，请根据你本机 GPU/CUDA 环境单独安装（见下文安装步骤）。

---
## 6. 安装与启动（从零搭建 Python 3.11 + GPU venv）

以下命令假设你已经安装了 Python 3.11，比如 `python3.11` 或 `C:\Python311\python.exe` 可用。

### 6.1 创建新的 virtualenv

```powershell
cd D:\Projects\NLP\multi_agent_NLP

# 使用 Python 3.11 创建虚拟环境
python3.11 -m venv .venv

# 激活环境
.\.venv\Scripts\activate
```

### 6.2 安装 GPU 版 PyTorch（RTX 4060 + CUDA 12.1）

```powershell
# 在已激活的 .venv 中执行
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 验证 CUDA 是否可用
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"
```

若输出 `cuda available: True` 且 `device count >= 1`，说明 4060 已被 PyTorch 识别。

### 6.3 安装项目依赖

```powershell
cd D:\Projects\NLP\multi_agent_NLP
pip install -r requirements.txt
```

> 若 `bitsandbytes` 或 `faiss-cpu` 安装失败，可暂时忽略：项目已内置回退逻辑，不会影响基础功能。

### 6.4 准备 `.env` 文件

```powershell
cd D:\Projects\NLP\multi_agent_NLP
copy .env.example .env
```

编辑 `.env`（可用记事本/VSCode 打开），根据实际情况设置：

```ini
OPENAI_API_KEY=        # 若使用 DeepSeek/其他 OpenAI 兼容接口则填写，否则留空
OPENAI_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-reasoner
SERPAPI_API_KEY=       # 若使用 SerpAPI 则填写，否则留空
EMBED_MODEL_NAME=text-embedding-3-small
ENABLE_INTERACTIVE=0

# 学生模型（本地 Qwen）示例，详见下一节
STUDENT_BASE_MODEL=D:/Projects/NLP/models/Qwen1.5-1.8B-Chat
STUDENT_LORA_DIR=
STUDENT_MAX_NEW_TOKENS=256
FORCE_STUDENT_STUB=0
```

### 6.5 快速启动 CLI 示例

```powershell
cd D:\Projects\NLP\multi_agent_NLP
.\.venv\Scripts\activate

python multi_agent_nlp_project.py demo ^
  --rounds 2 ^
  --text "这是一个需要提升学术表达与逻辑清晰度的段落。" ^
  --requirements "学术表达提升;逻辑结构优化" ^
  --html-report demo.html
```

生成的 `demo.html` 即为完整 HTML 报告，可在浏览器中打开查看。

### 6.6 启动 Web 界面

```powershell
cd D:\Projects\NLP\multi_agent_NLP
.\.venv\Scripts\activate

cd web_interface
python start_web.py
```

然后在浏览器访问：`http://localhost:5000`。

---
## 7. 本地 Qwen 模型配置（重点）

本项目已针对本地 Qwen 模型做了专门封装，支持：

- 从 **HuggingFace 仓库完整克隆的 Qwen2 / Qwen1.5 模型目录**；
- 从本地权重目录（含 `config.json` / `tokenizer.json` / 权重文件）加载；
- 针对 GPTQ 量化模型（如 `Qwen1.5-14B-Chat-GPTQ-Int4`）借助 `optimum` 进行加载（需要额外显存与配置）。

### 7.1 推荐：本地 Qwen2 1.8B Chat 模型

你可以在一台能访问 HuggingFace 的机器上运行：

```bash
cd D:/Projects/NLP/models
git lfs install
git clone https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat
```

然后将该目录完整拷贝到当前开发机，例如：

```text
D:\Projects\NLP\models\Qwen1.5-1.8B-Chat
```

在 `.env` 中配置：

```ini
STUDENT_BASE_MODEL=D:/Projects/NLP/models/Qwen1.5-1.8B-Chat
STUDENT_LORA_DIR=
STUDENT_MAX_NEW_TOKENS=256
FORCE_STUDENT_STUB=0
```

此时 `hf_student_llm.HFChatLLM` 会使用：

```python
AutoTokenizer.from_pretrained("D:/Projects/NLP/models/Qwen1.5-1.8B-Chat")
AutoModelForCausalLM.from_pretrained("D:/Projects/NLP/models/Qwen1.5-1.8B-Chat", torch_dtype=..., device_map="auto")
```

在你已经安装 GPU 版 PyTorch 的前提下，模型会被自动加载到 RTX 4060 上。

### 7.2 可选：本地 Qwen1.5-14B-Chat GPTQ 模型

若你有类似目录：

```text
D:\Projects\NLP\models\Qwen1.5-14B-Chat-GPTQ-Int4
  ├─ config.json
  ├─ tokenizer.json
  ├─ model-00001-of-00003.safetensors
  ├─ model-00002-of-00003.safetensors
  ├─ model-00003-of-00003.safetensors
  └─ ...
```

则需要：

1. 在 venv 中安装 `optimum`（requirements 已包含，也可单独指定更高版本）：

   ```powershell
   pip install "optimum>=1.20.0"
   ```

2. 写一个单独脚本测试加载（示意）：

   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   import torch

   MODEL_DIR = r"D:\Projects\NLP\models\Qwen1.5-14B-Chat-GPTQ-Int4"

   tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
   model = AutoModelForCausalLM.from_pretrained(
       MODEL_DIR,
       torch_dtype=torch.float16,
       device_map="auto",
   )
   ```

3. 成功后可将 `.env` 中 `STUDENT_BASE_MODEL` 指向该目录，但请注意：
   - 14B GPTQ 对显存要求远高于 1.8B，RTX 4060 需要合理设置 `max_new_tokens` 以及 `device_map`；
   - 建议先在 1.8B Qwen2 模型上跑通整个系统再尝试 14B GPTQ。

### 7.3 Stub 学生模型（CI / 无模型时）

若设置：

```ini
FORCE_STUDENT_STUB=1
```

则 `hf_student_llm` 会启用一个轻量 stub：

- 不会尝试加载任何本地模型权重；
- `.invoke()` 返回带有占位提示的短文本，用于测试流程或 CI；
- 适合在没有本地模型、也没有 GPU 的情况下快速验证多智能体与报告生成逻辑。

---
## 8. CLI 使用模式：demo / synthesize / eval / distill

### 8.1 `demo`：单文本/长文本多轮优化

```powershell
python multi_agent_nlp_project.py demo ^
  --rounds 2 ^
  --text "这是一个需要提升学术表达与逻辑清晰度的段落。" ^
  --requirements "学术表达提升;逻辑结构优化" ^
  --html-report demo.html
```

常用参数：

- `--text` / `--text-file`：输入文本或文本文件；
- `--rounds`：多轮协作轮数；
- `--chunk-size` / `--chunk-overlap` / `--max-chunks`：长文本分段与重叠控制；
- `--no-tools` / `--no-memory`：禁用工具调用或记忆功能用于消融实验；
- `--html-report`：输出 HTML 报告路径；
- `--hybrid` / `--student-base-model` / `--student-lora-dir`：启用本地学生模型混合模式（见下一节）。

### 8.2 `synthesize`：数据合成

```powershell
python multi_agent_nlp_project.py synthesize ^
  --rounds 3 ^
  --seeds-file data\seeds.txt ^
  --out data\synth.jsonl
```

从 `seeds.txt` 中读取种子句子，为每个种子运行多轮 Agent 协作，生成包含 teacher signal 的合成数据集。

### 8.3 `distill`：蒸馏对构造

```powershell
python multi_agent_nlp_project.py distill ^
  --distill-src data\synth.jsonl ^
  --distill-out data\distill_pairs.jsonl
```

从合成日志中抽取指令-输出对，并保留评分等元数据，供 LoRA/QLoRA 微调使用。

### 8.4 `eval`：评估模式

```powershell
python multi_agent_nlp_project.py eval ^
  --rounds 2 ^
  --report data\eval_report.json ^
  --html-report eval_report.html
```

对多组样本运行多轮协作与指标评估，输出 JSON 与 HTML 报告。

---
## 9. 多智能体协作机制（Agent A / Agent B）

- **Agent A（Optimizer / 学生）**：
  - 负责生成优化后的文本；
  - Prompt 中包含原始文本、需求列表、上一轮评分结果与记忆检索摘要；
  - 在 Hybrid 模式下由本地 Qwen 学生模型驱动。

- **Agent B（Reviewer / 教师）**：
  - 负责审稿与打分；
  - 给出结构化反馈与 JSON 格式评分（quality / rigor / logic / novelty 等）；
  - 可使用远程 DeepSeek 等高质量模型或 DummyLLM 占位。

每轮流程：

1. Agent A 根据当前文本与需求/记忆生成候选优化稿；
2. Agent B 根据前后文本差异、需求与评分体系进行评审与打分；
3. 系统记录本轮 `optimized_text`、`agent_b_feedback`、`scores`、`diff`、`tool_observations`、`timestamp`；
4. 进入下一轮或输出最终结果。

---
## 10. 工具调用与回退策略（Search / Python REPL / 文件读写）

- 工具集合在 `multi_agent_nlp_project.py` 中初始化：
  - `SerpAPIWrapper`：需要 `SERPAPI_API_KEY`，用于网络搜索；
  - `PythonREPL`：执行小段 Python 代码；
  - 文件读写工具：基于简单 `open` 封装的读/写函数；
- 触发条件：
  - 文本中出现“search/检索/事实/最新/引用”等关键词 → 搜索；
  - 出现 `python: ```python ... ``` ` 代码块 → PythonREPL；
- 回退逻辑：
  - 未配置 SerpAPI：搜索工具返回占位提示，不中断流程；
  - 缺少 LangChain 相关依赖：自动使用 stub 工具类与简化 PythonREPL，实现接口兼容；
  - 文件读写工具始终使用 Python 标准库，无额外依赖。

---
## 11. 向量记忆与检索（FAISS / 简易回退）

- 默认使用 `OpenAIEmbeddings`（或 DummyEmbeddings）+ `FAISS`：
  - 文本被嵌入为向量，存入向量索引；
  - 查询时通过余弦/欧氏距离检索 Top-k 相似片段；
- 若 faiss-cpu 或相关依赖安装失败：
  - 自动切换为 `SimpleVectorStore`，使用基于分词（中英文）后的 Jaccard 相似度排序；
- MemoryManager 提供：
  - `add_memory(text, metadata)`：写入带时间戳与命名空间的记忆；
  - `recall(query, k)`：检索最相关的若干记忆片段，加入下一轮 Prompt。

---
## 12. 学术质量评估体系（9 维指标）

详见 `metrics.py`，包括但不限于：

- 学术规范性
- 引用与证据完整性
- 创新度
- 语言流畅度
- 句子平衡
- 论证强度
- 表达多样性
- 结构完整性
- 时态一致性

支持：

- 单文本打分；
- 优化前后对比；
- 加权综合评分；
- 提升率（improvement_rate）计算。

`demo_metrics.py` 提供完整示例。

---
## 13. 数据管线：合成 → 蒸馏 → LoRA/QLoRA 微调

典型闭环流程：

1. **合成（synthesize）**：基于种子句子与多轮协作，生成高质量 teacher_signal；
2. **蒸馏（distill）**：从合成日志抽取指令-输出对（JSONL 格式），保留评分等元信息；
3. **微调（lora_distill.py）**：
   - 以 Qwen 基座模型（如 `Qwen/Qwen1.5-1.8B-Chat`）与蒸馏对为输入；
   - 使用 LoRA/QLoRA 策略训练；
   - 输出 LoRA 适配器目录（如 `runs/qwen-mini-lora`）。
4. **学生模型加载（hf_student_llm.py）**：
   - 在 Hybrid 模式下，Agent A 使用「基座 + LoRA」，Agent B 使用远程教师模型；
   - 系统提供「学生 vs 教师」的真实协作对比场景。

---
## 14. Web 图形界面与 API 概览

Web 后端位于 `web_interface/app.py`：

- 主要 API：

```text
POST /api/optimize/text      # 文本优化
POST /api/optimize/file      # 文件上传优化
POST /api/synthesize         # 数据合成
POST /api/evaluate           # 批量评估
POST /api/distill            # 蒸馏对构造
GET  /api/task/<task_id>     # 查询任务状态
GET  /api/download/<task_id>/text|html|json  # 下载结果
POST /api/config             # 动态更新本次会话的模型与 Key
```

- 前端位于 `web_interface/index.html` 与 `static/js/app.js`：
  - 多标签页展示不同功能（文本/文件优化、合成、评估、蒸馏、配置）；
  - 使用 SocketIO 订阅 `task_update` / `round_update` 实时更新进度；
  - 支持 HTML 报告与原始 JSON 日志下载。

---
## 15. 报告与可视化（HTML / JSON / 文本）

- **HTML 报告**：
  - 最终优化文本；
  - 每轮协作摘要与评分；
  - Diff 高亮（新增/删除/修改）；
  - 学术质量指标表格与改进幅度；
  - 工具调用与记忆检索记录。
- **JSON 报告**：
  - 完整结构化日志，便于后处理与统计分析。
- **文本导出**：
  - 通过 `--out-text-file` 或 Web 下载接口，导出最终优化结果文本文件。

---
## 16. 测试与质量保障

- `pytest` 集成：
  - 运行 `pytest` 可执行基本单元测试（如文本分段、需求解析等）。

```powershell
cd D:\Projects\NLP\multi_agent_NLP
.\.venv\Scripts\activate
pytest
```

- 回退设计：
  - 缺少 LLM / OpenAI Key → 使用 DummyLLM 占位输出；
  - 缺少 SerpAPI → 搜索工具返回占位字符串；
  - 缺少 FAISS → 使用 SimpleVectorStore；
  - 缺少 LangChain 相关包 → 使用 stub 工具类与 PythonREPL stub。

---
## 17. 常见问题 FAQ（含本地模型 & GPU）

**Q1：没有 OPENAI_API_KEY 能运行吗？**  
A1：可以。系统会使用 DummyLLM，生成占位输出，但多智能体流程与报告仍然会正常生成，适合教学与离线演示。

**Q2：如何确认本地 Qwen2 1.8B 已经在 GPU 上运行？**  
A2：在 `.venv` 中运行：

```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python test_local_qwen.py   # 观察输出来确认 '使用设备: cuda'
```

或在 `hf_student_llm.HFChatLLM.__init__` 中临时打印 `self.device`。

**Q3：安装 PyTorch 时提示 `No matching distribution found for torch`？**  
A3：通常是 Python 版本过新（如 3.13）。请使用 Python 3.11 创建虚拟环境，并使用 PyTorch 官方指定的 `--index-url` 安装对应 CUDA 版本。

**Q4：FAISS 安装失败怎么办？**  
A4：可暂时忽略，项目会自动回退到基于词集相似度的 SimpleVectorStore，功能可用但检索效果略下降。

**Q5：本地 Qwen 模型目录中没有 `.py` 源码文件，会影响加载吗？**  
A5：对于 Qwen2 系列，`transformers>=4.37` 已内置 `qwen2` 支持，无需额外 `.py` 源码文件；只要 `config.json` 中的 `model_type` 为 `qwen2`，即可通过 `AutoModelForCausalLM.from_pretrained(本地目录)` 直接加载。

**Q6：14B GPTQ 模型加载时报 `requires optimum`？**  
A6：请在 venv 中执行 `pip install "optimum>=1.20.0"`，然后按前文示例脚本加载；同时注意 14B GPTQ 对显存要求更高，需适当控制 `max_new_tokens` 与 `device_map`。

**Q7：Web 端“连接失败”或无法访问？**  
A7：检查以下几点：
- 确认 `start_web.py` 已经在运行且端口未被占用；
- 若开启了防火墙，确保 5000 端口允许本地访问；
- 在浏览器中手动访问 `http://localhost:5000` 测试。

---
## 18. 扩展与维护建议

- 新增学术指标：在 `metrics.py` 中添加新的指标函数，并在综合评分处加入权重配置；
- 新增工具：在 `multi_agent_nlp_project.py` 中扩展 TOOLS 列表，并在规划逻辑中增加触发规则；
- 多模型混合：通过修改 `hf_student_llm.py` 与 CLI 参数，实现不同任务/Agent 使用不同模型（例如 Agent A 本地学生、Agent B 远程教师）；
- 性能优化：对长文本可结合分段策略与缓存机制，减少重复调用；
- 安全/对齐：可在 Agent Prompt 中加入安全与对齐要求，或在输出后增加过滤模块。

---
## 19. 路线图

| 阶段 | 计划 | 价值 |
|------|------|------|
| 短期 | 补充更多中英文完整示例与测试用例 | 提升稳定性与多语言覆盖 |
| 中期 | 引入显著性检验与因果链分析工具 | 提升学术严谨性与可解释性 |
| 中期 | Web 端支持批量文件处理与进度汇总 | 提升生产力与易用性 |
| 长期 | 引入可学习的策略调度（RL 选择工具/记忆） | 提升多智能体自适应能力 |
| 长期 | 增加模型对齐与安全过滤模块 | 满足生产与合规要求 |

---
## 20. 许可与引用

- 本项目采用开放许可（请查阅根目录 `LICENSE`，如未包含可根据需要补充）。

若在论文或项目中引用本仓库，建议使用类似描述：

```text
作者. 多智能体学术写作优化框架: 协作、评审与蒸馏闭环. 项目仓库, 2025.
```

---

> 到此，你已经拥有：
> - 一套适配 **Python 3.11 + RTX 4060 + 本地 Qwen2 模型** 的完整环境方案；
> - 从零重建 venv → 安装依赖 → 加载本地模型 → 运行 CLI/Web → 扩展与调试 的全链路说明。
> 
> 后续如果你增加新的本地模型或部署到其他机器，只需：
> - 拷贝模型目录（如 `Qwen1.5-1.8B-Chat`）；
> - 使用同样的 PyTorch CUDA 安装命令；
> - 更新 `.env` 中的 `STUDENT_BASE_MODEL` 路径，即可快速迁移。
