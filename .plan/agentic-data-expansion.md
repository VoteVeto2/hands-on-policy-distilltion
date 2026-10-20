# 扩充 agentic 训练数据：调研与计划

> 日期：2026-10-19
> 背景：上一轮在 A100 上把 Qwen3-8B 蒸馏到 Qwen3-1.7B，共 5 个阶段，结果见 [`results/trl/a100/README.md`](../results/trl/a100/README.md)。那一轮的数据只有计算器、SQL、JSON 查询；带奖励的阶段几乎没有学习信号。
> 本文分三步：先看现有数据和真实世界 agent 任务差在哪，再列出可用的数据集和环境，最后给出在本仓库里的落地顺序。
> 数字都来自论文、数据卡或排行榜，附了链接。标 `[未核实]` 的是调研时没能和原文对上的。
>
> **更新（2026-10-20）：已决定整套换到 Qwen3.5**（学生 Qwen3.5-2B，教师 Qwen3.5-9B）。代码已迁移，stage 1–4 已在 A100 上重跑完，结果见 [`results/trl/a100-qwen3.5/README.md`](../results/trl/a100-qwen3.5/README.md)，摘要见 §8.5。迁移细节、新的教师 / 学生数字和对本计划的影响见 [§8](#8-换到-qwen35-之后2026-10-20)。§0–§7 写于换代之前，其中 Qwen3 的数字保留作参照，阶段 A–E 里的教师、学生一律换成 Qwen3.5-9B / 2B。

## 0. 结论速览

1. **8B 教师能教给 1.7B 的，主要是多轮、有状态的工具交互，不是单轮函数调用。**
   BFCL v4（FC 模式）上 Qwen3-1.7B 对 Qwen3-8B 的分数：

   | 类别 | 1.7B | 8B | 差距 |
   |---|---|---|---|
   | 单轮 non-live | 82.9 | 87.6 | 约 5 分 |
   | 单轮 live | 74.6 | 80.5 | 约 6 分 |
   | **多轮** | **11.0** | **41.8** | **约 30 分** |
   | 网页搜索 | 2.5 | 12.0 | |
   | 记忆 | 6.0 | 14.6 | |

   所以扩数据的重点是多轮环境：能执行、有状态、能用代码检查最终状态。
2. **首选下面四类。** 它们都不需要 Docker、能在 Colab 上跑，并且有 Qwen3 小模型的公开结果。
   - **EnvScaler**：191 个 Python 类环境，约 7K 个场景，按最终状态的 checklist 给部分得分。Qwen3-1.7B 训练后，BFCL-v3 多轮从 9.75 提到 23.0。
   - **AgentWorldModel-1K（AWM）**：1,000 个 SQLite 环境，1 万道题，约 3.5 万个工具。有纯代码判分器，OpenEnv 已经封装好。
   - **τ²-bench telecom**：官方给了训练 / 测试划分，`full` 集有 2,285 题，全部用代码判分。solo 模式不需要用户模拟器。
   - **检索问答（Search-R1 那一套）+ ALFWorld**：SAGE-OPD 用的正是 Qwen3-8B → Qwen3-1.7B，可以直接对标：

     | 任务 | 学生 | OPD 后 | 教师 |
     |---|---|---|---|
     | 检索问答，平均 EM | 18.3 | 30.1 | 35.9 |
     | ALFWorld unseen 成功率 | 26.1 | 61.9 | 67.2 |
3. **暂时不做两类。**
   - **SWE / 终端**：Qwen3-8B 在 Terminal-Bench 2.0 上 0.0%，在 SWE-bench Verified 上 0.7%，教师自己都不会。而且每道题要一个 Docker 镜像，Colab 跑不了。
   - **开放网页的 deep research**：要付费搜索 API 和 LLM 评判，题目对 8B 也太难。
4. **教师决定上限，换教师必须同词表。**
   - Qwen3.5 / 3.6 / 3.8 的词表是 248,320，Qwen3 是 151,936，所以前者不能直接当 Qwen3 学生的教师。
   - ~~同词表、更强的选择有：Qwen3-14B；Qwen3-30B-A3B-Thinking-2507 或 Qwen3-32B（要量化）。~~ 已改为整套换到 Qwen3.5（§8）。Qwen3.5-27B、35B-A3B 和 Qwen3.6 的 tokenizer 与 Qwen3.5-2B 完全相同，以后换更强的教师不用再动学生。
5. **扩数据要和修 GRPO 无信号一起做：**
   - 选难度可调、能给部分得分的任务；
   - 按当前学生的通过率筛题，只留 0 < 通过率 < 1 的；
   - 提高 RL 阶段的采样温度；
   - 参考 ATOD，把教师系数从 1 退火到 0.1，奖励系数从 1 升到 2。

## 1. 现有数据缺什么

现有数据包括：GSM8K、计算器、SQLite 查询、JSON 查询，以及它们的组合，最多 6 轮。全部是程序生成的，答案写在 `\boxed{}` 里。

这些数据确实教会了学生按教师的方式用这 3 个工具。但真实世界的 agent 任务还要求下面这些能力：

| 真实世界的要求 | 现在的数据 | 可以补上的数据 / 环境 |
|---|---|---|
| **用陌生工具**：读 schema、选工具、填参数 | 永远是同样 3 个工具，名字、参数、文档都不变 | AWM-1K（约 3.5 万个工具）、EnvScaler（191 个领域）、xLAM-60k（3,673 个 API） |
| **改状态**：订票、退款、改配置，看最终状态对不对 | 只有只读查询，结果是一个数 | EnvScaler、AWM、τ²-bench（都按数据库的最终状态判分） |
| **和人打交道**：信息不全时追问，遵守业务政策 | 题目一次给全 | τ²-bench（用户模拟器 + 政策文档）、When2Call |
| **知道什么时候不该调用工具** | 只有约 15% 的简单无工具题 | When2Call、XLAM-7.5k-Irrelevance |
| **处理有噪声的信息**：检索结果不相关、需要多跳 | 答案一定在数据库里 | Search-R1（wiki-18 语料）、MuSiQue、2Wiki |
| **长程规划**：几十步、拆子目标 | 最多 6 轮 | ALFWorld、AWM（最多 20 轮）、τ²（多轮对话） |
| **出错后恢复**：超时、空结果、接口变动 | 只有学生自己写错时才出错 | 在现有环境里注入故障（见 §6 的"以后再做"） |
| **难度有梯度**，给 RL 留信号 | 蒸馏后几乎全对或全错，stage 4 有 98% 的组奖励相同 | 带部分得分、难度可调的环境，再加上按通过率筛题 |

## 2. 选数据之前的三个约束

### 2.1 OPD 和 RL 对数据的要求不同

- **纯 OPD（stage 1–3 的做法）只要"任务 + 能执行的环境"。**
  - 学生调用工具时，环境要能真的返回结果，教师再逐 token 打分。
  - **不需要标准答案**，所以 EnvScaler 那 4,684 个没有标签的场景也能用。
- **带奖励的阶段（stage 4–5）还要能用代码验证的奖励**，比如最终状态比对、EM / F1、调用比对。
- **只有录好轨迹、没有可执行环境的数据不能直接做 on-policy。** 这类数据的工具返回值是事先录的，比如 Toucan-1.5M、Nemotron-SFT-Agentic。
  - 学生一走出录好的路径，就拿不到工具返回值。
  - 它们只能做离线 SFT 预热，或者配一个 LLM 来模拟工具，比如 StableToolBench 的 MirrorAPI。
  - 用 LLM 模拟要多占一个模型的显存，模拟出来的返回值也可能是编的。

### 2.2 教师必须先会

学生蒸馏后最多接近教师。所以每接一个新环境，都要先测教师的通过率。已知数字如下：

| 任务 | Qwen3-1.7B | Qwen3-8B | 判断 |
|---|---|---|---|
| BFCL v4 单轮（non-live / live） | 82.9 / 74.6 | 87.6 / 80.5 | 差距小，蒸馏收益有限，主要用来防止能力退化 |
| BFCL v4 多轮 | 11.0 | 41.8 | **差距最大，是主战场** |
| BFCL v4 网页搜索 / 记忆 | 2.5 / 6.0 | 12.0 / 14.6 | 8B 本身也弱 |
| 检索问答，7 个数据集平均 EM（SAGE-OPD） | 18.3 | 35.9 | OPD 后 30.1 |
| ALFWorld unseen 成功率（SAGE-OPD） | 26.1 | 67.2 | OPD 后 61.9 |
| τ²-bench retail / airline / telecom | — | 41.2 / 30.5 / 23.5 | 中等，telecom 偏弱 |
| AppWorld（TGC） | — | 6.6 | 教师不会，暂缓 |
| Terminal-Bench 2.0 / SWE-bench Verified | — | 0.0 / 0.7 | 教师不会，暂缓 |

来源：
- BFCL：[BFCL v4 排行榜数据](https://gorilla.cs.berkeley.edu/data_overall.csv)
- 检索问答、ALFWorld：[SAGE-OPD](https://arxiv.org/abs/2606.19659)
- τ²：[AgentScaler](https://arxiv.org/abs/2509.13311)，论文未注明是否思考模式
- AppWorld：[AppWorld 8B](https://arxiv.org/abs/2607.16062)
- TB2 / SWE-V：[OpenThinker-Agent-v1 模型卡](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1)

需要更强的教师时，必须和学生同一个词表：

| 教师 | 显存（bf16） | 能否和学生、vLLM 一起放进 A100-80GB | 适合 |
|---|---|---|---|
| Qwen3-8B（现在用的） | 16.4 GB | ✓ | 多轮函数调用、检索、ALFWorld |
| [EnvScaler-Qwen3-8B](https://huggingface.co/XXHStudyHard/EnvScaler-Qwen3-8B)、Arctic-AWM-8B | 16.4 GB | ✓，但要先核对 chat template。Arctic-AWM-8B 的模型卡和论文对 base 模型的说法不一致 `[未核实]` | 合成的工具环境 |
| Qwen3-14B | 29.6 GB | ✓ | τ² retail 48.0 |
| Qwen3-30B-A3B-Thinking-2507 | 61.1 GB | 只能 int8（约 30.5 GB） | τ² airline 58.0；只有思考模式，学生也得跟着思考 |
| Qwen3-32B | 65.6 GB | 只能 int8 / AWQ | BFCL v4 总分 48.7，多轮 47.9 |
| Qwen3.5 / 3.6 / 3.8 | — | ✗ 词表 248,320，不兼容 | 只能整套换代：学生换成 Qwen3.5-2B，教师换成 Qwen3.5-9B（**已采用，见 §8**） |

- **A100 用 FP8 省不了显存。** A100 没有 FP8 硬件，transformers 加载 FP8 权重时会还原成 bf16。进程内的教师要用 int8 或 AWQ。
- **另一条路：先用 RL 把教师在目标环境上训强，再拿它蒸馏。** ATOD 的教师就是用 GRPO 训过的 Qwen3-4B。代价是要多一轮训练的算力。

### 2.3 Colab 的限制

- **没有 Docker。**
  - 有些环境每道题要一个 Docker 镜像，比如 SWE-Gym、SWE-smith、R2E-Gym、Terminal-Bench、Toolathlon、MCP-Universe。这些都得用付费的远程沙箱（E2B、Daytona、OpenReward），每小时约 $0.08–0.20。
  - 据调研，TRL 的 Harbor 集成是一个一个地创建沙箱，这期间 GPU 会空等。
- **检索只能跑在 CPU 上。** wiki-18 的 BM25 索引只有 2.3 GB，没有问题。E5 稠密索引有 64.6 GB，会和训练抢内存，不现实。
- **用户模拟器要单独一个 LLM。**
  - τ² 的 airline / retail 需要模拟用户，默认用 gpt-4.1。
  - 也可以在本地用 Qwen3-8B 当用户，但要多占显存，分数也没法和官方榜单比。
  - telecom 有 solo 模式，可以绕开用户模拟器。
- **TRL 的工具集按环境类固定。**
  - v1.16 的 `environment_factory` 可以传一个字典，在一次训练里混合多个环境，每条数据用 `environment` 列选环境。
  - 但同一个环境类暴露的工具是固定的，见 `trl/trl/trainer/grpo_trainer.py` 里的 `_env_tools`。
  - AWM、xLAM 这类数据每道题的工具都不同，有两种接法：
    - 每个场景对应一个环境类；
    - 只暴露 `list_tools` / `call_tool` 两个通用工具，由它们去分发。AWM 论文评测时就是这么做的。

## 3. 候选数据与环境

### 3.1 可执行、有状态的工具环境（首推）

**[EnvScaler](https://github.com/RUC-NLPIR/EnvScaler)**（2026-01，[论文](https://arxiv.org/abs/2601.05808)）
- **规模：** 191 个环境。无标签场景 4,684 个，适合纯 OPD；RL 场景 2,550 个；SFT 轨迹 9K 条。
- **形态：** 每个环境是一个 Python 类，方法就是工具。每个场景给出初始状态和任务。
- **奖励：** checklist 里每一条是一个检查最终状态的 Python 函数，奖励等于通过的比例（部分得分）。
- **依赖：** 纯 Python。RL 用"任务一次给全"的模式，不需要用户模拟器。许可 MIT。
- **小模型证据（思考模式）：**
  - Qwen3-1.7B：BFCL-v3 多轮 9.75 → 23.0，ACEBench-Agent 32.0 → 50.0。
  - Qwen3-8B：BFCL-v3 多轮 28.9 → 41.9。

**[AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K)**（2026-02，ICML'26，[论文](https://arxiv.org/abs/2602.10090)）
- **规模：** 1,000 个环境，每个 10 道题，共 1 万道题，约 3.5 万个工具。
- **形态：** 每个环境是一个 SQLite 数据库，加上生成的 Python / MCP 接口。[OpenEnv 封装](https://github.com/meta-pytorch/OpenEnv/tree/main/envs/agent_world_model_env)可以在本地起服务。
- **奖励：** 两种判分器：纯代码比对数据库的最终状态；或者代码加 LLM 评判。
- **依赖：** 不需要 Docker。许可 CC-BY-4.0。
- **小模型证据：** 论文用 GRPO 训练（GPT-5 评判，每步约 $1.8）。Qwen3-8B 的 BFCLv3 53.8 → 65.9，τ² pass@1 26.4 → 33.5。

**[τ²-bench](https://github.com/sierra-research/tau2-bench)**（[论文](https://arxiv.org/abs/2506.07982)）
- **规模：** 官方的训练 / 测试划分是 airline 30 / 20、retail 74 / 40、telecom 74 / 40。telecom 另有 `full` 集 2,285 题。
- **形态：** 客服场景，有政策文档、工具和数据库。telecom 是"双控"，用户也要操作手机。
- **奖励：** 比对最终状态。telecom 全部是代码断言。
- **依赖：** 用户模拟器默认用 gpt-4.1；telecom 支持 solo 模式，不需要用户。许可 MIT。有 [OpenEnv 封装](https://github.com/meta-pytorch/OpenEnv/tree/main/envs/tau2_env)。
- **小模型证据：** 以 Qwen3-32B 为教师做 token 级 OPD，Qwen3-4B 在测试集上的平均 pass¹ 从 29.2 提到 39.6，教师是 44.2（[ActionOPD](https://arxiv.org/abs/2610.19820) 表 2）。

**[NeMo Gym Workplace Assistant](https://github.com/NVIDIA-NeMo/Gym)**
- **规模：** 训练 1,255 题，验证 545 题；5 个数据库上共 27 个工具。
- **形态：** 用户只说一轮指令，agent 需要多步调用工具。
- **奖励：** 回放标准调用序列，比对最终状态。
- **依赖：** 工具是 pandas 代码，容易抽出来单独用。许可标注不一致 `[未核实]`。
- **小模型证据：** 暂无。

**[BFCL 多轮 API 类](https://github.com/ShishirPatil/gorilla)**
- **规模：** 8 个 API 类，包括文件系统、交易、旅行等。
- **形态：** 纯 Python，有状态。
- **奖励：** 每轮检查状态。
- **许可：** Apache-2.0。
- **注意：** 可以用这些类造新题，但这样 BFCL 多轮评测就成了同分布数据，外部评测只能改用 τ² 和 ACEBench。

旁证：[ActionOPD](https://arxiv.org/abs/2610.19820)（2026-10）正是在这两套环境上做 agent 蒸馏：
- 数据：AWM（794 个环境、5,942 道题）加 EnvScaler（49 个环境、755 道题）。
- 结果：外部评测是 BFCL、ToolSandbox、AppWorld、τ² 四个。Qwen3-4B 学生的平均分提高 6.9，Qwen3-8B 学生提高 4.4。
- 注意：它的教师是 API 模型，蒸馏的是动作，不是 token。

### 3.2 单步函数调用：扩大工具广度，学会"什么时候不调用"

这类数据只有"问题 + 工具 schema + 正确调用"，没有工具返回值，所以只能做**单步**任务：学生只输出第一步调用，按函数名、参数名、参数值打分（[ToolRL](https://arxiv.org/abs/2504.13958) 的做法）。

1.7B 和 8B 在单轮上只差约 5 分，预期收益不大。它的主要用途有三个：
- 扩大见过的工具种类；
- 学会不乱调用；
- 防止多轮训练把单轮能力练坏。

| 名称 | 规模 | 内容 | 许可 |
|---|---|---|---|
| [xLAM-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 6 万条，3,673 个 API | 简单调用、多选一、并行调用 | CC-BY-4.0（需申请访问） |
| [XLAM-7.5k-Irrelevance](https://huggingface.co/datasets/MadeAgents/XLAM-7.5k-Irrelevance) | 7.5K | 给了工具，但不该调用 | `[未核实]` |
| [When2Call](https://huggingface.co/datasets/nvidia/When2Call)（NVIDIA） | 15K SFT + 9K 偏好对 | 分四种情况：该调用、该追问、无法回答、直接回答 | CC-BY-4.0 |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 11.3K 段对话，2.65 万个合成 API | 并行、依赖、嵌套参数 | Apache-2.0 |
| [Nemotron-RL-Agentic-Function-Calling-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1)（2026-03） | 838 个领域 | 把专家轨迹的每一步拆成一道题，题里带着之前的工具返回；附通过率字段 | CC-BY-4.0 |

- **接入方式：** TRL 的工具集按环境类固定，而这些数据每行的工具都不同。做法是把每行的 schema 按 Qwen3.5 的模板手动渲染进 system prompt，不执行工具，由奖励函数解析 `<tool_call>` 里的调用来打分（Qwen3.5 写的是 `<function=...>` / `<parameter=...>` 的 XML，不是 JSON）。
- **[PivotRL](https://arxiv.org/abs/2603.21383) 的发现：** 随机抽的轮次里约 71% 没有学习信号。只在奖励有方差的轮次上训练，Qwen3-30B-A3B 的 τ² 从 58.4 提到 63.8。这和我们 GRPO 没信号是同一个问题。

### 3.3 检索 / 深度研究

**[Search-R1](https://github.com/PeterGriffinJin/Search-R1) 训练集（NQ + HotpotQA）**
- **规模：** 17 万题。
- **运行需要：** wiki-18 语料，约 2,100 万段，压缩后 5.1 GB；加上 [BM25 索引](https://huggingface.co/datasets/PeterJinGo/wiki-18-bm25-index)，2.3 GB，用 Pyserini，需要 JDK，CPU 即可。
- **奖励：** EM。
- **说明：** SAGE-OPD 是同一套设置，但用的是 E5 稠密检索（索引 64.6 GB）。Qwen3-8B → 1.7B，7 个数据集的平均 EM 从 18.3 提到 30.1。

**MuSiQue / 2WikiMultihopQA**
- **规模：** 训练集分别是 19,938 / 167,454 题。
- **运行需要：** 用同一份语料，或者用它们自带的段落池（几十万段，用 [bm25s](https://github.com/xhluca/bm25s) 估计不到 1 GB 内存）。
- **奖励：** F1。
- **说明：** 多跳、难度中等，教师和学生的分数都在中间档，适合带奖励的阶段。

**[InfoSeek](https://huggingface.co/datasets/Lk123/InfoSeek)**
- **规模：** 5.2 万题，其中 Hard 子集 1.8 万。
- **运行需要：** Wikipedia。
- **奖励：** 对 / 错。
- **说明：** 更难的后续选择。答案是否都在 wiki-18 里 `[未核实]`。

**[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)**
- **规模：** 830 题，固定语料 100,195 篇。
- **运行需要：** 索引共 5.25 GB。
- **奖励：** LLM 评判。
- **说明：** Qwen3-32B 也只有 3.5–10.4%，**只作评测**。

开放网页类不推荐，包括 WebSailor、WebExplorer、MiroVerse、Tongyi DeepResearch，以及 ASearcher 的网页模式。原因是它们都要 Serper、Jina 之类的付费 API 和 LLM 评判，而且题目对 8B 教师也太难。

### 3.4 文本环境与长程任务

| 名称 | 规模 | 依赖 |
|---|---|---|
| [ALFWorld](https://github.com/alfworld/alfworld) | 训练 3,553 局；评测 seen 140 局、unseen 134 局 | 文本模式纯 Python，不需要 GPU · MIT |
| [WebShop](https://github.com/princeton-nlp/WebShop) | 约 1.2 万条购物指令 | 搜索引擎需要 JDK |
| [ScienceWorld](https://github.com/allenai/ScienceWorld) | 30 类任务、7,207 个变体 | Java 8+ · Apache-2.0 |

已有结果：

- **ALFWorld unseen**
  - SAGE-OPD（Qwen3-8B → 1.7B）：26.1 → 61.9。
  - ATOD（教师是用 GRPO 训过的 Qwen3-4B）：1.7B 学生 10.2 → 72.7（OPD）→ 80.5（ATOD）。
- **WebShop**
  - ATOD：1.7B 学生 5.5 → 77.3（OPD）→ 89.1（ATOD）。
- **ScienceWorld**
  - 按 0–100 给部分得分，天然有难度梯度。

### 3.5 代码、数据分析、终端、SWE

**[data_agent_rl_environment_train](https://huggingface.co/datasets/AdithyaSK/data_agent_rl_environment_train)**
- **规模：** 2,238 题，另有 366 道评测题，难度分 L1–L5。
- **依赖：** Kaggle 数据 + Python。可以在本地子进程里跑（加资源限制），也可以通过 TRL 的 Harbor 集成配 E2B。许可 Apache-2.0。
- **说明：** 规则判分，要去掉 `llm_judge_long` 那类题。"写代码分析数据"是计算器和 SQL 的自然推广。

**[CodeActInstruct](https://huggingface.co/datasets/xingyaoww/code-act)**
- **规模：** 7.1K 段多轮对话。
- **依赖：** Python 解释器。许可 Apache-2.0。
- **说明：** 把 Python 当作通用的动作。

**Endless Terminals、SETA（通过 OpenReward 使用）**
- **规模：** 分别是 2,490 题和 1,376 题。
- **依赖：** 每题一个容器，按小时计费。
- **说明：** 教师得先会，而 Qwen3-8B 在 TB2 上是 0.0%。

**SWE-Gym / SWE-smith / R2E-Gym / SWE-rebench V2**
- **规模：** 分别是 2.4K、5 万、8.1K、3.2 万题。
- **依赖：** 每题一个 Docker 镜像，总量从几百 GB 到 TB 级。
- **说明：** **不适合我们。** Colab 跑不了，而且 Qwen3-8B 在 SWE-bench Verified 上只有 0.7%。

### 3.6 环境集市：以后找新环境的地方

**[OpenEnv](https://huggingface.co/docs/openenv/environments)**（HF / Meta）
- **规模：** 38 个官方环境，外加社区环境。
- **接入 TRL：** 原生支持，见 `trl/docs/source/openenv.md`。
- **备注：** 在本地用 uvicorn 起服务。包括以下环境：
  - agent_world_model
  - tau2
  - reasoning_gym
  - calendar：37 个工具，但没有附带任务
  - websearch：需要 Serper key

**[OpenReward](https://openreward.ai)**（ORS 协议）
- **规模：** 厂商称有 330 个以上。
- **接入 TRL：** `trl.experimental.openreward`。
- **备注：** 需要 API key，按秒计费。

**[Harbor](https://www.harborframework.com)**
- **规模：** 包括 Terminal-Bench 2.0 等。
- **接入 TRL：** `trl.experimental.harbor`。
- **备注：** 沙箱后端可选 docker、e2b、daytona。

**[Prime Intellect Environments Hub](https://primeintellect.ai/blog/environments)**（`verifiers`）
- **规模：** 400 个以上。
- **接入 TRL：** 要自己包装成 `environment_factory` 类。
- **备注：** 带沙箱的环境要付费。

**[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)**
- **规模：** 约 150 个环境配置。
- **接入 TRL：** 不支持，可以把工具代码抽出来用。
- **备注：** 各环境的许可不同。

**[Reasoning Gym](https://github.com/open-thought/reasoning-gym)**
- **规模：** 100 多个程序生成器，难度可调。
- **接入 TRL：** 单轮任务，最容易接。
- **备注：** 可以替代或扩充 stage 1 的 GSM8K。

## 4. 评测：怎么证明"真实世界能力"变强了

两条原则：
- **训练环境和评测基准分开。** 在 A、B、C 上训练，用没见过的基准衡量泛化能力。
- **旧的 stage 1–3 评测集继续跑**，防止遗忘。

| 评测 | 测什么 | 本地能跑吗 | 注意 |
|---|---|---|---|
| [BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard.html) | 单轮、并行、多轮、无关检测、记忆、网页搜索 | ✓，用 vLLM；网页搜索需要 SerpAPI | **多轮是主指标**。如果用 BFCL 的 API 类造训练题，多轮就不能再当外部评测 |
| τ²-bench 测试集（20 + 40 + 40 题） | 客服对话、遵守政策、最终状态是否正确 | ✓，需要一个用户模拟 LLM | 用户模型不是 gpt-4.1 时，分数没法和官方榜单比。如果训练用了 APIGen-MT（领域相同），airline / retail 就不算外部评测了 |
| ACEBench | 多类函数调用 + Agent 沙箱 | 部分能跑 | 出自 ToolACE 团队，训练用了 ToolACE 就不独立 |
| Search-R1 的 7 个问答集 | 检索问答 EM | ✓ | 直接对标 SAGE-OPD |
| BrowseComp-Plus | 固定语料上的深度检索 | ✓，LLM 评判 | 1.7B 预期很低，只看趋势 |
| ALFWorld unseen（134 局） | 长程文本环境 | ✓ | 直接对标 SAGE-OPD 和 ATOD |

## 5. 数据配比与 GRPO 信号

stage 4/5 为什么没有 GRPO 信号：
- 数据太窄、太容易，蒸馏后学生几乎是确定性的，同一题采样几次结果都一样，组内优势就是 0。
- stage 4 有 98% 的组奖励完全相同，说明绝大多数题对学生来说不是必对就是必错。
- 只靠筛题，剩下的题会很少，所以**必须同时换成更难、难度可调的任务**。

具体做法：

1. **按通过率筛题。**
   - 每个阶段开始前，用当前学生按 RL 的温度对每道题采样 8 次，只留 0 < 通过率 < 1 的题。可参考 MiniMax-M1：它的数学数据保留 pass@10 在 0 到 0.9 之间的题。
   - TRL 不支持 DAPO 的动态采样，`trl/docs/source/paper_index.md` 写明 "Not supported in TRL"。所以这一步放在 `build_dataset` 里离线做。
2. **用部分得分做奖励。** EnvScaler 的 checklist 通过比例、检索的 F1、函数调用按名称 / 参数 / 取值分级打分，都比 0/1 奖励更不容易打平。
3. **提高 RL 阶段的采样温度。** 同时配合 `epsilon_high`，即 DAPO 的 clip-higher，取 0.28。TRL 的 `GRPOConfig` 已经有 `temperature`、`epsilon_high`、`use_adaptive_entropy` 这些选项。
4. **参考 ATOD 做系数退火。**
   - 它的优势是 A = κ·A_OPD + ρ·A_GRPO：κ 在前 80 步从 1.0 线性降到 0.1，ρ 从 1.0 升到 2.0。
   - 在 ALFWorld、WebShop、检索三个任务上，1.7B 学生的平均成功率从 OPD 的 64.9 提到 71.6，超过了教师的 68.9。
   - 我们 stage 4 的公式就是它的一个特例：两个系数都固定为 1。
5. **多个环境混合训练，而不是一个接一个地训。**
   - TRL 的字典 `environment_factory` 支持在一次训练里混合多个环境。
   - MiniMax 报告说，分阶段做 RL 常出现负迁移；Nemotron 3 同时训练所有环境，效果更好。
   - 每个 batch 留 10–20% 的旧任务做回放，包括 GSM8K、计算器、SQL / JSON、单步函数调用。
6. **监控已有的 `frac_reward_zero_std`。** stage 4 是 0.98，目标是降到 0.5 以下。

## 6. 落地计划

按顺序做，每一步都有验收标准。算力按上次的实际消耗粗估，约 6–7 CU/小时。

各阶段共用的工程改动有四项，尽量小：
- `opd/trl_train.py` 支持字典 `environment_factory`，数据集加一列 `environment`。
- 每个环境自带 system prompt 和 `get_reward()`。TRL 原生支持由环境给奖励，新环境不再统一要求 `\boxed{}`。
- 在 `build_dataset` 里加离线的通过率筛题。
- 把 RL 阶段的温度、`epsilon_high` 和 ATOD 退火做成配置项。

### 阶段 A：摸底（不训练，约 1 小时）

**做法：**
- 每个候选环境抽 100–200 道题。教师 Qwen3.5-9B 和当前学生（Qwen3.5 重跑后 stage 4 的 `final/`）各采样 8 次，都用非思考模式（模型卡上的分数是思考模式，见 §8）。
- 记录四项：
  - 教师的 pass@1；
  - 学生的 pass@1；
  - 学生通过率落在 (0, 1) 的题目占比；
  - 平均轮数和 token 数。
- 同时跑一遍外部评测的基线：BFCL v4、τ² telecom 测试集、Search-R1 的 7 个问答集、ALFWorld unseen。

**验收：** 只保留同时满足两个条件的环境：
- 教师比学生高 10 分以上；
- 学生有相当比例的题只部分通过。

不满足的，就换教师或换难度。

### 阶段 B：EnvScaler（多轮、有状态的工具调用）

**为什么先做它：**
- 它的环境本来就是"一个 Python 类，方法即工具"，和本仓库 `opd/envs/base.py` 的约定一样。
- 有 Qwen3-1.7B 的公开结果可以对照。

**接入：** 新建 `opd/envs/envscaler.py`。从 HF 加载 191 个环境类，为每个类生成一个 `Env` 子类：
- 只暴露该环境声明的工具；
- `reset(**task)` 载入初始状态；
- `get_reward()` 运行 checklist。

环境类是 LLM 写的代码，要在子进程里加资源限制运行。

**训练：**
1. 先用 4,684 个无标签场景做纯 OPD，`reward_coef` 设为 0。
2. 再在 2,550 个 RL 场景上做"OPD + 部分得分奖励"，用 ATOD 的退火。

**验收：**
- BFCL v4 多轮比 11.0 有明显提升。参照：EnvScaler 论文里，1.7B 经 SFT+RL 后 BFCL-v3 多轮 9.75 → 23.0。
- 旧的 stage 1–3 评测不掉点。
- `frac_reward_zero_std` < 0.5。

**算力：** 约 2–3 小时。

### 阶段 C：检索 + ALFWorld（有同一对教师 / 学生的公开对照）

**接入：**
- `opd/envs/search.py`：一个 `search(query)` 工具，在 wiki-18 上用 BM25 检索，返回前 3 段。按 EM / F1 判分。
- `opd/envs/alfworld_env.py`：一个 `act(command)` 工具，完成任务即得分。

**数据：**
- 纯 OPD 用 NQ + HotpotQA。
- 带奖励的阶段用 MuSiQue 和 2Wiki，F1 奖励，按通过率筛过。
- ALFWorld 用 3,553 局训练游戏。

**验收：**
- 对标 SAGE-OPD：检索问答 30.1 EM，ALFWorld unseen 成功率 61.9%。
- 我们用的是 BM25，数字会偏低。参考 Search-R1 的[检索器对比](https://arxiv.org/abs/2505.15117)：7B 模型用 BM25 是 0.352，用 E5 是 0.430。

**算力：** 约 2–3 小时，另需约 10 GB 磁盘。

### 阶段 D：AWM-1K + τ²-bench telecom（扩大工具和领域的广度）

**AWM：**
- 用纯代码判分器，省掉 GPT-5 评判的费用。
- 可以直接用 OpenEnv 的封装，也可以自己写一个 `list_tools` / `call_tool` 的分发环境。

**τ² telecom：**
- 训练用 `full` 集去掉 40 道测试题后剩下的 2,245 题，跑 solo 模式，不需要用户模拟器。
- 评测用那 40 道测试题，外加 airline / retail 作为外部评测。

**注意：** Qwen3-8B 在 telecom 上只有 23.5；Qwen3.5-9B 的 TAU2 总分是 79.1，但那是思考模式。阶段 A 要特别确认教师在非思考、solo 模式下的通过率；不够的话，换 Qwen3.5-27B（int8）当教师。

**算力：** 约 2–3 小时。

### 阶段 E：混合训练

**做法：** 用字典 `environment_factory`，把阶段 B–D 的环境放进一次训练。每个 batch 再加上：
- 10–20% 的单步函数调用题（xLAM-60k、XLAM-Irrelevance、When2Call）；
- 旧任务的回放。

**初始配比**（之后按阶段 A 和各阶段的结果调整）：

| 环境 | 占比 |
|---|---|
| EnvScaler + AWM | 35% |
| τ² telecom | 15% |
| 检索 | 20% |
| ALFWorld | 15% |
| 单步函数调用 + 回放 | 15% |

**验收：** §4 的所有外部评测，都不低于各单项阶段的结果。

**算力：** 约 3–4 小时。

### 以后再做（需要额外的预算或你来决定）

- **τ² airline / retail 的用户模拟对话：** 需要本地的用户模型，或者付费 API。
- **数据分析的代码环境**（data_agent_rl_environment）：要先实现子进程沙箱。
- **现有 SQL / JSON 环境"去记忆化"：**
  - 随机改写工具名、参数名和文档；
  - 加入干扰工具；
  - 注入超时、空结果等故障；
  - 加入需要追问的题。
- **换更强的教师：** Qwen3.5-27B / 35B-A3B 或 Qwen3.6-35B-A3B，tokenizer 与学生相同，在 A100 上要 int8（§8）。

## 7. 风险与需要你决定的事

- **许可：**
  - 只能非商用的：APIGen-MT-5k（CC-BY-NC-4.0，另有附加条款）、ToolScale（NVIDIA 非商用）、MiroVerse、AgentGym 的数据、InfiAgent 的数据。
  - 首推的几个许可都宽松：EnvScaler（MIT）、AWM（CC-BY-4.0）、τ²-bench（MIT）、Search-R1（Apache-2.0，数据 CC BY-SA）、ALFWorld（MIT）。
- **环境质量：**
  - EnvScaler 和 AWM 的环境代码、判分器都是 LLM 生成的，会有 bug。AWM 论文报告训练中约 4% 的环境调用出错。
  - 所以要在子进程里运行、人工抽查，并过滤掉教师也做不对的题。
- **思考模式：** Qwen3.5 模型卡上的 agent 分数都是思考模式，而本仓库一直用非思考模式。换成思考模式做 OPD，序列更长、训练更慢，而且 2B 的模型卡提醒它比别的 Qwen3.5 更容易陷入思考循环。
- **序列长度：**
  - AWM、τ² 的对话可能有 20–30 轮，比 stage 3 的 6 轮长得多，显存和时间都会上去。
  - 要按阶段 A 测出的 token 数来设 `max_completion_length`。
- **需要你决定：**
  1. 教师怎么选：先只用 Qwen3.5-9B，还是在阶段 D 之前就换上 int8 的 Qwen3.5-27B / 35B-A3B？
  2. 是否愿意为 τ² 的用户模拟或代码沙箱付费？
  3. ~~是否考虑整套换代到 Qwen3.5（学生 2B、教师 9B）？~~ **已决定（2026-10-20）：换。** 见 §8。
  4. （新）agent 任务用不用思考模式？阶段 A 先在非思考模式下测；教师在非思考模式下明显不够时再讨论。

## 8. 换到 Qwen3.5 之后（2026-10-20）

### 8.1 核实过的事实

| 项目 | Qwen3.5-2B（学生） | Qwen3.5-9B（教师） |
|---|---|---|
| 参数（含视觉塔） | 2.21B | 9.41B |
| 语言模型部分 | 1.88B，bf16 约 3.8 GB | 8.95B，bf16 约 17.9 GB |
| 层结构 | 24 层，每 3 层 Gated DeltaNet（线性注意力）配 1 层门控注意力 | 32 层，同样 3 : 1 |
| 默认模式 | 不思考 | 思考 |
| 词表 | 248,320 | 248,320 |

- 两者的 `tokenizer.json` 完全相同（sha256 一致），0.8B 也一样，可以做 token 级 OPD。
- 都是带视觉编码器的模型（`Qwen3_5ForConditionalGeneration`），本仓库只做文本任务：
  - 教师用 `AutoModelForCausalLM` 只加载语言模型；
  - TRL 训练学生时加载完整模型，LoRA 跳过视觉塔。
- **工具调用格式变了。** Qwen3 写 JSON；Qwen3.5 写 XML：`<tool_call>` 里套 `<function=名字>`，参数写成 `<parameter=参数名>值</parameter>`。参数值不带引号，类型只能靠工具的 schema 判断。TRL 已支持这种格式；从零实现的 trainer 换了解析器。
- 没有 `generation_config.json`；结束一轮的仍是 `<|im_end|>`。
- 训练时要装 `flash-linear-attention`，给线性注意力层提供 Triton kernel；不装的话 transformers 会退回到正确但很慢的 PyTorch 实现。Mac 上只能用后者。

### 8.2 教师和学生差多少（模型卡，思考模式）

| 基准 | 0.8B | **2B（学生）** | 4B | **9B（教师）** | 27B | 35B-A3B |
|---|---|---|---|---|---|---|
| BFCL-V4 | 25.3 | **43.6** | 50.3 | **66.1** | 68.5 | 67.3 |
| TAU2-Bench | 11.6 | **48.8** | 79.9 | **79.1** | 79.0 | 81.2 |

对照：Qwen3-1.7B 和 Qwen3-8B 的 BFCL v4（FC 模式）总分是 28.4 和 42.6。

对本计划的影响：
1. **能教的东西更多了。** 9B 比 2B 在 BFCL-V4 上高 22.5 分，在 TAU2 上高 30 分；原来 8B 比 1.7B 的 BFCL 总分只高 14 分。
2. **这些都是思考模式的分数，本仓库用的是非思考模式。** 非思考模式序列短，也是 2B 的默认。9B 在非思考模式下的 agent 分数没有公开，阶段 A 必须先测。
3. **4B 的 TAU2 和 9B 一样高（79.9 对 79.1）。** τ² 上教师未必越大越好；以后可以试 4B 当教师，显存更省。
4. **SWE / 终端仍然暂缓。** 9B 的模型卡没有 Terminal-Bench、SWE-bench 的分数。27B 有（TB2 41.6、SWE-V 72.4），教师能力够了，但在 A100 上要 int8，而且每道题要一个 Docker 镜像，Colab 跑不了。
5. **§3 引用的小模型证据（EnvScaler、SAGE-OPD、ATOD）都是在 Qwen3 上做的。** 环境照用，数字只能作间接参照。

### 8.3 以后可换的更强教师（同一个 tokenizer）

| 教师 | 参数 | bf16 | A100-80GB 上（和学生、vLLM 同卡） | 已知分数（思考模式） |
|---|---|---|---|---|
| Qwen3.5-27B | 27.8B | 约 56 GB | 要 int8（约 28 GB） | BFCL-V4 68.5，TB2 41.6，SWE-V 72.4 |
| Qwen3.5-35B-A3B（MoE） | 36.0B | 约 72 GB | 要 int8 | BFCL-V4 67.3，TAU2 81.2 |
| Qwen3.6-35B-A3B（MoE） | 36.0B | 约 72 GB | 要 int8 | TB2 51.5，SWE-V 73.4，MCPMark 37.0 |
| Qwen3.8-27B | 27.8B | 约 56 GB | 要 int8 | 它的 `tokenizer.json` 和 Qwen3.5 不同（词表大小一样），用前要核对 |

参数量含视觉塔和 MTP 层；前三个的 `tokenizer.json` 与学生完全相同。

### 8.4 代码改动（提交 `f71c46b`）

- `opd/configs/base.py`：学生 `Qwen/Qwen3.5-2B`，教师 `Qwen/Qwen3.5-9B`；`debug` 预设换成 0.8B / 2B。
- `opd/chat.py`：
  - 解析 XML 工具调用。
  - Qwen3.5 的模板不能单独渲染工具消息，所以工具结果的 token 改为：把一段假对话带结果、不带结果各渲染一次，取两者之差。
- `opd/trl_train.py`：
  - 从 stage 2 起，用和 TRL 一样的方式加载学生（`create_model_from_path`，带视觉塔的完整模型类）。原来的 `AutoModelForCausalLM` 只加载语言模型，模块名不同，上一阶段的 LoRA 会被**静默丢掉**，vLLM 也认不出参数名。
  - LoRA 跳过视觉塔。
  - 把我们自己的 tokenizer 传给 TRL，不让它加载图像处理器。
- 测试：
  - 替身模型换成小号 Qwen3.5（带视觉塔），72 个测试全部通过。
  - 跨阶段测试在旧的加载方式下会失败，能抓住上面那个 bug。
- 用真实的 Qwen3.5-2B tokenizer 核对过两件事：拼接出的 token 和模板重新渲染整段对话完全一致；TRL 能正确解析 XML 调用。

### 8.5 重跑 stage 1–4（已完成）

- 2026-10-20 在 A100-80GB 上用 TRL 后端跑完，配置和 Qwen3 那一轮相同。结果和完整对比见 [`results/trl/a100-qwen3.5/README.md`](../results/trl/a100-qwen3.5/README.md)；Qwen3 那一轮保留在 `results/trl/a100/`。
- 训练共约 7.1 小时、约 50 CU。Qwen3 那一轮是 2.8 小时、23 CU。

| Stage | eval reward（括号内为 Qwen3） | eval `teacher_kl`（括号内为 Qwen3） |
|---|---|---|
| 1. GSM8K | 0.656 → 0.781（0.809 → 0.836） | 0.218 → 0.168（0.229 → 0.136） |
| 2. 计算器 | 0.645 → 0.902（0.473 → 1.000） | 0.108 → 0.047（0.328 → 0.057） |
| 3. 多工具、多轮 | 0.754 → 0.957（0.660 → 0.746） | 0.163 → 0.034（0.378 → 0.068） |
| 4. 蒸馏 + 任务奖励 | 0.945 → 0.953（0.754 → 0.758） | 0.034 → 0.025（0.065 → 0.067） |

对本计划的影响：
1. **算力要按 2 倍以上估。**
   - 两轮每个 token 的成本一样，但 Qwen3.5 的回答长 2–5 倍：每次调用工具前后都写分步解释。
   - §6 各阶段的时长是按 Qwen3 的速度估的，阶段 A 测出 token 数后要重估。
2. **GRPO 信号多了一点，但仍不够。**
   - 蒸馏后学生的策略熵约 0.14，Qwen3 是 0.01–0.03。stage 4 里有奖励差异的组从 1.5% 升到 11%。
   - 但学生已能做对约 95% 的 stage 3/4 题，89% 的组仍然没有信号。
   - §5 的结论不变：要换更难的任务，并按通过率筛题。
3. **长度和轮数上限要按模型重设。**
   - GSM8K 上有 10–18% 的回答写满 1024 token，被截断得 0 分。这些不是复读：模型遇到非整数之类的中间结果后，在回答里反复自我怀疑。
   - 计算器任务里，剩下的失败大多是学生验算时多调用工具，把 4 轮用完了。
   - 新环境的 `max_completion_length` 和轮数上限，要按阶段 A 测出的长度来设（见 §7「序列长度」）。
4. 阶段 A 的学生用 `results/trl/a100-qwen3.5/stage4/final/`。

## 附：调研中引用的关键论文

- **用 on-policy distillation 训练 agent 的直接证据：**
  - [SAGE-OPD](https://arxiv.org/abs/2606.19659)：Qwen3-8B → 1.7B，任务是检索、ALFWorld、ScienceWorld。
  - [ATOD](https://arxiv.org/abs/2606.27814)：OPD 和 GRPO 结合，系数退火。
  - [ActionOPD](https://arxiv.org/abs/2610.19820)：在 AWM + EnvScaler 上训练，并在 τ² 上和 token 级 OPD 做了对照。
- **环境合成：**
  - [EnvScaler](https://arxiv.org/abs/2601.05808)
  - [AWM](https://arxiv.org/abs/2602.10090)
  - [AgentScaler](https://arxiv.org/abs/2509.13311)
  - [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)：1,827 个合成环境。
- **训练信号：**
  - [PivotRL](https://arxiv.org/abs/2603.21383)
  - [DAPO](https://arxiv.org/abs/2503.14476)
  - [ToolRL](https://arxiv.org/abs/2504.13958)
- **教师能力：**
  - [Qwen3 技术报告](https://arxiv.org/abs/2505.09388)
  - Qwen3.5 模型卡：[0.8B / 2B](https://huggingface.co/Qwen/Qwen3.5-2B)、[9B](https://huggingface.co/Qwen/Qwen3.5-9B)、[27B / 35B-A3B](https://huggingface.co/Qwen/Qwen3.5-27B)、[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
  - [BFCL v4 排行榜](https://gorilla.cs.berkeley.edu/leaderboard.html)
