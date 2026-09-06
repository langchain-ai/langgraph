# LangGraph 源码学习计划（可执行版）

> 目标读者：Python 熟练、想**深入读懂 LangGraph 源码**（而不仅是会用）的开发者。
> 前置：Python 3.11+、`uv`、git、本地能联网；Docker 可选（有则跑全量测试，无则用 `uv run pytest`）。
> 时间设定：每天 1 个 Session ≈ 3 小时；共 31 个 Session ≈ **6~8 周**（含 3 个弹性缓冲日）。

---

## 0. 这套计划的"玩法规则"（先读我）

本计划与普通读书计划的最大区别：**每个 Session 都有强制产出物和验证动作**，读完 ≠ 学会。

1. **先预测，再验证**：动手写一段小实验脚本，先猜输出，再运行，再打开源码解释"为什么"。
2. **产出物必须存在**才算完成，否则该 Session 不勾。
3. **源码一律带 `文件:行号` 精读锚点**，不整文件通读。
4. 需要精读前，**先自己画心智图**，最后用 codegraph 交叉校对。
5. 计划末尾有**进度跟踪表**，每天勾选；建议每天用最后 10 分钟回答当日 Quiz。

### 代码约定

- 所有实验脚本**统一放在 `libs/langgraph/study-lab/` 下**（新建目录，最后可删除或加入 `.gitignore`），命名如 `s01_min_graph.py`。
- 运行命令**一律在对应库目录下**用 `uv run` 执行，确保用上库内依赖而非全局旧版：

```bash
cd /root/code/langgraph/libs/langgraph
uv run python study-lab/s01_min_graph.py
```

- 笔记与进度表放 `libs/langgraph/study-lab/NOTES.md`。

### 命令速查

| 用途 | 命令（在 `libs/langgraph/` 下执行） |
|---|---|
| 装全依赖 | `make install`（= uv sync 全 workspace+dev group，较重，只装一次） |
| 跑单个测试文件 | `uv run pytest tests/test_state.py -x -q`（不用 Docker，推荐） |
| 跑测试并自动起 docker 依赖 | `TEST=tests/test_pregel.py make test` |
| 只跑格式/lint/类型 | `make format` / `make lint`（AGENTS.md 要求改动前必跑） |
| 单文件行数/函数定位 | 用 codegraph：`codegraph explore "Pregel stream invoke superstep"` |

---

## 阶段一：使用层 —— 先做"会开车的人"（D1–D7，约 21h）

> 原则：只学 API 语义，不读实现。所有实验**不用真实 LLM**（用假模型或纯图逻辑），保证离线可跑、可复现。

### Session D1 ｜ 图与状态的最小闭环

- 🎯 目标：能用 `StateGraph` 搭一个无 LLM 的图并跑通，理解 `Node + Edge + State` 三者关系。
- 📖 动作：
  1. 读仓库 `README.md` 与 `libs/langgraph/langgraph/graph/state.py:131`（`class StateGraph`）、`:1177`（`compile`）附近的类注释，建立整体感。
  2. 在 `libs/langgraph/study-lab/s01_min_graph.py` 写最小计数器图（两个节点，观察 reducer 累加）：

```python
from typing import Annotated, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    total: Annotated[int, add]

def inc(st: S) -> dict:  # reducer 是 add：返回值会与当前值"累加"，而非覆盖
    return {"total": 1}

g = (
    StateGraph(S)
    .add_node("a", inc)
    .add_node("b", inc)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
)
app = g.compile()
print(app.invoke({"total": 0}))   # 先预测输出
```

  3. **先手写预测输出再运行**，对照差异。
- ✅ 验证/产出：
  - 脚本能跑通，输出 `{'total': 2}`；能解释为什么是 `2` 而非 `1`（`a`、`b` 两次写入各 +1，被 `add` 累加）。
  - 对比实验：把 `total` 的 `Annotated[int, add]` 改成普通 `int`（无 reducer，等价 LastValue 覆盖），再跑，观察结果变为 `1` 并解释差异。
  - 答不出就去读 `libs/langgraph/langgraph/channels/binop.py`（`add` → `BinaryOperatorAggregate`）与 `channels/last_value.py`（无 reducer 的覆盖语义）。

### Session D2 ｜ 消息态与 Reducer 语义

- 🎯 目标：搞懂 `MessagesState`/`add_messages` 与"写入—合并"规则。
- 📖 动作：
  1. 精读 `libs/langgraph/langgraph/graph/message.py:61`（`add_messages`）、`:372`（`MessagesState`）和 `graph/state.py` 中 `Annotated`/reducer 处理逻辑。
  2. 在 `study-lab/s02_messages.py` 用 `StateGraph(MessagesState)` 挂两个"伪节点"（返回假 `{"messages":[("ai","hi")]}`），观察消息如何累积。
  3. 替换 reducer 为 `operator.add` 再跑，对比报错或行为差异。
- ✅ 验证/产出：
  - 跑通并回答：ChatMessage 列表做 reducer 时靠什么判断"追加 vs 覆盖"？
  - 运行 `uv run pytest tests/test_messages_state.py -x -q` 全绿。

### Session D3 ｜ 分支、循环与并行扇出（send）

- 🎯 目标：会画条件边、会用 `send()` 做动态 fan-out，能理解执行顺序不确定性。
- 📖 动作：
  1. 浏览示例 `examples/customer-support/customer-support.ipynb`（条件分支+多工具）。
  2. 在 `study-lab/s03_fanout.py` 复刻：一个入口节点用 `send()` 动态发给 N 个子节点，再汇总；每个节点打印执行顺序。
  3. 翻 `libs/langgraph/langgraph/graph/_branch.py` 了解条件边如何被表达。
- ✅ 验证/产出：
  - 跑 3 次，能解释输出顺序为何可能不同（并发）而结果一致（reducer 合并）。
  - 运行 `uv run pytest tests/test_algo.py -x -q`（只看 fanout/并行相关用例）。

### Session D4 ｜ 持久化：从内存到 Sqlite

- 🎯 目标：理解 `thread_id` + checkpointer 是"有状态会话"的根基，会用两种后端。
- 📖 动作：
  1. 读 `libs/checkpoint/langgraph/checkpoint/` 顶层 `base`（`base/__init__.py:177` 的 `BaseCheckpointSaver`）与 `memory/__init__.py`（`InMemorySaver`，旧别名 `MemorySaver`）。
  2. `study-lab/s04_checkpoint.py`：同一张图先后用 `InMemorySaver`（`from langgraph.checkpoint.memory import InMemorySaver`）与 `SqliteSaver`（`from langgraph.checkpoint.sqlite import SqliteSaver`），分两次 `invoke`（同 `thread_id`）验证状态跨调用保留。
  3. 调用 `app.get_state(config)` 查看中间快照，体会"每一步都存 checkpoint"。
- ✅ 验证/产出：
  - 两次 invoke 之间数据不丢；能画出 checkpoint 在"调用→调用"间的桥梁作用。
  - 运行 `uv run pytest tests/test_checkpoint_migration.py -x -q` 全绿。

### Session D5 ｜ 人机回环：interrupt 与恢复

- 🎯 目标：会用 `interrupt()` 暂停图、用 `Command(resume=...)` 或 `update_state` 恢复。
- 📖 动作：
  1. 读 `libs/langgraph/langgraph/types.py:851`（`interrupt`）的 docstring 语义。
  2. `study-lab/s05_approval.py`：搭"请求→审批→执行"两节点流程，第二个节点先 `interrupt`，主进程 `invoke` 后捕获暂停值，再恢复。
  3. 体验 `app.get_state()` / `app.update_state()` 手动改状态后继续（`pregel/main.py:1392` / `:2515` 是本库的入口，不必精读）。
- ✅ 验证/产出：
  - 脚本完整演示"暂停→人工干预→继续→拿终态"。
  - 运行 `uv run pytest tests/test_interruption.py -x -q` 全绿。

### Session D6 ｜ 流式输出四件套

- 🎯 目标：分清 `stream` 的 mode（values/updates/custom）与 `astream_events` 的事件流。
- 📖 动作：
  1. 读 `libs/langgraph/langgraph/pregel/main.py` 中 `stream`（`@overload` 于 `:2616` 起，实现于 `:2655`）与 `astream`（实现于 `:3063`）的 docstring 和签名。
  2. `study-lab/s06_stream.py`：对同一图分别用 `stream(..., mode="updates")` 和 `mode="values"` 打印，对比差异；再在节点间 `asyncio.sleep` 看流式节奏。
- ✅ 验证/产出：
  - 能说清 values 与 updates 的差异；Quiz：什么场景必须用 `astream_events`？
  - 运行 `uv run pytest tests/test_pregel.py -x -q` 中的 stream 相关用例通过。

### Session D7 ｜ 周作业：不靠 prebuilt 手写 ReAct

- 🎯 目标：综合运用前 6 天知识。
- 📖 动作：在 `study-lab/s07_react.py` 用 `StateGraph(MessagesState)` 手写：LLM 节点（模型可用 `libs/langgraph/tests/fake_chat.py` 的假模型替换，保证离线）+ 工具节点 + 条件边（有 tool_call 就调工具，否则 END），最后接入 Sqlite 持久化。
  - 对照官方参考答案：`examples/react-agent-from-scratch.ipynb`。
- ✅ 验证/产出：
  - 完整跑通一轮"思考→调工具→汇总"；回答 5 道自测题并记录在 `study-lab/NOTES.md`。

> **阶段一里程碑 ✅**：不查资料能 20 分钟内搭出一个带分支/循环/持久化的图。

---

## 阶段二：核心引擎源码精读（D8–D16，约 27h）

> 方法强制：**每个源码模块先 15 分钟"黑盒实验"观察行为 → 再精读对应文件 → 最后补笔记**。禁止拿到文件就从头读到尾。

### Session D8 ｜ 入口链路：invoke 到底调了什么

- 📖 动作：
  1. 用 `python -m pdb` 或 IDE 断点，在 `pregel/main.py:3783`（`invoke`）、`:3960`（`ainvoke`）下断点，单步走 5 层调用。
  2. 用 codegraph 查 "Pregel stream astream 主循环" 拿到真实调用路径。
  3. 把观察到的调用栈记到 `study-lab/NOTES.md`（先凭自己写 v1，不做核对）。
- ✅ 验证/产出：一份从 `app.invoke` → checkpoint 落盘的**第一版调用链图**（v1），能指着每层说清职责。

### Session D9 ｜ 图对象 vs 编译产物

- 📖 动作：
  1. 精读 `graph/state.py` 的 `StateGraph`（`compile` 于 `:1177`）与 `graph/_node.py`，弄清：`add_node` 存了什么、`compile` 之后得到什么。
  2. `dir(app)`、`app.nodes`、`app.channels` 打印编译产物结构，观察 key 命名规律。
- ✅ 验证/产出：回答 Quiz：为何 runtime 是 `Pregel` 而用户看到的是编译后的图？两者的边界在哪。

### Session D10 ｜ Channels 家族（状态合并的地基）

- 📖 动作：
  1. 精读 `channels/base.py:19`（`BaseChannel` 抽象：`update`/`consume`/`finish`/`checkpoint`/`from_checkpoint`/`get`）→ `last_value.py`（覆盖）、`topic.py`（消息队列，注意 `consume` 的消费语义）、`binop.py`（`add` 归并）。
  2. 黑盒实验：跑 `uv run pytest tests/test_channels.py -x -q`，挑 2 个用例用断点看 `update` 被调用的时机。
- ✅ 验证/产出：能画出"一次节点写入 → channel.update → 读回"的数据流；能说清 `update`（写）与 `consume`（通知订阅任务已跑）的区别。
  - 注意：本仓库的 channel **不再自带 version**，版本号已上移到 checkpoint 元数据（`pregel/_checkpoint.py` 的 `channel_versions`），该机制留到 D15 精读。

### Session D11 ｜ Pregel 主循环（上）：如何决定"下一步执行谁"

- 📖 动作：
  1. 精读 `pregel/_algo.py`：定位任务规划函数（`prepare_next_tasks` 一类），理解它如何读「上一步写入 + 条件边」产出下一批 tasks。
  2. 结合 `protocol.py`（抽象接口，先建立"哪些行为是被约定的"）。
  3. 黑盒验证：在 fanout 图上打印每次"一波 task"的构成。
- ✅ 验证/产出：笔记中记录"一个 superstep = ?"，能用一句话讲清 Pregel 的"并行成批执行→同步屏障→写回"节奏。

### Session D12 ｜ Pregel 主循环（下）：loop 与 runner

- 📖 动作：
  1. 精读 `pregel/_loop.py` 与 `pregel/_runner.py`：superstep 循环怎么驱动、task 结果如何回写状态。
  2. 回读 `main.py` 的 `stream`（`~:2616`）生成器结构，看每步 `yield` 的时机。
- ✅ 验证/产出：能对着 v1 调用链图补上"loop/runner 属于哪一层"，图升级到 v2。

### Session D13 ｜ 并行与调度细节

- 📖 动作：
  1. 精读 `pregel/_executor.py`（线程池/并发上限）、`pregel/_config.py`（`max_concurrency`、`recursion_limit` 等配置）。
  2. 黑盒实验：N 个节点各 `asyncio.sleep`，改变 `max_concurrency` 观察总耗时变化。
- ✅ 验证/产出：能解释配置项最终作用在调度哪一环；记录 1 个"并发导致踩坑"的案例。

### Session D14 ｜ 重试、超时与错误边界

- 📖 动作：
  1. 精读 `pregel/_retry.py`；读配置里 retry policy 字段。
  2. 黑盒实验：让节点第 1 次抛异常、第 2 次成功，验证自动重试；再验证超过 `recursion_limit` 报错。
  3. 跑 `uv run pytest tests/test_retry.py -x -q`。
- ✅ 验证/产出：能用配置实现"有界重试+退避"，并解释 checkpoint 对重试的作用（呼应 D15 的 `channel_versions` 版本机制）。

### Session D15 ｜ 时隙回放（Time Travel）与状态改写

- 📖 动作：
  1. 精读 `pregel/_checkpoint.py`（引擎如何写 checkpoint、`checkpoint_id` 如何定位），重点看 `channel_versions`（`:34`）与 `get_next_version`（`:156`）——这是 D10 提到的版本号机制的归属地。
  2. 精读 `main.py` 的 `get_state`(`:1392`)、`get_state_history`(`:1480`)、`update_state`(`:2515`)。
  3. 黑盒：对图连续 invoke 3 次，枚举历史快照，fork 到早期快照重跑。
  4. 跑 `uv run pytest tests/test_time_travel*.py -x -q`。
- ✅ 验证/产出：独立跑通"回退到第 1 步→改输入→重演"的完整 demo；能讲清 checkpoint 与 memory 的差别，以及"为什么靠 channel 版本号能只增量保存变化"。

### Session D16 ｜ 复盘：把引擎讲给自己听

- 📖 动作：不看代码，在 `study-lab/NOTES.md` 里手写完整架构说明（编译→执行→持久化三段），再用 codegraph 查询关键符号核对纠偏。
- ✅ 验证/产出：**能对同学/同事 10 分钟讲清 v2 架构图**；圈出 3 个自己仍模糊的点作为 D17+ 重点。

> **阶段二里程碑 ✅**：能徒手画出 `invoke` 全链路时序图，讲清 superstep/version/checkpoint 三概念。

---

## 阶段三：持久化与记忆体系（D17–D21，约 15h）

### Session D17 ｜ checkpoint 抽象接口精读

- 📖 动作：精读 `libs/checkpoint/langgraph/checkpoint/`：`base`（`BaseCheckpointSaver` 方法契约）、`memory.py`、`serde/`（序列化机制）。
- ✅ 验证/产出：画出"引擎(saver 调用方) ↔ saver ↔ 存储"边界；说明一个 saver 必须实现哪几个方法。

### Session D18 ｜ 引擎侧交互与序列化

- 📖 动作：精读 `langgraph/langgraph/pregel/_checkpoint.py` 与 `_serde`/序列化相关 utils；弄清写 checkpoint 的触发时机与内容（state + version + next + 元数据）。
- ✅ 验证/产出：回答 Quiz：checkpoint 何时落盘？粒度是一个节点还是一个 superstep？（对照 D10/D12 结论）

### Session D19 ｜ 读一个真实后端实现

- 📖 动作：对比 `libs/checkpoint-sqlite` 与 `libs/checkpoint-postgres` 的同名方法实现，找"同一接口、不同 SQL 引擎"差异点（表结构、连接、事务）。
  - 本地验证（无 docker 可跑，只跑 sqlite 本端用例）：`cd libs/checkpoint-sqlite && uv run pytest tests/test_sqlite.py tests/test_aiosqlite.py -x -q`。
- ✅ 验证/产出：写一份"若我实现一个 Redis 版 saver，要复刻哪些行为"的清单。

### Session D20 ｜ 长记忆 Store 与 ManagedValue

- 📖 动作：精读长记忆 Store（`libs/checkpoint/langgraph/store/` 的 `base` 与 `memory`），再读 `libs/langgraph/langgraph/managed/base.py:18`（`ManagedValue` 生命周期，图运行时如何按需注入共享值）。
- ✅ 验证/产出：跑通 `uv run pytest tests/test_managed_values.py -x -q`；能用一句话区分 Checkpoint（运行态）vs Store（长期记忆）。

### Session D21 ｜ 周作业：写一个自定义 CheckpointSaver

- 🎯 目标：落地为代码。
- 📖 动作：继承 `BaseCheckpointSaver` 实现**内存+写盘 JSON**版 saver（约 80~120 行），接入 D7 的 ReAct，验证断点续跑与 fork 回放。
- ✅ 验证/产出：自定义 saver 通过 `get_state_history`/`update_state` 全流程；代码过 `make lint`。

---

## 阶段四：高层封装与周边库（D22–D27，约 18h）

### Session D22–23 ｜ prebuilt：create_react_agent 解剖

- 📖 动作：精读 `libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py:278`（`create_react_agent`）、`tool_node.py`、`tool_validator.py`、`interrupt.py`。
  - 黑盒：用 `libs/prebuilt/tests/model.py` 的 `FakeToolCallingModel` 假模型 + 假工具跑通 `create_react_agent`，打印它编译出的图结构（`graph.get_graph().draw_mermaid()`）。
- ✅ 验证/产出：能列出"prebuilt 帮你包了多少样板"，指出与 D7 手写版的差距。

### Session D24 ｜ cli 与本地服务

- 📖 动作：在 `libs/cli/` 下读 README 与 `langgraph_cli/cli.py` 的命令分派；跑 `make start-dev-server`（在 langgraph lib 内）看 `langgraph dev` 如何起本地 API。
- ✅ 验证/产出：本地起服务、curl 健康检查、能讲清 dev/build 各自做啥。

### Session D25 ｜ sdk-py：与远端 Graph 通信

- 📖 动作：读 `libs/sdk-py` 的 client 分层（connect→get graph→invoke），对比 `pregel/remote.py` 的 `RemoteGraph`。
- ✅ 验证/产出：能回答：本地进程调用 vs 走 API 调用，对使用者差异是什么？

### Session D26 ｜ sdk-js 快速对照（可选）

- 📖 动作：扫一遍 `libs/sdk-js` 的包结构与 README，不逐行读。
- ✅ 验证/产出：能说出 JS/Py SDK 在"线程/中断/流式"上 API 对齐度。

### Session D27 ｜ 周作业：复刻精简版 create_react_agent

- 🎯 目标：从"读得懂"到"写得出来"。
- 📖 动作：不 import prebuilt，仅用 `langgraph` 底层 API 复刻一个仅支持 messages+tools 的精简版 `create_react_agent(agent) `，跑通与 prebuilt 同一批行为用例（多轮工具调用、空输入）。
- ✅ 验证/产出：`libs/prebuilt/tests/` 中挑 3 个行为用例，用自己实现跑通或解释差异。

---

## 阶段五：动手贡献（D28–D31+，约 12h + 弹性）

### Session D28 ｜ 用"读测试"验证理解

- 📖 动作：精读 3 个"高信号"测试文件：`tests/test_state.py`、`tests/test_pregel.py`、`tests/test_algo.py`，注意 fixture（`tests/conftest.py`、`tests/messages.py`、`tests/fake_chat.py`）怎么造图与假模型。
- ✅ 验证/产出：能向别人解释一条 fixture 链路；会用 `pytest --pdb` 在任意断言处停下查状态。

### Session D29 ｜ 用 codegraph 做定向"源码考古"

- 📖 动作：选一个 D16 标记的模糊点，用 `codegraph explore "<你的问题>"` 反向验证你的 v2 架构图；修正笔记。
- ✅ 验证/产出：笔记定稿 v3（架构图 + 3 个深挖专题）。

### Session D30 ｜ 第一个贡献候选

- 📖 动作：在 `openspec/` 看项目变更管理方式；跑 `make format && make lint && make test` 三连建立本地基线。
- ✅ 验证/产出：本地基线全绿；从文档/issue 中圈定 1 个"良好首改"（补 docstring、补测试覆盖、修边界 bug）。

### Session D31+ ｜ 走完一次完整 PR 流程

- 📖 动作：实现改动 → 加测试 → `make format` → `make lint` → `TEST=<你的改动相关测试> make test` → 自查 diff。
- ✅ 验证/产出：提交 PR（若目标仓库接受外部贡献），或至少产出一份可评审的 diff 自评清单。

---

## 时间弹性路线（可裁剪）

| 路线 | 覆盖 | 总时长 |
|---|---|---|
| 🚀 速成（只用） | D1–D7 全部 + D8/D11/D12 通读 | ~2 周 |
| 🔬 标准（读懂+能改） | 全文 D1–D27 | ~6 周 |
| 🏗️ 深入（贡献者） | 全文 D1–D31 | ~8 周 |

压缩技巧：D9/D13/D15 各砍 1/3 时间；D21/D27/D30 三份作业必做（它们决定你是"读过"还是"会了"）。

---

## 进度跟踪表（复制到 `study-lab/NOTES.md` 使用）

| Session | 完成日 | 验证命令通过? | 产出物 | 自评(1-5) | 遗留疑问 |
|---|---|---|---|---|---|
| D1 最小图 |  |  | s01_min_graph.py |  |  |
| D2 消息态 |  |  | s02_messages.py |  |  |
| D3 fanout |  |  | s03_fanout.py |  |  |
| D4 持久化 |  |  | s04_checkpoint.py |  |  |
| D5 interrupt |  |  | s05_approval.py |  |  |
| D6 流式 |  |  | s06_stream.py |  |  |
| D7 ReAct |  |  | s07_react.py + NOTES |  |  |
| D8 入口链路 |  |  | 调用链 v1 |  |  |
| D9 编译产物 |  |  | quiz 答案 |  |  |
| D10 channels |  |  | 数据流图 |  |  |
| D11 主循环上 |  |  | superstep 一句话 |  |  |
| D12 主循环下 |  |  | 架构图 v2 |  |  |
| D13 调度 |  |  | 并发结论 |  |  |
| D14 重试 |  |  | demo |  |  |
| D15 回放 |  |  | demo |  |  |
| D16 复盘 |  |  | v2 讲解 |  |  |
| D17 checkpoint 接口 |  |  | 边界图 |  |  |
| D18 引擎交互 |  |  | quiz 答案 |  |  |
| D19 真实后端 |  |  | Redis 清单 |  |  |
| D20 Store |  |  | 一句话区分 |  |  |
| D21 自定义 saver |  |  | 代码+lint |  |  |
| D22-23 prebuilt |  |  | 差异清单 |  |  |
| D24 cli |  |  | 本地服务 |  |  |
| D25 sdk-py |  |  | quiz 答案 |  |  |
| D26 sdk-js |  |  | 对齐说明 |  |  |
| D27 复刻 agent |  |  | 代码 |  |  |
| D28 读测试 |  |  | fixture 链路 |  |  |
| D29 源码考古 |  |  | v3 笔记 |  |  |
| D30 首改候选 |  |  | 基线全绿 |  |  |
| D31 PR |  |  | PR/自评 |  |  |

---

## 常见坑清单（先看，少踩）

1. **状态合并**：`{"total": 1}` 是"增量"还是"覆盖"取决于 reducer，别靠猜；先看 channel 类型。
2. **checkpoint 不等于状态对象**：存的是"版本化增量+next+元数据"，fork 回放靠它。
3. **stream 的 mode 混用**：`values` 给"快照"，`updates` 给"本次变化"，别在错误 mode 里找你要的字段。
4. **在 lib 目录里跑测试**：`uv run pytest` 而非裸 `pytest`，否则可能 import 到已安装的旧版 langgraph。
5. **并发顺序别做假设**：fan-out 后谁先完成不确定，靠 reducer 收敛才安全。
6. **改代码前先 `make lint`**：仓库 lint 严格（ruff + ty 类型检查），提交前跑 `make format && make lint && make test`。
