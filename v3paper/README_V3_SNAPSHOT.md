# v3 冻结快照 (Temporal-ZwZ-v3)

**用途**：复现 `run_v3_s225.log` 那一次训练的代码状态。与主线 `Temporal-ZwZ` 完全隔离，可并存、互不影响。

创建于 2026-09-17，对应 v3 实验：
`output/opd-grpo/opd-coarse_box_multi-kl1.0-gw1.0-grpo-spg4-lr1e-6-fixed-all-152-v3paper-s225-20260916-1150`

---

## 目录内容

```
Temporal-ZwZ-v3/
├── zwz/                 # v3 状态的训练代码 (trainer / data / train)
├── scripts/             # v3 状态的启动脚本
│   ├── run_opd_tvg.sh                             # PROJECT_ROOT 已改指向本快照
│   ├── run_opd_tvg_h20_fixed_all_152_v3_paper.sh  # v3 主入口
│   ├── verify_v3_snapshot.py                      # 隔离性/冻结性自检
│   ├── run_opd_tvg_h20_fixed_all_152_v4.sh        # 已 stub 化(不属于 v3)
│   └── probe_teacher_channel.py                   # 已 stub 化(不属于 v3)
├── configs/             # prompt 模板
└── output/              # 训练输出 (运行时自动创建, 与主线隔离)
```

**未复制**：`output/2d_sets/*.jsonl` 标注、视频。这些是只读数据，仍引用主线路径。

---

## 隔离机制

| 项 | 主线 | v3 快照 | 隔离方式 |
|---|---|---|---|
| 训练代码 | `Temporal-ZwZ/zwz/` | `Temporal-ZwZ-v3/zwz/` | 整树复制。`zwz` 是 regular package（有 `__init__.py`），Python 不会跨路径合并同名包，所以 `PYTHONPATH` 里谁在前就完整用谁 |
| `PROJECT_ROOT` | `.../Temporal-ZwZ` | `.../Temporal-ZwZ-v3` | 快照的 `run_opd_tvg.sh` 里已硬改 |
| 训练输出 | `Temporal-ZwZ/output/opd-grpo` | `Temporal-ZwZ-v3/output/opd-grpo` | `output_root="${PROJECT_ROOT}/output/opd-grpo"` 自动跟随 |
| prompt 模板 | 主线 `configs/` | 快照 `configs/` | 启动脚本改为 `${PROJECT_ROOT}/configs/` |
| 标注/视频 | — | 引用主线 | 只读数据，无需隔离 |
| TimeLens 父类 | 共享 | 共享 | 本轮未改动父类，当前状态即 v3 状态（见下方说明） |
| teacher 视频缓存 | `/dockerdata/zhenjunchen/teacher_cache` | 同一目录 | **安全共享**，见下方说明 |

### 关于 TimeLens 父类

`TimeLens/training/trainer/grpo_trainer_qwenvl.py` 由 v3 和主线共享。第 5 轮（v4）**没有改动**它，所以当前状态就是 v3 需要的状态（含 `rewards` / `group_all_zero` / `group_reward_mean` / `group_is_std_zero` 四个返回 key）。

这四个都是**纯新增 key**，对不读它们的旧代码无影响。TimeLens 本身是 git 仓库，`grpo_trainer_qwenvl.py` 显示为 `M`（已修改未提交），可用 `git diff` 查看全部改动。

⚠️ **今后若要改父类，必须先把它也快照化**，否则会同时影响 v3 和主线。

### 关于 teacher 视频缓存

共享是安全的，因为 `cache_key` 里包含了所有影响渲染结果的参数：

```
v3: md5(video_path_gtstart_gtend_strategy_otherspans_ze1.50_bz1.00)
v4: md5(video_path_gtstart_gtend_strategy_otherspans_ze1.50_bz1.00_jt0.40)
```

key 不同 → 不会串味；同 key 必同内容（渲染是纯函数）→ 共享还能省重复渲染时间。

如需彻底隔离，改快照里 `scripts/run_opd_tvg.sh` 的这一行：
```bash
teacher_cache_dir="/dockerdata/zhenjunchen/teacher_cache_v3"
```
代价是首轮要重新渲染全部 teacher 视频。

---

## v3 与主线（v4）的代码差异

快照**故意保留**了 v3 的原始状态，包括那个导致实验失败的 bug：

| 能力 | v3 快照 | 主线 (v4) |
|---|---|---|
| `_repeat_visual_kwargs` | ✗ 无 | ✓ 有 |
| teacher rollout 实现 | `generate(num_return_sequences=nb)` ← **会崩** | 手工复制 batch 行 |
| `zoom_jitter`（非可逆粗提示） | ✗ 无 | ✓ 有 |
| `--max_grad_norm` 透传 | ✗ 无 | ✓ 有 |
| 组级 controller / `blue_zoom_expand` / `all_zero` | ✓ 有 | ✓ 有 |

**重要**：v3 的 teacher rollout 在 8 卡上会全部抛
```
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 1
```
然后回退成 `λ≡1`。这是**预期行为**——`run_v3_s225` 就是这么跑的。因此：

> v3 快照实际等价于 **v1 + 蓝框修正 + 2.5 倍 epoch**，
> 其 controller / teacher 独立 rollout / λ=0 屏蔽三项均不生效。
> 它只用于复现历史结果，**不应作为新实验的基线**。

---

## 用法

### 自检（先跑这个）
```bash
python3 /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v3/scripts/verify_v3_snapshot.py
```
应输出全部 PASS。任何 FAIL 都说明快照被污染，不可信。

### 复现 v3 训练
```bash
source /dockerdata/zhenjunchen/venv/bin/activate
nohup bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v3/scripts/run_opd_tvg_h20_fixed_all_152_v3_paper.sh \
    >> /dockerdata/zhenjunchen/logs/run_v3_replay.log 2>&1 &
```
脚本会自动把 `PYTHONPATH` 设为快照目录，不会加载主线代码。

### 确认真的在用快照代码
启动后检查日志：
```bash
grep -m1 "Output:" /dockerdata/zhenjunchen/logs/run_v3_replay.log
# 应包含 Temporal-ZwZ-v3/output/opd-grpo/
```
以及 `ps` 里的 `--output_dir` 应指向 `Temporal-ZwZ-v3`。

---

## 主线（v4）入口

不要在本快照里跑 v4：

```bash
# 特权通道探测 (纯推理, 先跑)
python3 /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/probe_teacher_channel.py --help

# v4 训练
bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/run_opd_tvg_h20_fixed_all_152_v4.sh
```
