# v4（真）= v3 + controller 真正生效

与 `Temporal-ZwZ-v3`（冻结快照）和 `Temporal-ZwZ`（探索主线）完全隔离。

`verify_v4.py` 校验 **45/45 通过**，51 个字段默认值与 v3 完全一致。

---

## 相对 v3 只改两处

### [A] controller 的 r̄^T 改用 v1 的估计方式

`--teacher_tiou_mode forced` → 新方法 `_teacher_forced_group_tiou`。

teacher 看特权视频，在 **student 已采样的那条 completion** 上做一次 `no_grad` 前向，逐位置取 argmax 解码成时间区间，与 GT 算 tIoU，再按组聚合成 r̄_g^T。

与 v1 的 `_estimate_teacher_tiou` **语义完全一致**，唯一区别是提前到 shuffle 之前算 —— 因为 controller 需要组级统计，而 shuffle/split 之后凑不出整组。

**不生成** → 没有 KV cache 增长、没有采样随机性、各卡耗时对称 → 不会 hang。

### [B] 修掉 `controller_skip_zero` 的 NCCL 死锁

这是 v4-rollout hang 的**真正原因**，与 generate 无关。

```
[CTRL] step=0 r̄T=0.1585 r̄S=0.2753 λ=0:8/8 λ̄=0.000      ← 本卡 λ 全 0
[rank7] Watchdog caught collective operation timeout:
        WorkNCCL(SeqNum=19, OpType=_ALLGATHER_BASE, NumelIn=1, NumelOut=8)
        ran for 7200006 milliseconds                      ← 2 小时超时
```

JSD 分支内部含两处 `accelerator.gather`（`teacher_tiou_mean` / `teacher_advantage_mean`），而 skip 是**每张卡独立**判断的 → 有的卡 λ 全 0 跳过、有的卡不跳 → 跳过的卡不参与 gather → NCCL 永久失配。`NumelIn=1, NumelOut=8` 正是 gather 一个标量的特征。

修法：skip 决策先做一次 `all_reduce(MAX)` 全卡一致化，只有**所有卡**都 λ=0 才整体跳过。all_reduce 本身无条件执行，不引入新失配；异常时退化为不跳过（宁可多算，不可死锁）。

---

## 为什么放弃 teacher 独立 rollout

两次实测都失败：

| 版本 | 现象 | 根因 |
|---|---|---|
| v3 | `[CTRL]` 只打印 1 次警告，λ≡1 全程 | `generate(num_return_sequences=nb)` 让 HF 扩 batch，但它只 expand `dim0==batch_size` 的 kwargs → `video_grid_thw`(1,3) 扩了、`pixel_values_videos`(N,D) 没扩 → `split_with_sizes` 崩，8 卡静默回退 |
| v4-rollout | rollout 真跑起来了（`[TGEN]` 形状正确、`[CTRL]` 算出 r̄T=0.1585），但 hang 2h 被 watchdog 杀 | ① 8×13667 token 的 prefill + 逐 token 采样太慢；② `λ=0:8/8` 触发 skip，而 skip 分支含 gather → 集合通信失配 |

**v1 的机制本来就不做独立 rollout**：teacher 只在学生已采出的轨迹上做 teacher-forced forward（只算 logits），所以 v1 每步只多一次前向（+12%，37.9s vs 33.7s）。本版回到这个机制。

---

## 已知偏差（建议写进论文 limitation）

teacher-forced argmax 会被 student 的前缀锚定：时间数字是逐位生成的，前缀一旦确定，teacher 只能顺着往下补 → **r̄^T 系统性偏向 r̄^S**，controller 更容易判"teacher 无优势"（λ=0）。

这是 v1 就有的性质，不是本版新引入的。量化参照：

| 估计方式 | teacher | student | 差值 |
|---|---|---|---|
| 独立采样（v4-rollout 实测） | 0.1585 | 0.2753 | **−0.117** |
| argmax 前缀（v3 实测） | 0.49~0.63 | 0.50~0.64 | ≈ −0.01 |

这个差距正是"锚定效应"的量级，说明 forced 估计**偏乐观**。好处是 λ=0 不会过于频繁（否则蒸馏被全关，v4 退化成纯 GRPO）。

---

## 刻意保持不变（单变量对照）

`max_steps=225`、`bpd=2`、`global_batch=32`、`spg=4`、`G=8`、`lr=1e-6`、`zoom_expand=1.5`、`blue_zoom_expand=1.0`、`all_zero` 判据、`margin=0.0`、师生同 prompt、不传 `--dead_rescue_norm`、无 `zoom_jitter`、`max_grad_norm` 默认。

`tvg_opd_dataset.py` / `reward_funcs.py` / `prompt_student_same.txt` 与 v3 **md5 逐字节相同**；`run_opd_tvg.sh` 除 `PROJECT_ROOT`/注释外差异 0 行；`train_opd_tvg.py` 51 个字段默认值全部一致（只改了 `teacher_tiou_mode` 的 help 文本）。

---

## 用法

```bash
# 1. 自检（秒级，应输出 45/45）
python3 /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v4/scripts/verify_v4.py

# 2. 训练
source /dockerdata/zhenjunchen/venv/bin/activate
nohup bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v4/scripts/run_opd_tvg_h20_v4_controller.sh \
    >> /dockerdata/zhenjunchen/logs/run_v4real.log 2>&1 &
```

### 前 10 分钟必须确认

```bash
grep -m1 "TFORCED" /dockerdata/zhenjunchen/logs/run_v4real.log   # 形状正常
grep -c "CTRL"     /dockerdata/zhenjunchen/logs/run_v4real.log   # 应持续增长, 不是 1
grep -m1 "Output:" /dockerdata/zhenjunchen/logs/run_v4real.log   # 应含 Temporal-ZwZ-v4
```

**`[CTRL]` 的计数是关键**：v3 停在 1（controller 没生效），v4 应该每个 generation block 都打印一次。

若 `λ=0` 的比例长期 > 0.8，说明 teacher 几乎总被判无优势 → 蒸馏基本被关掉，此时 v4 ≈ 纯 GRPO，需要调 `--teacher_advantage_margin` 为负值（如 −0.05）或重新审视特权通道是否打开。

**开销预期**：每个 block 多一次 teacher-forced 前向（本卡 8 条，no_grad），约 +12%，预计 ~42s/步，远低于 v3-rollout 的 132.5s/步。

---

## 三条线

| 目录 | 定位 | 入口 |
|---|---|---|
| `Temporal-ZwZ-v3` | 冻结快照，复现历史（controller 不生效） | `scripts/run_opd_tvg_h20_fixed_all_152_v3_paper.sh` |
| `Temporal-ZwZ-v4` | **本目录**：v3 + controller 真正生效 | `scripts/run_opd_tvg_h20_v4_controller.sh` |
| `Temporal-ZwZ` | 探索主线（zoom_jitter / max_grad_norm / probe 脚本） | `scripts/run_opd_tvg_h20_fixed_all_152_v4.sh` |

共享且安全：TimeLens 父类（本轮未改）、teacher 视频缓存（cache_key 含全部渲染参数）、标注/视频（只读）。

`_teacher_generate_tiou`（独立 rollout 实现）代码**仍保留**在 trainer 里，可用 `--teacher_tiou_mode generate` 调用，但已实测会 hang，仅作存档对照。
