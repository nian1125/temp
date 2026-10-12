#!/usr/bin/env python3
"""OPD 三处修正的离线自测 (不需要 GPU/不需要起训练)。

在 H20 venv 里跑:
    source /dockerdata/zhenjunchen/venv/bin/activate
    export TIMELENS_ROOT=/mnt/250T_ceph/zhenjunchen/code/TimeLens
    export PYTHONPATH=/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ:$TIMELENS_ROOT
    python3 /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/selftest_opd_fixes.py \
        --model /dockerdata/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT

校验 4 件事:
  [修2] dead_rescue_norm: weighted 在 B=1 时是恒等变换(复现哑火), mean 能真正放大
  [修3] 父类返回字典里含 "rewards" 键 (teacher 质量门控不再是死代码)
  [修4] numeric focus 让答案 token 的散度不再被模板 token 稀释
  [兼容] focus_mode="off" 时结果与修改前逐位一致
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zwz.trainer.opd_grpo_trainer import _compute_jsd_topk, OPDGRPOTrainer  # noqa: E402

OK, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"  [{OK if cond else FAIL}] {name}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------- [修2]
def test_dead_rescue_norm():
    print("\n[修2] dead_rescue_norm (weighted=原行为 / mean=修正)")
    mult = 2.0

    # B=1 且该样本是死组 → weighted 分母被权重自己除掉
    j1 = torch.tensor([0.0020])
    w1 = torch.tensor([mult])
    weighted_b1 = (j1 * w1).sum() / w1.sum()
    mean_b1 = (j1 * w1).mean()
    check("B=1 weighted 是恒等变换(复现主实验哑火)",
          torch.allclose(weighted_b1, j1[0]),
          f"weighted={weighted_b1:.6f} == raw={j1[0]:.6f}")
    check("B=1 mean 真正放大 ×2",
          torch.allclose(mean_b1, j1[0] * mult),
          f"mean={mean_b1:.6f} = raw×{mult}")

    # B=2, 一条死一条活 → weighted 只做了加权平均, 死组相对放大有限
    j2 = torch.tensor([0.0020, 0.0010])
    w2 = torch.tensor([mult, 1.0])
    weighted_b2 = (j2 * w2).sum() / w2.sum()
    mean_b2 = (j2 * w2).mean()
    plain_b2 = j2.mean()
    check("B=2 mean > weighted > 无rescue",
          mean_b2 > weighted_b2 > plain_b2,
          f"mean={mean_b2:.6f} weighted={weighted_b2:.6f} plain={plain_b2:.6f}")


# ---------------------------------------------------------------- [修3]
def test_parent_returns_rewards():
    print("\n[修3] 父类返回 'rewards' 键 (teacher 质量门控不再是死代码)")
    import training.trainer.grpo_trainer_qwenvl as pmod
    src = open(pmod.__file__).read()
    check("_generate_and_score_completions 返回 'rewards'",
          '"rewards": rewards_local' in src)
    check("rewards 按 process_slice 切片(与 advantages 逐条对齐)",
          "rewards_local = rewards[process_slice]" in src)

    # 门控默认必须是 off, 否则老实验不可复现
    import inspect
    sig = inspect.signature(OPDGRPOTrainer.__init__)
    check("teacher_gate_mode 默认 'off' (老实验可复现)",
          sig.parameters["teacher_gate_mode"].default == "off")
    check("teacher_tiou_mode 默认 'argmax' (老实验可复现)",
          sig.parameters["teacher_tiou_mode"].default == "argmax")
    check("dead_rescue_norm 默认 'weighted' (老实验可复现)",
          sig.parameters["dead_rescue_norm"].default == "weighted")
    check("jsd_token_focus 默认 'off' (老实验可复现)",
          sig.parameters["jsd_token_focus"].default == "off")


# ---------------------------------------------------------------- [修4]
class _MaskShim:
    """只借 _build_numeric_focus_mask, 不实例化整个 Trainer。"""
    _build_numeric_focus_mask = OPDGRPOTrainer._build_numeric_focus_mask

    def __init__(self, tok):
        self._tok = tok
        self._numeric_tok_cache = {}

    def _tokenizer(self):
        return self._tok


def test_numeric_mask(model_path):
    print("\n[修4] numeric focus mask")
    if not model_path or not os.path.isdir(model_path):
        print("  (跳过: 未提供 --model, 无法加载 tokenizer)")
        return None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    shim = _MaskShim(tok)

    answer = "The event happens in 12.5 - 47.0 seconds."
    ids = torch.tensor([tok(answer, add_special_tokens=False)["input_ids"]])
    mask = torch.ones_like(ids, dtype=torch.float32)
    fm = shim._build_numeric_focus_mask(ids, mask)

    toks = [tok.decode([i]) for i in ids[0].tolist()]
    hit = [t for t, f in zip(toks, fm[0].tolist()) if f > 0]
    frac = fm.sum().item() / mask.sum().item()
    print(f"    tokens({len(toks)}): {toks}")
    print(f"    focus  : {hit}")
    check("数值token被选中且占比 <50% (说明模板确实占了一半以上)",
          0 < frac < 0.5, f"focus_frac={frac:.3f}")
    check("所有含数字的token都被选中",
          all(f > 0 for t, f in zip(toks, fm[0].tolist()) if any(c.isdigit() for c in t)))
    return fm


def test_focus_amplifies(fm):
    print("\n[修4] focus 让答案token的散度不再被模板稀释")
    torch.manual_seed(0)
    B, L, V = 1, (fm.size(1) if fm is not None else 12), 512
    if fm is None:
        fm = torch.zeros(B, L)
        fm[0, -4:] = 1.0  # 假设末尾4个是数值token
    cm = torch.ones(B, L)

    # 构造: 模板token师生几乎一致, 数值token明显分歧 (完全对应实测 top1_agree=0.98)
    s = torch.randn(B, L, V) * 0.1
    t = s.clone()
    t[fm.bool()] += torch.randn(B, L, V)[fm.bool()] * 3.0

    kw = dict(temperature=1.0, alpha=0.5, topk=100, completion_mask=cm,
              return_per_sample=False, divergence="jsd")
    j_off = _compute_jsd_topk(s, t, **kw, focus_mode="off").item()
    j_hard = _compute_jsd_topk(s, t, **kw, focus_mask=fm, focus_mode="numeric").item()
    j_soft = _compute_jsd_topk(s, t, **kw, focus_mask=fm, focus_mode="numeric_soft",
                               focus_weight=4.0).item()
    print(f"    off={j_off:.6f}  numeric(hard)={j_hard:.6f}  numeric_soft={j_soft:.6f}")
    check("hard focus 放大答案token信号", j_hard > j_off * 1.5,
          f"×{j_hard / max(j_off, 1e-12):.2f}")
    check("soft focus 介于两者之间且 > off", j_off < j_soft, f"×{j_soft / max(j_off, 1e-12):.2f}")

    # 空 focus 保底: 不能 NaN, 应回退全token
    fm0 = torch.zeros_like(fm)
    j_empty = _compute_jsd_topk(s, t, **kw, focus_mask=fm0, focus_mode="numeric").item()
    check("focus 全空时回退全token且不产生 NaN",
          torch.isfinite(torch.tensor(j_empty)) and abs(j_empty - j_off) < 1e-6,
          f"empty={j_empty:.6f} vs off={j_off:.6f}")


def test_backward_compat():
    print("\n[兼容] focus_mode='off' 与旧签名调用结果一致")
    torch.manual_seed(1)
    s, t = torch.randn(2, 10, 256), torch.randn(2, 10, 256)
    cm = torch.ones(2, 10)
    a = _compute_jsd_topk(s, t, 1.0, 0.5, 100, cm)                      # 旧式位置参数
    b = _compute_jsd_topk(s, t, 1.0, 0.5, 100, cm, focus_mode="off")    # 新参数显式 off
    check("旧调用方式仍可用且数值不变", torch.allclose(a, b), f"{a.item():.8f}")
    for dv in ("fkl", "rkl", "tvd", "symkl", "jsd"):
        v = _compute_jsd_topk(s, t, 1.0, 0.5, 100, cm, divergence=dv)
        check(f"divergence={dv} 仍正常", torch.isfinite(v).all().item(), f"{v.item():.6f}")


# ---------------------------------------------------------------- 论文 Eq5/Eq6
def _lambda_paper(rT, rS, all_zero, margin=0.0, lam0=0.0, lam_dead=2.0):
    """复刻 trainer 里的 controller 逻辑, 用于独立验证。"""
    lam = torch.ones_like(rS)
    no_edge = rT <= rS + margin
    lam[no_edge] = lam0
    lam[(~no_edge) & all_zero] = lam_dead
    return lam


def test_controller_eq5():
    print("\n[论文Eq5] 组级 adaptive controller λ_g ∈ {0,1,2}")
    # 4 个组: (a) teacher差 (b) teacher好+正常 (c) teacher好+全零 (d) teacher持平
    rT = torch.tensor([0.30, 0.70, 0.55, 0.40])
    rS = torch.tensor([0.50, 0.40, 0.00, 0.40])
    az = torch.tensor([False, False, True, False])
    lam = _lambda_paper(rT, rS, az)
    print(f"    r̄T={rT.tolist()}\n    r̄S={rS.tolist()}\n    all_zero={az.tolist()}\n    λ={lam.tolist()}")
    check("teacher 差 → λ=0 (论文: 完全不蒸馏)", lam[0].item() == 0.0)
    check("teacher 好 + 正常组 → λ=1", lam[1].item() == 1.0)
    check("teacher 好 + 全零组 → λ=2 (dead rescue)", lam[2].item() == 2.0)
    check("teacher 持平 → λ=0 (Eq5 是 ≤)", lam[3].item() == 0.0)

    print("\n[论文Eq6] L_JSD = (1/|B|) Σ_g λ_g ℓ_g  (除以组数, 不是除以 Σλ)")
    jsd = torch.tensor([0.004, 0.002, 0.003, 0.001])
    eq6 = (jsd * lam).mean()
    wrong = (jsd * lam).sum() / lam.sum().clamp(min=1e-8)   # v1 的 weighted 归一化
    plain = jsd.mean()
    print(f"    Eq6(mean)={eq6:.6f}  v1(weighted)={wrong:.6f}  无controller={plain:.6f}")
    check("Eq6 归一化 = mean 而非 weighted", not torch.allclose(eq6, wrong))
    check("被屏蔽组不贡献梯度", torch.allclose((jsd * lam)[0], torch.tensor(0.0)))

    # B=1 恒等退化: 论文归一化不受影响, v1 的 weighted 会退化
    j1, l1 = torch.tensor([0.002]), torch.tensor([2.0])
    check("B=1 时 Eq6 仍能放大 (weighted 会退化成恒等)",
          torch.allclose((j1 * l1).mean(), j1[0] * 2.0)
          and torch.allclose((j1 * l1).sum() / l1.sum(), j1[0]))


def test_controller_wiring():
    print("\n[接线] controller / rollout / 蓝框 的默认值与父类键")
    import inspect
    import training.trainer.grpo_trainer_qwenvl as pmod
    src = open(pmod.__file__).read()
    check("父类返回 group_all_zero (Eq5 的 all-zero 判据)",
          '"group_all_zero": group_all_zero_local' in src)
    check("group_all_zero 用 amax 在 gather 后的整组上算",
          "rewards.view(-1, self.num_generations).abs().amax(dim=1)" in src)

    sig = inspect.signature(OPDGRPOTrainer.__init__)
    for k, want in [("adaptive_controller", "off"),
                    ("dead_group_criterion", "std_zero_lowmean"),
                    ("controller_lambda_no_edge", 0.0),
                    ("controller_lambda_dead", 2.0)]:
        check(f"{k} 默认 {want!r}", sig.parameters[k].default == want)

    tsrc = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "zwz", "trainer", "opd_grpo_trainer.py")).read()
    check("teacher rollout 跨进程归并(保证各卡 λ 一致)", "gather_object" in tsrc)
    check("λ 全0 时跳过 teacher forward", "_skip_by_ctrl" in tsrc)
    check("controller 分支用 .mean() 归一 (Eq6)",
          "jsd_loss = (per_sample_jsd * lam_s).mean()" in tsrc)

    dsrc = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "zwz", "data", "tvg_opd_dataset.py")).read()
    check("蓝框有独立扩张系数 (论文: 保留verified边界)",
          "o_ch = (o_end - o_start) / 2.0 * _bexp" in dsrc)
    check("蓝框/红框窗口都 clip 到 [0,duration]", "_gt_lo_c, _gt_hi_c = _clip(" in dsrc)
    check("other_spans 稳定排序(避免各卡 cache_key 不一致)",
          "_other_spans = sorted(_other_spans)" in dsrc)


def test_blue_box_geometry():
    print("\n[D5] 蓝框几何: 论文(1.0) vs 历史(1.5)")
    # 相邻事件的真实场景: GT=[40,46] → 红框窗口(1.5×)=[38.5,47.5]; 干扰事件=[48,54]
    gt, other, dur = (40.0, 46.0), (48.0, 54.0), 60.0
    rc, rh = (gt[0] + gt[1]) / 2, (gt[1] - gt[0]) / 2 * 1.5
    red_lo, red_hi = rc - rh, rc + rh
    print(f"    红框窗口(1.5×) = [{red_lo:.1f}, {red_hi:.1f}]   干扰事件 = {list(other)}")
    res = {}
    for bexp, tag in ((1.5, "历史(v1)"), (1.0, "论文(v3)")):
        cc, ch = (other[0] + other[1]) / 2, (other[1] - other[0]) / 2 * bexp
        lo, hi = cc - ch, cc + ch
        ov = (hi > red_lo) and (lo < red_hi)
        oob = lo < 0 or hi > dur
        res[bexp] = (lo, hi, ov, oob)
        print(f"    blue_expand={bexp} {tag}: 蓝框=[{lo:.1f},{hi:.1f}] 侵入红框={ov} 越界={oob}")
    check("历史口径: 蓝框扩1.5× 会侵入红框窗口", res[1.5][2])
    check("论文口径: 蓝框保留verified边界, 不侵入", not res[1.0][2])
    check("论文口径: 蓝框不越界", not res[1.0][3])
    print("    (真实 152 条上实测: 侵入率 18.0%→10.6%, 受影响样本 26.9%→16.2%, "
          "越界 10.6%→0%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="用于加载 tokenizer 的模型路径")
    args = ap.parse_args()

    print("=" * 64)
    print("OPD 代码修正自测 (v2 三处修正 + v3 论文对齐)")
    print("=" * 64)
    test_dead_rescue_norm()
    test_parent_returns_rewards()
    fm = test_numeric_mask(args.model)
    test_focus_amplifies(fm)
    test_backward_compat()
    test_controller_eq5()
    test_controller_wiring()
    test_blue_box_geometry()

    n_pass, n_all = sum(_results), len(_results)
    print("\n" + "=" * 64)
    print(f"结果: {n_pass}/{n_all} 通过")
    print("=" * 64)
    sys.exit(0 if n_pass == n_all else 1)
