"""3단계-B: "같은 문제, 다른 오답" 학습 세트 (문항 일치 설계)

아이디어
  모든 조건이 완전히 같은 문항(같은 그림·질문·과목·지문)을 같은 순서로 쓰고,
  붙이는 오답 하나만 바꿉니다. 그래서 조건 간 차이는 오직 '오답 난이도'입니다.

  - 대상 문항: 오답이 3개 이상인 train 문항 (선택지 K >= 4)
  - Hard : 그 문항의 오답 중 m이 가장 작은 것 (모델이 가장 헷갈린 오답)
  - Mid  : 그 문항의 오답 중 m이 중앙값에 가장 가까운 것
  - Easy : 그 문항의 오답 중 m이 가장 큰 것
  - Rand : 그 문항의 오답 중 무작위 1개 (대조군)
  - Full : 원래 선택지 전체 (대조군)

만드는 파일
  trainsets/{model}_matched/{cond}_N{n}_s{seed}.jsonl
  logs/03b_matched_{model}.log, logs/03b_checks_{model}.json
  reports/matched_report.md, reports/fig5_matched_difficulty.png

실행: python scripts/03b_build_matched.py --model smolvlm   (GPU 불필요, 수 분)
"""
import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, LETTERS, LOGS, ROOT, TRAINSETS, Logger, load_config, save_json

DEFAULTS = {"min_K": 4, "conditions": ["Hard", "Mid", "Easy", "Rand", "Full"],
            "sizes": [100, 200, 400, 800, 1200], "n_max": 1200}
COLORS = {"Hard": "#104281", "Mid": "#2a78d6", "Easy": "#86b6ef", "Rand": "#8a8984"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def ensure_config(cfg, log):
    """config.yaml에 matched 설정이 없으면 기본값을 기록합니다(재현성)."""
    if "matched" in cfg:
        return {**DEFAULTS, **cfg["matched"]}
    path = os.path.join(ROOT, "config.yaml")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n# ---- 3단계-B: 같은 문제, 다른 오답 (문항 일치 설계) ----\n"
                "matched:\n"
                f"  min_K: {DEFAULTS['min_K']}            # 오답이 3개 이상인 문항만\n"
                f"  conditions: {DEFAULTS['conditions']}\n"
                f"  sizes: {DEFAULTS['sizes']}\n"
                f"  n_max: {DEFAULTS['n_max']}\n")
    log("[설정] config.yaml에 matched 설정 기록")
    return dict(DEFAULTS)


def slot_sequence(p, n_max):
    strata = sorted(p)
    assigned, seq = Counter(), []
    for t in range(1, n_max + 1):
        h = max(strata, key=lambda h: (p[h] * t - assigned[h], -strata.index(h)))
        assigned[h] += 1
        seq.append(h)
    return seq


def position_sequence(seed, n_max):
    rng = np.random.default_rng([seed, 777])  # 3단계와 같은 규칙
    pos = []
    for _ in range(n_max // 2):
        pos += ["A", "B"] if rng.random() < 0.5 else ["B", "A"]
    return pos


def pick_distractors(g):
    """한 문항의 오답들(g)에서 Hard / Mid / Easy를 고릅니다. 동점이면 pair_id 순서로.
    오답이 2개뿐이면 Mid는 None."""
    g = g.sort_values(["m", "pair_id"])
    hard, easy = g.iloc[0], g.iloc[-1]
    middle = g.iloc[1:-1]
    if len(middle) == 0:
        return hard, None, easy
    med = g.m.median()
    mid = middle.iloc[(middle.m - med).abs().argsort().iloc[0]]
    return hard, mid, easy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvlm")
    args = ap.parse_args()
    cfg = load_config()
    log = Logger(f"03b_matched_{args.model}.log")
    mc = ensure_config(cfg, log)
    if mc["min_K"] < 4 and "Mid" in mc["conditions"]:
        raise SystemExit("min_K가 4보다 작으면 오답이 2개인 문항이 있어 Mid를 만들 수 없습니다. "
                         "config.yaml의 matched.conditions에서 Mid를 빼세요.")
    n_max, sizes, conds, seeds = mc["n_max"], mc["sizes"], mc["conditions"], cfg["seeds"]["train"]

    items = pd.read_parquet(f"{DATA}/items.parquet")
    items = items[(items.split == "train") & (items.K >= mc["min_K"])].copy()
    items["choices"] = items["choices"].apply(list)
    scored = pd.read_parquet(f"{DATA}/pairs_scored_{args.model}.parquet")
    scored = scored[scored.item_id.isin(set(items.item_id))]
    log(f"[대상] 선택지 {mc['min_K']}개 이상 train 문항 {len(items)}개, 오답 쌍 {len(scored)}개")
    if len(items) < n_max:
        raise SystemExit(f"대상 문항 {len(items)} < n_max {n_max}")

    # 문항별 Hard / Mid / Easy 오답 미리 결정 (시드와 무관하게 고정)
    picks = {}
    for iid, g in scored.groupby("item_id"):
        h, m, e = pick_distractors(g)
        picks[iid] = {"Hard": h, "Mid": m, "Easy": e, "all": g.sort_values("pair_id")}
    gap = pd.Series({i: p["Easy"].m - p["Hard"].m for i, p in picks.items()})
    log(f"[문항 안 난이도 차이] Easy m − Hard m: 평균 {gap.mean():.3f}, 중앙값 {gap.median():.3f}, "
        f"최소 {gap.min():.3f}")

    strata = sorted(items.stratum.unique())
    p = items.stratum.value_counts(normalize=True).to_dict()
    slots = slot_sequence(p, n_max)
    item = items.set_index("item_id")
    out_dir = os.path.join(TRAINSETS, f"{args.model}_matched")
    os.makedirs(out_dir, exist_ok=True)

    files = {}
    for s in seeds:
        # 1) 문항 순서: 시드마다 한 번만 정하고 모든 조건이 그대로 공유
        pools = {}
        for h in strata:
            ids = sorted(items[items.stratum == h].item_id)
            np.random.default_rng([s, 100, strata.index(h)]).shuffle(ids)
            pools[h] = ids
        order = [pools[h].pop(0) for h in slots]
        positions = position_sequence(s, n_max)
        rand_rng = np.random.default_rng([s, 110])
        rand_pick = {iid: picks[iid]["all"].iloc[int(rand_rng.integers(len(picks[iid]["all"])))]
                     for iid in order}

        # 2) 조건별로 오답만 바꿔서 기록
        for c in conds:
            recs = []
            for t, iid in enumerate(order, start=1):
                r = item.loc[iid]
                base = {"item_id": iid, "question": r.question, "hint": r.hint,
                        "stratum": r.stratum, "K": int(r.K), "slot": t}
                if c == "Full":
                    recs.append({**base, "pair_id": None, "options": list(r.choices),
                                 "answer_letter": LETTERS[int(r.answer)], "m": None, "bin": None,
                                 "position": int(r.answer) + 1})
                    continue
                pr = rand_pick[iid] if c == "Rand" else picks[iid][c]
                correct, wrong = r.choices[int(r.answer)], r.choices[int(pr.j)]
                ans = positions[t - 1]
                recs.append({**base, "pair_id": pr.pair_id,
                             "options": [correct, wrong] if ans == "A" else [wrong, correct],
                             "answer_letter": ans, "m": round(float(pr.m), 4), "bin": int(pr.bin),
                             "position": 1 if ans == "A" else 2})
            for n in sizes:
                files[(c, n, s)] = recs[:n]
                with open(os.path.join(out_dir, f"{c}_N{n}_s{s}.jsonl"), "w", encoding="utf-8") as f:
                    for rr in recs[:n]:
                        f.write(json.dumps(rr, ensure_ascii=False) + "\n")
    log(f"[생성] {len(files)}개 파일 → {out_dir}")

    # ---------------- 자동 검증
    res = {}
    for k, recs in files.items():
        ids = [r["item_id"] for r in recs]
        assert len(ids) == len(set(ids)), f"[검증1 실패] {k} 중복"
    res["1_중복없음"] = "통과"
    for c in conds:
        for s in seeds:
            big = [r["item_id"] for r in files[(c, max(sizes), s)]]
            for n in sizes:
                assert [r["item_id"] for r in files[(c, n, s)]] == big[:n], f"[검증2 실패] {c} s{s} N{n}"
    res["2_중첩성"] = "통과"
    for (c, n, s), recs in files.items():
        if c != "Full":
            ratio = sum(r["answer_letter"] == "A" for r in recs) / len(recs)
            assert ratio == 0.5, f"[검증3 실패] {c} N{n} s{s} A 비율 {ratio}"
    res["3_정답위치균형"] = "통과"
    for n in sizes:
        for s in seeds:
            ref = [r["item_id"] for r in files[(conds[0], n, s)]]
            for c in conds[1:]:
                assert [r["item_id"] for r in files[(c, n, s)]] == ref, f"[검증4 실패] {c} N{n} s{s} 문항 불일치"
    res["4_모든조건_문항일치"] = "통과 (과목·지문·그림·학년 완전 동일)"
    ties = 0
    levels = [c for c in ["Hard", "Mid", "Easy"] if c in conds]
    for s in seeds:
        for t in range(n_max):
            ms = [files[(c, max(sizes), s)][t]["m"] for c in levels]
            assert all(a <= b for a, b in zip(ms, ms[1:])), f"[검증5 실패] s{s} 슬롯{t + 1} m 순서 {ms}"
            ties += sum(a == b for a, b in zip(ms, ms[1:]))
    res["5_문항안_난이도순서"] = f"통과 (모든 문항에서 {' ≤ '.join(levels)}, 동점 {ties}건)"
    for k, v in res.items():
        log(f"[검증 {k}] {v}")

    # ---------------- 요약 보고
    s0 = seeds[0]
    rows, data = [], []
    shown = [c for c in ["Hard", "Mid", "Easy", "Rand"] if c in conds]
    for c in shown:
        ms = np.array([r["m"] for r in files[(c, n_max, s0)]])
        bins = Counter(r["bin"] for r in files[(c, n_max, s0)])
        data.append(ms)
        rows.append([c, f"{ms.mean():.3f}", f"{np.median(ms):.3f}",
                     " / ".join(f"{bins.get(k, 0) / len(ms):.0%}" for k in range(1, 6))])
    sub = Counter(r["stratum"].split("|")[0] for r in files[(conds[0], n_max, s0)])
    hint = np.mean([bool(str(r["hint"]).strip()) for r in files[(conds[0], n_max, s0)]])

    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bp = ax.boxplot(data, tick_labels=shown, patch_artist=True,
                    showfliers=False, widths=0.55, medianprops={"color": SURFACE, "linewidth": 2})
    for patch, c in zip(bp["boxes"], shown):
        patch.set_facecolor(COLORS[c]); patch.set_edgecolor(SURFACE)
    for el in ("whiskers", "caps"):
        for line in bp[el]:
            line.set_color(INK2)
    ax.axhline(0, color=INK, linewidth=1)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"Matched design: same {n_max} items, only the distractor differs (seed {s0})",
                 loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("m  (lower = harder)", color=INK2)
    rep_dir = os.path.join(ROOT, "reports")
    os.makedirs(rep_dir, exist_ok=True)
    fig.tight_layout(); fig.savefig(f"{rep_dir}/fig5_matched_difficulty.png", dpi=160); plt.close(fig)

    md = [f"# 문항 일치 설계 (같은 문제, 다른 오답) — {args.model}", "",
          "모든 조건이 **완전히 같은 문항을 같은 순서로** 쓰고, 붙이는 오답 하나만 다르다. "
          "따라서 과목·지문·그림·학년·선택지 수는 조건 간에 동일하다.", "",
          f"- 대상: 선택지 {mc['min_K']}개 이상 train 문항 {len(items):,}개",
          f"- 조건 {len(conds)} × 규모 {len(sizes)} × 시드 {len(seeds)} = {len(files)}개 파일",
          f"- 문항 안 난이도 차이(Easy m − Hard m): 평균 {gap.mean():.3f}, 중앙값 {gap.median():.3f}",
          f"- 과목 구성(모든 조건 동일, N={n_max}): " + ", ".join(f"{k} {v / n_max:.0%}" for k, v in sub.most_common()),
          f"- 지문(hint) 있는 문항 비율(모든 조건 동일): {hint:.0%}", "",
          f"## 조건별 오답 난이도 (N={n_max}, seed {s0})",
          "| 조건 | 평균 m | 중앙값 m | 전체 기준 구간 D1/D2/D3/D4/D5 비율 |", "|---|---|---|---|"]
    md += [f"| {' | '.join(r)} |" for r in rows]
    md += ["", "![문항 일치 설계 난이도](fig5_matched_difficulty.png)", "", "## 자동 검증", "| 검증 | 결과 |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in res.items()]
    md += ["", "> 해석 주의: 여기서 Hard/Easy는 '같은 문항 안에서 상대적으로' 어려운/쉬운 오답이다. "
           "전체 기준 구간(D1~D5)과는 다르며, 위 표의 구간 비율로 두 기준의 관계를 확인할 수 있다."]
    with open(f"{rep_dir}/matched_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    save_json({"checks": res, "n_files": len(files), "n_items": len(items),
               "gap_mean": float(gap.mean()), "settings": mc},
              os.path.join(LOGS, f"03b_checks_{args.model}.json"))
    log("==> 모든 검증 통과. reports/matched_report.md 생성")


if __name__ == "__main__":
    main()
