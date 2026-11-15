"""4단계: 데이터셋 구축 결과 보고 자료 만들기 (GPU 필요 없음)

만드는 파일 (reports/ 폴더)
  dataset_report.md           보고서 본문 (표 + 그림 연결)
  fig1_m_distribution.png     오답 난이도(m) 분포와 D1~D5 경계
  fig2_items_per_bin.png      구간별 고유 문항 수 (관문 A4)
  fig3_order_consistency.png  순서를 바꿔 잰 m_A vs m_B (관문 A3)
  fig4_trainset_difficulty.png  조건별 학습 세트의 m 분포 (N=1600, seed 0)

실행: python scripts/04_report.py --model smolvlm
그림 글자는 코랩 한글 폰트 문제를 피하려고 영어로 씁니다.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, LOGS, ROOT, TRAINSETS, load_config, load_json

# 순서형 파랑 (D1 = 가장 어려움 = 가장 진함)
BIN_COLORS = {1: "#104281", 2: "#1c5cab", 3: "#2a78d6", 4: "#5598e7", 5: "#86b6ef"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GRAY = "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def fig_m_distribution(scored, q, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    lo, hi = np.percentile(scored.m, [0.5, 99.5])
    edges = np.linspace(lo, hi, 61)
    for k in range(1, 6):
        ax.hist(scored.m[scored.bin == k].clip(lo, hi), bins=edges, color=BIN_COLORS[k],
                edgecolor=SURFACE, linewidth=0.6, label=f"D{k}")
    for v in q:
        ax.axvline(v, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.text(0, ax.get_ylim()[1] * 0.97, " m = 0 (model torn)", color=INK, va="top", fontsize=9,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    ax.set_title("Distractor difficulty (m) across all train pairs")
    ax.set_xlabel("m = log p(correct) - log p(distractor)   (lower = harder distractor)")
    ax.set_ylabel("Number of pairs")
    ax.legend(title="Bin (D1 = hardest)", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_items_per_bin(per_bin, threshold, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ks = sorted(per_bin)
    vals = [per_bin[k] for k in ks]
    bars = ax.bar([f"D{k}" for k in ks], vals, color=[BIN_COLORS[k] for k in ks], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom",
                color=INK, fontsize=9)
    ax.axhline(threshold, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.set_title(f"Unique items per difficulty bin  (dashed = Gate A4 minimum {threshold:,})")
    ax.set_ylabel("Unique items")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_order_consistency(scored, rho, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    s = scored.sample(n=min(4000, len(scored)), random_state=0)
    ax.scatter(s.m_A, s.m_B, s=9, color=BIN_COLORS[3], alpha=0.35, linewidths=0)
    lo = min(s.m_A.min(), s.m_B.min()); hi = max(s.m_A.max(), s.m_B.max())
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_title(f"Order consistency  (Spearman ρ = {rho:.2f})")
    ax.set_xlabel("m_A  (correct shown as A)")
    ax.set_ylabel("m_B  (correct shown as B)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_trainset(model, conds, n, seed, path):
    data, labels = [], []
    for c in conds:
        f = os.path.join(TRAINSETS, model, f"{c}_N{n}_s{seed}.jsonl")
        if c == "Full" or not os.path.exists(f):
            continue
        ms = [json.loads(l)["m"] for l in open(f, encoding="utf-8")]
        data.append(ms); labels.append(c)
    if not data:
        return False
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.55,
                    medianprops={"color": SURFACE, "linewidth": 2})
    for patch, c in zip(bp["boxes"], labels):
        patch.set_facecolor(BIN_COLORS[int(c[1])] if c.startswith("D") else GRAY)
        patch.set_edgecolor(SURFACE)
    for el in ("whiskers", "caps"):
        for line in bp[el]:
            line.set_color(INK2)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_title(f"Distractor difficulty in each training set (N={n}, seed {seed})")
    ax.set_ylabel("m  (lower = harder)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return True


def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvlm")
    args = ap.parse_args()
    cfg = load_config()
    out = os.path.join(ROOT, "reports")
    os.makedirs(out, exist_ok=True)

    items = pd.read_parquet(f"{DATA}/items.parquet")
    scored = pd.read_parquet(f"{DATA}/pairs_scored_{args.model}.parquet")
    rep = load_json(f"{DATA}/difficulty_report_{args.model}.json")
    q = rep["quantiles_q20_q40_q60_q80"]
    per_bin = scored.groupby("bin").item_id.nunique().to_dict()
    rho = rep["checks"]["A3_순서일치"]["value"]
    thr = cfg["gate_a"]["a4_min_items_per_bin"]

    fig_m_distribution(scored, q, f"{out}/fig1_m_distribution.png")
    fig_items_per_bin(per_bin, thr, f"{out}/fig2_items_per_bin.png")
    fig_order_consistency(scored, rho, f"{out}/fig3_order_consistency.png")
    conds = cfg["trainsets"][args.model]["conditions"]
    has_fig4 = fig_trainset(args.model, conds, cfg["trainsets"]["n_max"],
                            cfg["seeds"]["train"][0], f"{out}/fig4_trainset_difficulty.png")

    # ---- 보고서 본문
    tr = items[items.split == "train"]
    split_rows = [[s, int((items.split == s).sum())] for s in ["train", "validation", "test"]]
    k_rows = [[f"K={k}", int(v), f"{v / len(tr):.1%}"] for k, v in tr.K.value_counts().sort_index().items()]
    st_rows = [[h, int(v), f"{v / len(tr):.1%}"] for h, v in tr.stratum.value_counts().items()]
    gate_rows = [[k, "통과" if c["pass"] else "미통과",
                  c["value"] if not isinstance(c["value"], float) else f"{c['value']:.3f}", c["기준"]]
                 for k, c in rep["checks"].items()]
    bin_rows = [[f"D{k}", f"{scored.m[scored.bin == k].min():.3f} ~ {scored.m[scored.bin == k].max():.3f}",
                 f"{scored.m[scored.bin == k].mean():.3f}", int((scored.bin == k).sum()), int(per_bin.get(k, 0))]
                for k in range(1, 6)]

    checks_path = os.path.join(LOGS, f"03_checks_{args.model}.json")
    build = load_json(checks_path) if os.path.exists(checks_path) else None

    audit_txt = "D1 수작업 검수: 추후 진행 예정."
    ap_path = f"{DATA}/d1_audit.csv"
    if os.path.exists(ap_path):
        a = pd.read_csv(ap_path)
        col = [c for c in a.columns if c.startswith("label")]
        if col and a[col[0]].notna().any():
            vc = a[col[0]].dropna().astype(int).value_counts().sort_index()
            names = {1: "정상", 2: "정답 라벨 오류", 3: "이미지 판독 불가", 4: "복수 정답 타당"}
            audit_txt = "D1 수작업 검수 (" + str(int(vc.sum())) + "쌍): " + ", ".join(
                f"{names.get(k, k)} {v}" for k, v in vc.items())

    lines = [
        f"# 데이터셋 구축 결과 보고 ({args.model})", "",
        "## 1. 원본 데이터 (ScienceQA, 이미지 포함 문항)",
        md_table(split_rows, ["split", "문항 수"]), "",
        f"train 오답 쌍: **{len(scored):,}개**", "",
        "### 선택지 개수(K) 분포 (train)", md_table(k_rows, ["K", "문항 수", "비율"]), "",
        "### 과목·학년대 층 분포 (train)", md_table(st_rows, ["층", "문항 수", "비율"]), "",
        "## 2. 오답 난이도 측정",
        f"- 측정 모델: 학습 전 {cfg['models'][args.model]['id']}",
        f"- 정답 글자 형식: `{rep['letter_format']}`",
        f"- val500 정확도: **{rep['val500_acc']:.3f}** (찍기 정확도 {rep['chance_acc']:.3f})",
        "- m = 정답 로그확률 − 오답 로그확률 (정답 위치를 바꾼 두 프롬프트의 평균). 작을수록 어려운 오답.", "",
        "![m 분포](fig1_m_distribution.png)", "",
        md_table(bin_rows, ["구간", "m 범위", "평균 m", "오답 쌍 수", "고유 문항 수"]), "",
        "## 3. 관문 A (측정 타당성)", md_table(gate_rows, ["항목", "결과", "값", "기준"]), "",
        "![구간별 문항 수](fig2_items_per_bin.png)", "",
        "![순서 일치도](fig3_order_consistency.png)", "",
        audit_txt, "",
    ]
    if build:
        lines += ["## 4. 학습 세트 구성",
                  f"- 생성 파일: {build['n_files']}개 "
                  f"(조건 {len(conds)} × 규모 {len(cfg['trainsets'][args.model]['sizes'])} × 시드 {len(cfg['seeds']['train'])})",
                  "- 자동 검증 결과:", "", md_table([[k, v if isinstance(v, str) else "경고 있음 (로그 참고)"]
                                                for k, v in build["checks"].items()], ["검증", "결과"]), "",
                  f"- 층 대체 횟수(조건_시드별): {build['substitutions']}", ""]
        if has_fig4:
            lines += ["![학습 세트 난이도](fig4_trainset_difficulty.png)", ""]
    with open(f"{out}/dataset_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n==> reports/ 폴더에 보고서와 그림 저장 완료")


if __name__ == "__main__":
    main()
