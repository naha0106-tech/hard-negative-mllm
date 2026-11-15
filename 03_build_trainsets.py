"""3단계: 조건 × 규모 × 시드별 학습 세트(.jsonl) 생성 + 자동 검증 5개

만드는 파일
  trainsets/{model}/{cond}_N{n}_s{seed}.jsonl
  logs/03_build_{model}.log         (층 대체 횟수 등)
  logs/03_checks_{model}.json       (검증 결과)

실행: python scripts/03_build_trainsets.py --model smolvlm
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from common import DATA, LETTERS, TRAINSETS, Logger, load_config, save_json, LOGS

# Rand와 Full은 같은 문항·같은 순서를 쓰도록 같은 번호(0)를 줍니다.
POOL_CODE = {"Rand": 0, "Full": 0, "D1": 1, "D2": 2, "D3": 3, "D4": 4, "D5": 5}
PICK_CODE = {"Rand": 10, "Full": 10, "D1": 11, "D2": 12, "D3": 13, "D4": 14, "D5": 15}


def slot_sequence(p, n_max):
    """슬롯 1~n_max마다 '이 자리는 어느 층 문항 자리인지'를 정합니다.
    매 자리마다 (목표 비율 × t − 지금까지 배정 수)가 가장 큰 층을 고릅니다."""
    strata = sorted(p)
    assigned = Counter()
    seq = []
    for t in range(1, n_max + 1):
        h = max(strata, key=lambda h: (p[h] * t - assigned[h], -strata.index(h)))
        assigned[h] += 1
        seq.append(h)
    return seq


def position_sequence(seed, n_max):
    """두 슬롯씩 묶어 (A,B) 또는 (B,A)로 정답 위치를 정합니다 → 짝수 N에서 항상 50:50.
    조건과 관계없이 시드마다 같은 순서를 씁니다."""
    rng = np.random.default_rng([seed, 777])
    pos = []
    for _ in range(n_max // 2):
        pos += ["A", "B"] if rng.random() < 0.5 else ["B", "A"]
    return pos


def build_master(cond, seed, slots, positions, items, scored, strata, log):
    """길이 n_max짜리 마스터 시퀀스를 만듭니다. 이걸 앞에서부터 잘라 N별 파일을 만듭니다."""
    n_max = len(slots)
    if cond.startswith("D"):
        k = int(cond[1])
        cand_pairs = scored[scored.bin == k]
    else:
        cand_pairs = scored
    cand_items = items[items.item_id.isin(set(cand_pairs.item_id))]
    if len(cand_items) < n_max:
        raise SystemExit(f"{cond}: 후보 문항 {len(cand_items)}개 < n_max {n_max}. "
                         f"config.yaml의 n_max를 낮추세요(관문 A4).")

    pools = {}
    for h in strata:
        ids = sorted(cand_items[cand_items.stratum == h].item_id)
        rng = np.random.default_rng([seed, POOL_CODE[cond], strata.index(h)])
        rng.shuffle(ids)
        pools[h] = ids

    pairs_by_item = {i: g.sort_values("pair_id") for i, g in cand_pairs.groupby("item_id")}
    item = items.set_index("item_id")
    pick_rng = np.random.default_rng([seed, PICK_CODE[cond]])
    p = {h: (items.stratum == h).mean() for h in strata}

    used = Counter()
    records, n_sub = [], 0
    for t, h in enumerate(slots, start=1):
        if not pools[h]:  # 층 고갈 → 가장 부족한 다른 층에서 대체
            alt = [x for x in strata if pools[x]]
            h2 = max(alt, key=lambda x: p[x] * t - used[x])
            log(f"  [대체] {cond} s{seed} 슬롯{t}: {h} 고갈 → {h2}")
            h, n_sub = h2, n_sub + 1
        iid = pools[h].pop(0)
        used[h] += 1
        r = item.loc[iid]
        choices = list(r.choices)
        base = {"item_id": iid, "question": r.question, "hint": r.hint,
                "stratum": r.stratum, "K": int(r.K), "slot": t, "substituted": h != slots[t - 1]}

        if cond == "Full":
            records.append({**base, "pair_id": None, "options": choices,
                            "answer_letter": LETTERS[int(r.answer)],
                            "m": None, "bin": None, "position": int(r.answer) + 1})
            continue

        g = pairs_by_item[iid]
        prow = g.iloc[int(pick_rng.integers(len(g)))]
        correct, wrong = choices[int(r.answer)], choices[int(prow.j)]
        ans = positions[t - 1]
        opts = [correct, wrong] if ans == "A" else [wrong, correct]
        records.append({**base, "pair_id": prow.pair_id, "options": opts, "answer_letter": ans,
                        "m": round(float(prow.m), 4), "bin": int(prow.bin),
                        "position": 1 if ans == "A" else 2})
    return records, n_sub


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_checks(files, slots, sizes, conds, seeds, log):
    """계획서 5장의 자동 검증 5개. 하나라도 실패하면 멈춥니다."""
    res = {}

    # 1. 파일 안 문항 중복 0
    for key, recs in files.items():
        ids = [r["item_id"] for r in recs]
        assert len(ids) == len(set(ids)), f"[검증1 실패] {key} 문항 중복"
    res["1_중복없음"] = "통과"

    # 2. 중첩성: 작은 N이 큰 N의 앞부분과 똑같은지
    for c in conds:
        for s in seeds:
            big = [r["item_id"] for r in files[(c, max(sizes), s)]]
            for n in sizes:
                assert [r["item_id"] for r in files[(c, n, s)]] == big[:n], \
                    f"[검증2 실패] {c} s{s} N{n} 중첩 깨짐"
    res["2_중첩성"] = "통과"

    # 3. 정답 위치 50:50 (2지선다 조건)
    for (c, n, s), recs in files.items():
        if c == "Full":
            continue
        ratio = sum(r["answer_letter"] == "A" for r in recs) / len(recs)
        assert ratio == 0.5, f"[검증3 실패] {c} N{n} s{s} A 비율 {ratio}"
    res["3_정답위치균형"] = "통과"

    # 4. 층 비율: 목표(슬롯 순서) 대비 ±1. 층 대체가 있었던 파일은 경고만.
    warn4 = []
    for (c, n, s), recs in files.items():
        target = Counter(slots[:n])
        got = Counter(r["stratum"] for r in recs)
        ok = all(abs(got[h] - target[h]) <= 1 for h in set(target) | set(got))
        if not ok:
            if any(r["substituted"] for r in recs):
                warn4.append(f"{c}_N{n}_s{s}")
            else:
                raise AssertionError(f"[검증4 실패] {c} N{n} s{s} 층 비율 {dict(got)} vs {dict(target)}")
    res["4_층비율"] = "통과" if not warn4 else {"경고(층 대체 때문)": warn4}

    # 5. 난이도 순서: D1 < D2 < D3 < D4 < D5 (평균 m), Rand 중앙값은 D2~D4 사이인지 확인
    warn5 = []
    dconds = [c for c in ["D1", "D2", "D3", "D4", "D5"] if c in conds]
    for n in sizes:
        for s in seeds:
            means = [np.mean([r["m"] for r in files[(c, n, s)]]) for c in dconds]
            assert all(a < b for a, b in zip(means, means[1:])), \
                f"[검증5 실패] N{n} s{s} 평균 m 순서 {dict(zip(dconds, means))}"
            if "Rand" in conds and {"D2", "D4"} <= set(dconds):
                med = lambda c: np.median([r["m"] for r in files[(c, n, s)]])
                if not (med("D2") <= med("Rand") <= med("D4")):
                    warn5.append(f"N{n}_s{s}")
    res["5_난이도순서"] = "통과" if not warn5 else {"통과(단, Rand 중앙값 경고)": warn5}

    for k, v in res.items():
        log(f"[검증 {k}] {v}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["smolvlm", "qwen"], required=True)
    args = ap.parse_args()

    cfg = load_config()
    tcfg = cfg["trainsets"]
    n_max = tcfg["n_max"]
    conds, sizes = tcfg[args.model]["conditions"], tcfg[args.model]["sizes"]
    seeds = cfg["seeds"]["train"]
    log = Logger(f"03_build_{args.model}.log")

    items = pd.read_parquet(f"{DATA}/items.parquet")
    items = items[items.split == "train"].copy()
    items["choices"] = items["choices"].apply(list)
    scored = pd.read_parquet(f"{DATA}/pairs_scored_{args.model}.parquet")

    strata = sorted(items.stratum.unique())
    p = {h: (items.stratum == h).mean() for h in strata}
    slots = slot_sequence(p, n_max)
    log(f"[층 비율] { {h: round(float(v), 4) for h, v in p.items()} }")

    out_dir = os.path.join(TRAINSETS, args.model)
    os.makedirs(out_dir, exist_ok=True)
    files, subs = {}, {}
    for s in seeds:
        positions = position_sequence(s, n_max)
        for c in conds:
            master, n_sub = build_master(c, s, slots, positions, items, scored, strata, log)
            subs[f"{c}_s{s}"] = n_sub
            for n in sizes:
                files[(c, n, s)] = master[:n]
                write_jsonl(master[:n], os.path.join(out_dir, f"{c}_N{n}_s{s}.jsonl"))
    log(f"[생성] {len(files)}개 파일 → {out_dir}")
    log(f"[층 대체 횟수] {subs}")

    res = run_checks(files, slots, sizes, conds, seeds, log)
    save_json({"checks": res, "substitutions": subs, "n_files": len(files)},
              os.path.join(LOGS, f"03_checks_{args.model}.json"))
    if any(v != "통과" for v in res.values()):
        log("==> 검증은 통과했지만 경고가 있습니다. 위 경고를 계획서/논문에 기록하세요.")
    else:
        log("==> 모든 검증 통과. 학습 단계로 넘어가도 됩니다.")


if __name__ == "__main__":
    main()
