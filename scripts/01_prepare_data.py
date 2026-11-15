"""1단계: 원본 데이터 수집 및 정제

만드는 파일
  data/items.parquet        ScienceQA-IMG 전체 문항 표 (이미지 제외)
  data/pairs.parquet        train 문항의 (정답, 오답) 쌍 표
  data/val500_ids.json      검증 500문항 ID
  data/ai2d1000_ids.json    AI2D 1,000문항 행 번호
  data/pope1000_ids.json    POPE 1,000문항 행 번호
  logs/01_prepare.log       걸러낸 문항 수 등 기록

실행: python scripts/01_prepare_data.py
"""
import numpy as np
import pandas as pd
from datasets import Image

from common import (DATA, Logger, grade_num, largest_remainder, load_config,
                    load_scienceqa, save_json)


def build_items(cfg, log):
    rows = []
    for split in ["train", "validation", "test"]:
        ds = load_scienceqa(cfg, split)
        # 이미지 열은 빼고 읽습니다(빠르고 가벼움). 이미지는 나중에 row_idx로 다시 꺼냅니다.
        cols = [c for c in ds.column_names if c != "image"]
        meta = ds.select_columns(cols).to_pandas()
        # 이미지 유무만 빠르게 확인 (그림을 실제로 열지 않음)
        raw_img = ds.select_columns(["image"]).cast_column("image", Image(decode=False))["image"]
        has_image = [x is not None and (x.get("bytes") or x.get("path")) for x in raw_img]
        log(f"[{split}] 원본 {len(meta)}문항 (이미지 있는 문항 {sum(bool(h) for h in has_image)})")

        removed = {"no_image": 0, "bad_answer": 0, "dup_choices": 0}
        for row_idx, r in meta.iterrows():
            choices = list(r["choices"])
            K = len(choices)
            ans = int(r["answer"])
            if not has_image[row_idx]:
                removed["no_image"] += 1
                continue
            if not (0 <= ans < K):
                removed["bad_answer"] += 1
                continue
            if len(set(choices)) != K:
                removed["dup_choices"] += 1
                continue
            g = grade_num(r["grade"])
            band = "G1-6" if g is not None and g <= 6 else "G7-12"
            hint = r.get("hint", "") or ""
            rows.append({
                "item_id": f"{split}_{row_idx:05d}",
                "split": split,
                "row_idx": int(row_idx),
                "question": r["question"],
                "hint": hint,
                "choices": choices,
                "answer": ans,
                "K": K,
                "subject": r["subject"],
                "grade": r["grade"],
                "grade_num": g,
                "grade_band": band,
                "stratum": f"{r['subject']}|{band}",
                "has_hint": bool(str(hint).strip()),
            })
        log(f"[{split}] 제거: {removed}")
    items = pd.DataFrame(rows)
    for split in ["train", "validation", "test"]:
        log(f"[{split}] 남은 문항: {(items.split == split).sum()} "
            f"(정상이면 약 {dict(train=6218, validation=2097, test=2017)[split]})")
    return items


def build_pairs(items):
    rows = []
    for r in items[items.split == "train"].itertuples():
        for j in range(r.K):
            if j == r.answer:
                continue
            rows.append({"pair_id": f"{r.item_id}_d{j}", "item_id": r.item_id,
                         "j": j, "stratum": r.stratum, "K": r.K})
    return pd.DataFrame(rows)


def sample_val500(items, cfg):
    val = items[items.split == "validation"]
    alloc = largest_remainder(val.stratum.value_counts().to_dict(), cfg["val_n"])
    rng = np.random.default_rng(cfg["seeds"]["val500"])
    picked = []
    for h in sorted(alloc):
        ids = sorted(val[val.stratum == h].item_id)
        picked += list(rng.choice(ids, size=alloc[h], replace=False))
    return sorted(picked), alloc


def sample_aux(cfg, log):
    from datasets import load_dataset

    seed = cfg["seeds"]["aux_eval"]

    a = cfg["datasets"]["ai2d"]
    ai2d = load_dataset(a["repo"], split=a["split"])
    rng = np.random.default_rng(seed)
    idx = sorted(int(i) for i in rng.choice(len(ai2d), size=a["n"], replace=False))
    log(f"[AI2D] 전체 {len(ai2d)}문항 중 {len(idx)}개 추출. 열 이름: {ai2d.column_names}")
    log(f"[AI2D] answer 예시(정답 표기 방식 확인용): {ai2d.select_columns(['answer'])[:3]}")
    save_json({"repo": a["repo"], "split": a["split"], "row_idx": idx},
              f"{DATA}/ai2d1000_ids.json")

    p = cfg["datasets"]["pope"]
    pope = load_dataset(p["repo"], split=p["split"])
    cats = np.array(pope.select_columns(["category"])["category"])
    rng = np.random.default_rng(seed)
    out = {}
    for c in p["categories"]:
        pool = np.where(cats == c)[0]
        out[c] = sorted(int(i) for i in rng.choice(pool, size=p["n_per_category"], replace=False))
        log(f"[POPE] {c}: {len(pool)}개 중 {len(out[c])}개 추출")
    save_json({"repo": p["repo"], "split": p["split"], "row_idx": out},
              f"{DATA}/pope1000_ids.json")


def main():
    cfg = load_config()
    log = Logger("01_prepare.log")

    items = build_items(cfg, log)
    items.to_parquet(f"{DATA}/items.parquet", index=False)

    pairs = build_pairs(items)
    pairs.to_parquet(f"{DATA}/pairs.parquet", index=False)
    train = items[items.split == "train"]
    log(f"[pairs] 오답 쌍 {len(pairs)}개 (train {len(train)}문항)")
    log(f"[K 분포] {train.K.value_counts().sort_index().to_dict()}")
    log(f"[train 층 분포] {train.stratum.value_counts().to_dict()}")

    val_ids, alloc = sample_val500(items, cfg)
    save_json(val_ids, f"{DATA}/val500_ids.json")
    log(f"[val500] 층별 배정: {alloc}")

    sample_aux(cfg, log)
    log("완료")


if __name__ == "__main__":
    main()
