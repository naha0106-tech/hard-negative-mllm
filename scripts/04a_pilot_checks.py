"""4단계 P0: 학습 전 점검 C1~C8 (파일럿 실행 가이드 6장)

C1 채점 재현     무학습 val500 정확도가 2단계 값(0.642)과 같은가
C2 실제 출력     val500 앞 20문항을 생성시켜 무엇으로 답을 시작하는지 기록
C3 길이          학습 세트(Rand·Hard·Easy·Full N=1200) + val500의 최대 토큰 길이 < max_length
C4 마스킹        손실에 들어가는 토큰이 " X<end_of_utterance>"뿐인가 (2지 A·B + Full의 C·D 포함)
C5 글자 토큰     학습 타깃의 글자 토큰 = 채점에 쓰는 글자 토큰
C6 LoRA 위치     언어모델 7종 모듈에만, 개수 = 층 수 x 7
C7 LoRA 초기값   LoRA를 붙인 직후(학습 전) 점수 = C1
C8 기록·재시작   GPU·버전 기록, DONE 있으면 건너뛰는지

결과: runs/pilot_P0_ZS/ (p0_checks.json, p0_samples.txt, val_preds_best.csv, run_config.json, summary.json, DONE)
하나라도 실패하면 DONE을 만들지 않고 멈춥니다.
실행: python scripts/04a_pilot_checks.py                       (코랩, 완료)
      python scripts/04a_pilot_checks.py --run_id pilot_P0K_ZS  (Kaggle)
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LETTERS, load_config  # noqa: E402
import train_common as tc  # noqa: E402

LEN_SETS = ["trainsets/smolvlm_matched/Rand_N1200_s0.jsonl",
            "trainsets/smolvlm_matched/Hard_N1200_s0.jsonl",
            "trainsets/smolvlm_matched/Easy_N1200_s0.jsonl",
            "trainsets/smolvlm_matched/Full_N1200_s0.jsonl"]


def pick_mask_examples(sets):
    """C4용: 2지 정답 A 1개, B 1개 + Full 정답 C 1개, D 1개."""
    rand, full = sets[LEN_SETS[0]], sets[LEN_SETS[3]]
    out = [next(r for r in rand if r["answer_letter"] == "A"),
           next(r for r in rand if r["answer_letter"] == "B")]
    for L in ("C", "D"):
        out.append(next(r for r in full if r["answer_letter"] == L))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="pilot_P0_ZS", help="Kaggle에서는 pilot_P0K_ZS")
    RUN_ID = ap.parse_args().run_id
    cfg = load_config()
    pcfg, tcfg = cfg["pilot"], cfg["train"]
    out = os.path.join(tc.runs_dir(cfg), RUN_ID)
    if os.path.exists(os.path.join(out, "DONE")):
        print(f"[건너뜀] {RUN_ID}는 이미 통과했습니다.")
        return
    os.makedirs(out, exist_ok=True)
    checks = {}

    def record(key, ok, value, criterion):
        checks[key] = {"pass": bool(ok), "value": value, "기준": criterion}
        print(f"[{key}] {'통과' if ok else '실패'}  값={value}  기준: {criterion}", flush=True)

    items = tc.load_items()
    val = tc.val500_records(items)
    sets = {p: tc.load_trainset(p) for p in LEN_SETS}
    ids = {r["item_id"] for s in sets.values() for r in s} | {r["item_id"] for r in val}
    print(f"[이미지] {len(ids)}장 불러오는 중...", flush=True)
    images = tc.load_images(cfg, items, sorted(ids))

    model, processor = tc.load_model_and_processor(cfg)
    lids, eou = tc.letter_ids(processor)

    # C8(기록)
    env = tc.env_info(cfg["models"]["smolvlm"]["id"], cfg["models"]["smolvlm"]["revision"])
    tc.save_json({"run_id": RUN_ID, "env": env, "train_cfg": tcfg}, os.path.join(out, "run_config.json"))

    # C1
    preds = tc.score_records(model, processor, val, images, lids, tcfg["eval_batch"])
    acc = float(preds.correct.mean())
    preds.to_csv(os.path.join(out, "val_preds_best.csv"), index=False)
    record("C1_채점재현", abs(acc - pcfg["zs_ref_acc"]) <= pcfg["c1_tolerance"], round(acc, 4),
           f"{pcfg['zs_ref_acc']} ± {pcfg['c1_tolerance']}")

    # C2
    gen = tc.generation_check(model, processor, val[:20], images, lids, tcfg["eval_batch"],
                              n_examples=20, max_new_tokens=10)
    with open(os.path.join(out, "p0_samples.txt"), "w", encoding="utf-8") as f:
        for e in gen["examples"]:
            f.write(f"{e['item_id']}\t정답 {e['answer']}\t글자응답 {e['is_letter']}\t{e['output']!r}\n")
    record("C2_실제출력", True, f"20개 중 글자 응답 {sum(e['is_letter'] for e in gen['examples'])}개",
           "기록만 (p0_samples.txt)")

    # C3
    max_len, worst = 0, None
    for name, recs in list(sets.items()) + [("val500", val)]:
        for i in range(0, len(recs), 16):
            b = recs[i:i + 16]
            texts = [tc.chat_text(processor, r) + (tc.target_text(r["answer_letter"]) if name != "val500" else "")
                     for r in b]
            enc = tc.encode(processor, texts, [images[r["item_id"]] for r in b], "right")
            lens = enc["attention_mask"].sum(1).tolist()
            j = max(range(len(lens)), key=lambda k: lens[k])
            if lens[j] > max_len:
                max_len, worst = lens[j], b[j]["item_id"]
    record("C3_길이", max_len < tcfg["max_length"], {"max_tokens": max_len, "item_id": worst},
           f"< {tcfg['max_length']}")

    # C4, C5
    ex = pick_mask_examples(sets)
    enc = tc.train_batch(processor, ex, images, lids, eou, tcfg["max_length"])  # 내부 assert도 함께 확인
    decoded, ok4, ok5 = [], True, True
    for i, r in enumerate(ex):
        lab = enc["labels"][i]
        tok_ids = lab[lab != -100].tolist()
        text = processor.tokenizer.decode(tok_ids, skip_special_tokens=False)
        decoded.append({"item_id": r["item_id"], "answer": r["answer_letter"], "K": r["K"], "label_text": text})
        ok4 &= (text == f" {r['answer_letter']}<end_of_utterance>")
        ok5 &= (tok_ids[0] == lids[r["answer_letter"]])
    record("C4_마스킹", ok4, decoded, "각 문항의 손실 토큰이 ' 정답글자<end_of_utterance>'뿐 (A·B·C·D 모두)")
    record("C5_글자토큰", ok5, {L: lids[L] for L in LETTERS}, "학습 타깃 글자 id = 채점 글자 id")

    # C6, C7
    pm = tc.attach_lora(model, cfg, seed=0)
    rep = tc.lora_module_report(pm, model.config)
    record("C6_LoRA위치", not rep["bad_modules"] and rep["n_modules"] == rep["expected"], rep,
           "vision·connector 없음, 개수 = 언어모델 층 수 x 7")
    preds2 = tc.score_records(pm, processor, val, images, lids, tcfg["eval_batch"])
    acc2 = float(preds2.correct.mean())
    record("C7_LoRA초기값", abs(acc2 - acc) <= 0.004, round(acc2, 4), f"C1 값 {round(acc, 4)} ± 0.004")

    # C8(재시작): DONE 건너뛰기 규칙이 04_train.py에 있는지 확인
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_train.py"), encoding="utf-8").read()
    record("C8_기록재시작", bool(env["gpu"]) and '"DONE"' in src,
           {"gpu": env["gpu"], "torch": env["torch"], "transformers": env["transformers"], "peft": env["peft"]},
           "GPU·버전 기록됨, 04_train.py가 DONE이면 건너뜀")

    all_pass = all(c["pass"] for c in checks.values())
    tc.save_json({"checks": checks, "all_pass": all_pass}, os.path.join(out, "p0_checks.json"))
    tc.save_json({"run_id": RUN_ID, "val_acc": acc, "val_acc_with_lora_init": acc2,
                  "letter_rate_first20": gen["letter_rate"], "max_tokens": max_len},
                 os.path.join(out, "summary.json"))
    if all_pass:
        open(os.path.join(out, "DONE"), "w").close()
        print("==> P0 전부 통과. P1~P3(학습률 고르기)로 가도 됩니다.")
    else:
        print("==> P0 실패 항목이 있습니다. 위 '실패' 줄을 보고 고친 뒤 이 셀을 다시 실행하세요.")
        sys.exit(1)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
