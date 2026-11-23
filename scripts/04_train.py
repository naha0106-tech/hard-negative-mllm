"""4단계: LoRA 학습 1회 (파일럿·본실험 공용)

하는 일
  1) 학습 세트(.jsonl)와 val500, 필요한 이미지를 불러옴
  2) SmolVLM에 LoRA를 붙이고 학습 (fp16 autocast + GradScaler, 실효 배치 16)
  3) 25 step마다 val500 채점 → 최고 점수 LoRA를 저장, 4번 연속 개선 없으면 조기 종료
  4) 최고 체크포인트로 글자 출력 확인(B6)
  5) runs/{run_id}/ 에 run_config.json, train_log.csv, val_preds_best.csv,
     gen_check.json, adapter/, summary.json, DONE 저장

실행 예
  python scripts/04_train.py --trainset trainsets/smolvlm_matched/Rand_N400_s0.jsonl \
      --run_id pilot_P1_Rand_N400_lr5e-5 --lr 5e-5 --seed 0
이미 DONE이 있으면 아무것도 하지 않습니다. 끊겼던 run은 처음부터 다시 돕니다.
"""
import argparse
import csv
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, load_json  # noqa: E402
import train_common as tc  # noqa: E402


def batch_stream(n, micro, seed):
    """매 바퀴(epoch)마다 시드로 순서를 섞어 micro 크기 묶음을 끝없이 내보냅니다."""
    epoch = 0
    while True:
        perm = np.random.default_rng([seed, 9000 + epoch]).permutation(n)
        for i in range(0, n - micro + 1, micro):
            yield perm[i:i + micro].tolist()
        epoch += 1


def zs_reference(cfg, rdir):
    for rid in ("pilot_P0K_ZS", "pilot_P0_ZS"):  # Kaggle 점검값이 있으면 그것을 씀
        p = os.path.join(rdir, rid, "summary.json")
        if os.path.exists(p):
            return load_json(p)["val_acc"]
    return cfg["pilot"]["zs_ref_acc"]


def write_log(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "lr", "val_acc", "elapsed_sec"])
        w.writerows(rows)


def train_once(args, cfg, attempt, log):
    tcfg = cfg["train"]
    rdir = tc.runs_dir(cfg)
    out = os.path.join(rdir, args.run_id)
    if os.path.exists(out):
        shutil.rmtree(out)  # DONE이 없는 반쪽짜리 결과는 지우고 다시
    os.makedirs(out)

    items = tc.load_items()
    train = tc.load_trainset(args.trainset)
    val = tc.val500_records(items)
    n = len(train)
    max_steps = tc.max_steps_for(n, tcfg)
    log(f"[데이터] 학습 {n}문항, val500 {len(val)}문항, 최대 step {max_steps}")
    images = tc.load_images(cfg, items, [r["item_id"] for r in train] + [r["item_id"] for r in val])

    model, processor = tc.load_model_and_processor(cfg)
    lids, eou = tc.letter_ids(processor)
    pm = tc.attach_lora(model, cfg, args.seed)
    lora_rep = tc.lora_module_report(pm, model.config)
    assert not lora_rep["bad_modules"] and lora_rep["n_modules"] == lora_rep["expected"], \
        f"[LoRA 위치 오류] {lora_rep}"
    tc.set_all_seeds(args.seed)

    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    from transformers import get_cosine_schedule_with_warmup
    warmup = max(tcfg["warmup_min_steps"], int(round(tcfg["warmup_ratio"] * max_steps)))
    sched = get_cosine_schedule_with_warmup(opt, warmup, max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(tcfg["precision"] == "fp16"))
    amp_dtype = torch.float16 if tcfg["precision"] == "fp16" else torch.bfloat16

    run_config = {"run_id": args.run_id, "trainset": args.trainset, "n_items": n, "lr": args.lr,
                  "seed": args.seed, "max_steps": max_steps, "warmup_steps": warmup,
                  "train_cfg": tcfg, "lora_report": lora_rep, "attempt": attempt,
                  "env": tc.env_info(cfg["models"]["smolvlm"]["id"], cfg["models"]["smolvlm"]["revision"]),
                  "reused_from": None, "started": datetime.now().isoformat(timespec="seconds")}
    tc.save_json(run_config, os.path.join(out, "run_config.json"))

    log_path = os.path.join(out, "train_log.csv")
    log_rows = []  # 디스크에는 검증할 때만 씀 (느린 저장소 대비)

    stream = batch_stream(n, tcfg["micro_batch"], args.seed)
    elapsed = tc.timer()
    eval_sec, train_sec = 0.0, 0.0
    best = {"acc": -1.0, "step": 0, "preds": None, "state": None}
    no_improve, diverged, stopped_early, step = 0, False, False, 0

    while step < max_steps:
        pm.train()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(tcfg["grad_accum"]):
            recs = [train[i] for i in next(stream)]
            enc = tc.to_cuda(tc.train_batch(processor, recs, images, lids, eou, tcfg["max_length"]))
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = pm(**enc).loss / tcfg["grad_accum"]
            if not torch.isfinite(loss):
                diverged = True
                break
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        if diverged:
            log(f"[발산] step {step + 1}에서 손실이 NaN/inf")
            break
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, tcfg["max_grad_norm"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        train_sec += time.time() - t0

        val_acc = ""
        if step % tcfg["eval_every"] == 0 or step == max_steps:
            t1 = time.time()
            preds = tc.score_records(pm, processor, val, images, lids, tcfg["eval_batch"], amp_dtype)
            eval_sec += time.time() - t1
            val_acc = float(preds.correct.mean())
            if val_acc > best["acc"]:  # 동점이면 이른 step 유지
                best = {"acc": val_acc, "step": step, "preds": preds, "state": tc.lora_state(pm)}
                no_improve = 0  # 최고 LoRA는 메모리에만 두고 끝에서 한 번 저장
            else:
                no_improve += 1
            log(f"  step {step}/{max_steps}  loss {loss_sum:.4f}  val {val_acc:.4f}  best {best['acc']:.4f}@{best['step']}")
        log_rows.append([step, round(loss_sum, 6), sched.get_last_lr()[0], val_acc, round(elapsed(), 1)])
        if val_acc != "":
            write_log(log_path, log_rows)
        if val_acc != "" and no_improve >= tcfg["patience"]:
            stopped_early = True
            log(f"[조기 종료] {tcfg['patience']}번 연속 개선 없음 (step {step})")
            break

    zs = zs_reference(cfg, rdir)
    if not diverged and best["acc"] < zs - tcfg["diverge_drop"]:
        diverged = True
        log(f"[발산] 최고 검증 {best['acc']:.4f} < 무학습 {zs:.4f} - {tcfg['diverge_drop']}")

    write_log(log_path, log_rows)
    gen = None
    if best["state"] is not None:
        pm.load_state_dict(best["state"], strict=False)  # 최고 체크포인트로 되돌림
        pm.save_pretrained(os.path.join(out, "adapter"))
        best["preds"].to_csv(os.path.join(out, "val_preds_best.csv"), index=False)
        gen = tc.generation_check(pm, processor, val, images, lids, tcfg["eval_batch"])
        tc.save_json(gen, os.path.join(out, "gen_check.json"))
        log(f"[B6] 글자로 답한 비율 {gen['letter_rate']:.3f}")

    summary = {"run_id": args.run_id, "best_step": best["step"], "best_val_acc": best["acc"],
               "final_step": step, "max_steps": max_steps, "stopped_early": stopped_early,
               "diverged": diverged, "attempt": attempt,
               "letter_rate": gen["letter_rate"] if gen else None,
               "train_min": round(train_sec / 60, 2), "eval_min": round(eval_sec / 60, 2),
               "n_evals": step // tcfg["eval_every"] + (step % tcfg["eval_every"] != 0),
               "total_min": round(elapsed() / 60, 2)}
    tc.save_json(summary, os.path.join(out, "summary.json"))
    run_config["finished"] = datetime.now().isoformat(timespec="seconds")
    tc.save_json(run_config, os.path.join(out, "run_config.json"))
    del pm, model
    torch.cuda.empty_cache()
    return summary, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainset", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    rdir = tc.runs_dir(cfg)
    if os.path.exists(os.path.join(rdir, args.run_id, "DONE")):
        print(f"[건너뜀] {args.run_id}는 이미 끝났습니다 (DONE 있음).")
        return

    def log(msg):
        print(msg, flush=True)

    log(f"===== {args.run_id} (lr={args.lr}, seed={args.seed}) =====")
    summary, out = train_once(args, cfg, 1, log)
    if summary["diverged"]:
        log("[재시도] 발산해서 같은 설정으로 1번 더 돌립니다.")
        failed = out + "_diverged_try1"
        if os.path.exists(failed):
            shutil.rmtree(failed)
        shutil.move(out, failed)
        summary, out = train_once(args, cfg, 2, log)
        if summary["diverged"]:
            log("[멈춤] 재시도도 발산했습니다. 코드와 설정을 점검하세요. (DONE 기록함, diverged=true)")
    open(os.path.join(out, "DONE"), "w").close()
    log(f"===== 끝: best val {summary['best_val_acc']:.4f} @ step {summary['best_step']}, "
        f"총 {summary['total_min']}분 =====")


if __name__ == "__main__":
    main()
