# Overnight experiment log

User objective: improve GPT-2 124M on Apple Silicon toward Jev-style typed probabilistic decisions. The user authorized autonomous iteration through the morning of 2026-09-21; aim to deliver by 08:00 Asia/Shanghai.

## Working protocol

- Repository: `day253/microjev`, branch `main`. Keep tested code and reproducible commands committed and pushed.
- Local checkout: `outputs/microjev` relative to the task workspace. Python runtime: `work/venv-mlx/bin/python` relative to that workspace. Use the runtime, not the macOS system Python. Use `/Library/Developer/CommandLineTools/usr/bin/git` if the system Git shim reports an Xcode license error.
- Run one GPU training process at a time. Inspect process state and current logs before starting another. Preserve working baseline checkpoints.
- Train on public labeled data, use selection validation only to choose models/hyperparameters, fit temperatures on a distinct calibration split, and evaluate the test set only for the selected final experiment. The fixed-head baseline may have its own one-time final test report. Do not use test scores to tune experiments.
- Compare validation NLL, Brier score, accuracy, and inference cost. Keep failed or worse experiments in this log with their measured outcome; do not call interface similarity a reproduction of Jev's architecture or RLCD.
- Prioritize runtime question/candidate conditioning and candidate-order invariance over adding more fixed classification heads. Check type bounds and probability normalization.
- Before 08:00 local time, stop launching experiments that cannot finish and be evaluated by the deadline. Finish by reporting the best verified model, local checkpoint path, reproduction commands, metrics and limitations. Leave the repo clean and CI green.

## Baseline A: synthetic support tickets (complete)

- GPT-2 124M pretrained checkpoint, MLX float32, full fine-tuning, 288 synthetic English examples, 3 epochs.
- Training: 216 steps, 20.6 seconds.
- 18 held-out paraphrases: intent 17/18, urgency 9/18, impact 13/18. This exposed poor generalization, especially urgency.
- Checkpoint: `runs/gpt2-demo`; report: `docs/benchmarks/gpt2-demo.json`.
- This baseline has fixed task heads and does not read runtime instructions.

## Baseline B: real SST-5 sentiment (complete)

- Started 2026-09-20 around 23:42 Asia/Shanghai.
- Command: `python -m microjev.sst5 --base-model ../../work/gpt2-pretrained --data-dir ../../work/sst5 --epochs 3 --batch-size 8 --output runs/gpt2-sst5` from the checkout, using the workspace MLX runtime.
- Data: 8,544 train, 550 selection, 551 calibration, 2,210 test. All labels are projections of the same five-level human sentiment label.
- Process log: `work/sst5-training.log` relative to workspace. Tool session at initial launch: 46907; prefer checking actual processes/logs over assuming the session still exists.
- Checkpoints and progress: `runs/gpt2-sst5/checkpoints/epoch-N`, `runs/gpt2-sst5/progress.json`.
- Completed at 23:51 local. Total 522.95 seconds; training loop plus epoch evaluation/checkpoints 500.49 seconds.
- Selection mean NLL by epoch: 0.7748, 0.8344, 1.3007. Retained epoch 1; later epochs overfit.
- Calibrated test (2,210 rows, evaluated once): sentiment accuracy 0.7570 / NLL 0.6040; positive accuracy 0.8692 / NLL 0.3297; rating accuracy 0.5235 / NLL 1.0902.
- Offline-ready checkpoint: `runs/gpt2-sst5/best`; published report: `docs/benchmarks/sst5-training.json`.
- All 20 local tests passed after the runner changes.

## Experiment C1: shared candidate scorer (complete; insufficient learning)

- Started around 2026-09-21 00:00 local. Runtime tool session at launch: 20052; check processes/logs, not just the session ID.
- Implemented `candidate.py`, `candidate_sst5.py`, `candidate_cli.py`, `candidate_finalize.py`; all 28 local tests passed before launch.
- Command: `python -u -m microjev.candidate_sst5 --base-model ../../work/gpt2-pretrained --data-dir ../../work/sst5 --output runs/candidate-pilot-v1 --train-limit 1024 --selection-limit 550 --epochs 2 --batch-size 4 --learning-rate 1e-5`.
- Starts from original GPT-2, not the already fine-tuned baseline. 1,024 reviews -> 3,072 questions; 1,536 updates. Train question variants include positive/negative/neutral/not-negative propositions and reversed score scales.
- Logs: workspace `work/candidate-pilot-v1.log`; reports/checkpoints: `runs/candidate-pilot-v1`. Initial measured updates around 0.25 seconds each, roughly 7 minutes plus validation/probes/save. Test disabled; calibration untouched.
- Evaluate selection NLL and additional unseen instruction/negation probes. Preserve all results, including failures. The fixed-head baseline remains available.

## Next experiments and finalization

The active or latest experiment is the final entry in this log. Inspect its actual process and progress before launching anything. After completion, compare ONLY selection metrics and instruction probes. Prefer targeted learning improvements over adding API surface, and never launch two GPU jobs.

Candidate experiments default to no calibration and no test. Choose the final candidate checkpoint using selection results; then run `python -m microjev.candidate_finalize --model runs/<selected>/best --output runs/candidate-final --data-dir ../../work/sst5` once. It calibrates on the distinct 551-row split and tests once without retraining. Stop tuning after consulting that final test.

For each completed run, publish a sanitized report in `docs/benchmarks/`, update this log, and commit/push code/docs after necessary tests. The heartbeat runs every 30 minutes through 08:00 local; bounded caffeinate PID 77958 expires then. Preserve this protocol and stay quiet unless there is a meaningful change.

## Architectural objective

Build a shared scalar scorer over GPT-2 representations of state, question and candidate text. Softmax across each question's candidate scores yields a variable-length Choice distribution; two candidate scores can implement Noul; ordered candidates can implement Score. Candidate names/order must not be hard-coded into separate classifier weights.

Start with a bounded validation experiment and candidate permutation tests. Use a reproducible small training subset before committing to the full dataset. A successful API contract is necessary but not proof of generalization to unseen domains or questions. The straightforward candidate-batched implementation repeats the state and is not Jev's shared-context parallel sampler; measure its cost honestly.

### C1 outcome

- Code revision: `5314f84a15e6ef0c49324975c42fce3f5ce6cdad`; completed around 00:06 local. Total 377.20 seconds, training loop plus epoch evaluations 367.79 seconds.
- Selection mean NLL: epoch 1 = 1.2278, epoch 2 = 1.1713; selected epoch 2. This is worse than the fixed-head baseline (0.7748) and even the uniform-distribution average NLL (1.1337).
- Selection accuracy: sentiment 37.09%, positive 58.91%, five-level rating 22.73%. Noul is effectively close to the majority baseline.
- Unseen question probes: positive paraphrase 59.27% / NLL 1.4029; negative paraphrase 60.00% / NLL 1.4875; not-positive 58.91% / NLL 0.6817. These do not demonstrate robust instruction following.
- Ten-candidate / three-question inference: mean 23.18 ms over five warm runs, short self-authored English example. This is not an end-to-end service benchmark.
- Report `docs/benchmarks/candidate-pilot-v1.json`; checkpoint `runs/candidate-pilot-v1/best`. No calibration/test consulted.
- Next: verify the model can intentionally overfit a small training subset, then use the proven SST-5 fine-tuned backbone and full training split for a longer candidate run. This changes source/data/LR together as an exploratory improvement, not a controlled attribution of cause.

## Diagnostic: small-set memorization (complete)

- Started from the retained SST-5 fixed-head backbone, randomly initialized shared scalar head, 16 shuffled training reviews / 48 question variants, 20 epochs, batch 8 questions, LR 1e-4, seed 42.
- 120 updates / 35.95 seconds. Training-set accuracy reached 100% for all three question types; NLL 0.0002 / 0.0010 / 0.0009. Confirms optimization can learn this objective, not evidence of generalization. No checkpoint retained.
- Published report: `docs/benchmarks/candidate-overfit-check.json`.

## Experiment C2: full-data candidate scorer (complete; overconfident and weak negation)

- Started 2026-09-21 00:08 local, detached PID **85800**. Check process identity and log before any GPU work; do not duplicate. PID file: workspace `work/candidate-full-v2.pid`; launch manifest: `work/candidate-full-v2-launch.json`.
- Code revision: `5314f84a15e6ef0c49324975c42fce3f5ce6cdad` (subsequent docs commits do not change its implementation).
- Command: `python -u -m microjev.candidate_sst5 --base-model runs/gpt2-sst5/best --source-kind fixed --data-dir ../../work/sst5 --output runs/candidate-full-v2 --train-limit 0 --selection-limit 550 --epochs 2 --batch-size 8 --learning-rate 2e-5`.
- Source selection is based on Baseline B validation quality; its backbone previously trained on the same 8,544 training reviews. New shared head, all parameters trainable, full data, 25,632 question variants / epoch, 6,408 total updates. Optimizer starts fresh.
- Logs: workspace `work/candidate-full-v2.log`; reports and saved epochs: `runs/candidate-full-v2`. After completion inspect `training_report.json`. Test evaluation and calibration are disabled.
- Estimate around 35–45 minutes; use the actual progress log to refine. All full-data training and selection/probe sequences were checked to fit: max 113 training tokens, max 97 selection/probe tokens (limit 192).
- On completion, compare selection NLL and unseen instruction probes with C1 and fixed-head baseline. If worth improving, make the next bounded experiment; preserve both baselines. Only finalize/calibrate/test after model selection is finished.

### C2 outcome

- Completed around 00:50 local; total 2,501.85 seconds (41m42s), training loop including epoch evaluation/save 2,487.03 seconds. Detached PID 85800 has exited.
- Selection mean NLL by epoch: 1.2684, 1.9515. Retained epoch 1. It learned classification, but further training worsened probability quality.
- Selected checkpoint: sentiment accuracy 71.27% / NLL 1.2733; positive accuracy 83.64% / NLL 0.7488; rating accuracy 48.73% / NLL 1.7831. Baseline B remains better on canonical NLL (0.7748 average).
- Selection instruction probes: positive paraphrase 54.00% / NLL 1.0106, negative paraphrase 62.91% / NLL 0.6860, not-positive 24.36% / NLL 3.4725. Negation is a concrete failure; do not describe this model as generally following instructions.
- Warm inference mean 23.04 ms for one short input with ten candidates. Report: `docs/benchmarks/candidate-full-v2.json`; retained model: `runs/candidate-full-v2/best`. No candidate calibration or test consulted.

## C3 preparation: paired propositions and smoothing

- The next bounded experiment addresses C2's observed errors: each training review gets one proposition and its logical complement; both labels come from the same original sentiment annotation. Four alternative phrasings for each of positive, negative and neutral, including favorable/unfavorable wording. Probe strings remain disjoint from training templates.
- Optional label smoothing uses `(1-epsilon) * target + epsilon / candidates` on training targets only. Evaluation remains on original labels; input records are not mutated. Defaults preserve older experiments.
- New `--selection-objective instruction-balanced` averages canonical mean NLL and the three held-out-phrasing probe NLLs. They are selection diagnostics repeatedly used for development, not an independent final instruction-following test. `selection_nll` remains canonical; `selection_score` explicitly records the chosen objective.
- Plan: initialize from C2 epoch 1, reset optimizer, full 8,544 reviews, 2 epochs, batch 8 questions, LR 5e-6, smoothing 0.1, paired augmentation. This changes several factors as an exploratory improvement; do not attribute gains to one factor alone.
- 30 local tests passed, including paired-label correctness, held-out phrasing separation and actual smoothed training loss without changing caller labels. A tiny full GPT-2 pipeline smoke run also exercises the new selection objective before launching C3.

## Experiment C3: paired propositions with smoothing (complete; improved probes)

- Started 2026-09-21 01:05 local, detached PID **11164**. PID file: workspace `work/candidate-paired-v3.pid`; launch manifest: `work/candidate-paired-v3-launch.json`. C2 PID 85800 has exited.
- Code revision: `fab9908498a2e051bf162c0d3c24eca1ecb195ee`. Runtime remains workspace `work/venv-mlx/bin/python`.
- Command: `python -u -m microjev.candidate_sst5 --base-model runs/candidate-full-v2/best --source-kind candidate --data-dir ../../work/sst5 --output runs/candidate-paired-v3 --train-limit 0 --selection-limit 550 --epochs 2 --batch-size 8 --learning-rate 5e-6 --label-smoothing 0.1 --augmentation paired --selection-objective instruction-balanced`.
- Warm-starts from C2's chosen epoch 1. That source already saw the original SST-5 training data in Baseline B and C2; this is additional training, not a fresh small-data result. Fresh optimizer, all model parameters trainable.
- 8,544 reviews / 34,176 questions per epoch, 8,544 total updates. Expected 50–65 minutes, refine from log. Each row has canonical-style Choice/Score plus one varied proposition and its logical complement. No exact probe question is a training template.
- Logs: workspace `work/candidate-paired-v3.log`; checkpoints/progress/report: `runs/candidate-paired-v3`. Calibration/test disabled. Do not finalize while still selecting experiments.
- Before launch: all 30 local tests passed; full-GPT2 smoke run completed two updates, saved/reloaded, evaluated both selection groups, and verified the combined objective numerically. The tiny smoke output is diagnostic only, not a quality benchmark.
- After C3: check both canonical NLL and each probe. The old C2 canonical average is 1.2684; its combined canonical/probe objective is about 1.4957. C1's majority-like predictions have lower raw NLL than C2, so compare accuracy and instruction behavior alongside NLL rather than claiming every change is an improvement.

### C3 outcome

- Completed around 01:55 local, total 2,947.51 seconds (49m08s). PID 11164 exited normally. Report: `docs/benchmarks/candidate-paired-v3.json`; retained checkpoint: `runs/candidate-paired-v3/best`.
- Epoch 1 selected: canonical mean NLL 0.9348, instruction-balanced selection score 0.7298. Epoch 2 worsened to canonical NLL 1.0174 / balanced score 0.7673. Again, more epochs did not help probability quality.
- Epoch 1 canonical accuracy: sentiment 70.18%, positive 82.91%, five-level rating 50.18%. Canonical NLL remains worse than fixed-head baseline B (0.7748).
- Probe accuracy: positive paraphrase 82.91%, negative paraphrase 85.64%, not-positive 64.18%. Corresponding NLL: 0.4287, 0.4360, 0.7097. This improves substantially over C2's probe failures, but negation is still weaker.
- Warm short-input inference mean 23.17 ms, ten candidates. Calibration and test remain untouched for candidate models.
- Next: run a selection-only negation wording diagnostic before deciding another learning change. Preserve C3; it is currently the strongest candidate-interface prototype under the balanced selection objective. Do not infer broad domain competence from sentiment results.

### C3 negation wording diagnostic

- Selection-only comparison of five semantically related questions, report `docs/benchmarks/candidate-v3-negation-diagnostic.json`. No calibration or test touched.
- For negative reviews, the familiar not-positive question classified 95.43% correctly and a short unseen wording 91.78%; adding the probe's neutral-return-true clause dropped this to 51.14%. Lowercasing NOT dropped it further to 39.73%, so case normalization is not a fix.
- The explicit negative-or-neutral question was only 22.86% correct on neutral reviews. The model is sensitive to clause wording and does not reliably compose boundary instructions.
- C4 adds `--augmentation boundary`: compatible positive/negative/neutral return-true/false clauses on all three predicates and their complements, both before and after the base question, with NOT case variation. A quarter of rows retain short questions. Exact development-probe questions remain excluded from training templates.
- Retain C3 as the current best candidate. C4 will continue from its epoch 1 for just one epoch (previous second epochs repeatedly worsened NLL), LR 3e-6, smoothing 0.15. Treat this as an exploratory combined intervention, not a controlled estimate of one technique.
- All 31 local tests passed, including boundary-target consistency and full-cycle held-out-phrase checks.
