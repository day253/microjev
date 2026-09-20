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

## Experiment C1: shared candidate scorer (running)

- Started around 2026-09-21 00:00 local. Runtime tool session at launch: 20052; check processes/logs, not just the session ID.
- Implemented `candidate.py`, `candidate_sst5.py`, `candidate_cli.py`, `candidate_finalize.py`; all 28 local tests passed before launch.
- Command: `python -u -m microjev.candidate_sst5 --base-model ../../work/gpt2-pretrained --data-dir ../../work/sst5 --output runs/candidate-pilot-v1 --train-limit 1024 --selection-limit 550 --epochs 2 --batch-size 4 --learning-rate 1e-5`.
- Starts from original GPT-2, not the already fine-tuned baseline. 1,024 reviews -> 3,072 questions; 1,536 updates. Train question variants include positive/negative/neutral/not-negative propositions and reversed score scales.
- Logs: workspace `work/candidate-pilot-v1.log`; reports/checkpoints: `runs/candidate-pilot-v1`. Initial measured updates around 0.25 seconds each, roughly 7 minutes plus validation/probes/save. Test disabled; calibration untouched.
- Evaluate selection NLL and additional unseen instruction/negation probes. Preserve all results, including failures. The fixed-head baseline remains available.

## Next experiments and finalization

After C1 completes, inspect `training_report.json` and compare ONLY selection metrics and instruction probes. If learning is stable, run a bounded full-data candidate experiment at a conservative learning rate, recording original or warm-start source. Do not launch two GPU jobs. Prefer learning improvements over adding API surface.

Candidate experiments default to no calibration and no test. Choose the final candidate checkpoint using selection results; then run `python -m microjev.candidate_finalize --model runs/<selected>/best --output runs/candidate-final --data-dir ../../work/sst5` once. It calibrates on the distinct 551-row split and tests once without retraining. Stop tuning after consulting that final test.

For each completed run, publish a sanitized report in `docs/benchmarks/`, update this log, and commit/push code/docs after necessary tests. The heartbeat runs every 30 minutes through 08:00 local; bounded caffeinate PID 77958 expires then. Preserve this protocol and stay quiet unless there is a meaningful change.

## Architectural objective

Build a shared scalar scorer over GPT-2 representations of state, question and candidate text. Softmax across each question's candidate scores yields a variable-length Choice distribution; two candidate scores can implement Noul; ordered candidates can implement Score. Candidate names/order must not be hard-coded into separate classifier weights.

Start with a bounded validation experiment and candidate permutation tests. Use a reproducible small training subset before committing to the full dataset. A successful API contract is necessary but not proof of generalization to unseen domains or questions. The straightforward candidate-batched implementation repeats the state and is not Jev's shared-context parallel sampler; measure its cost honestly.
