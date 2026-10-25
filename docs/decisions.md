# Decisions log

## Constraints (set 2026-10-24)
- GPU: Colab Pro. CPU runtime for data work. L4 for development and all reported evaluation. A100 for full training runs only.
- Time: 5-8 hours per week, about 20-32 hours in total.
- Start: 2026-10-24. Hard stop: 2026-11-21. Ship whatever works.

## Scope (fixed before seeing any data)

Must do:
- Frozen speaker- and conversation-held-out splits, with an automatic overlap test
- Baselines, all zero-shot: E0 whisper-small, E0b whisper-medium, E0c whisper-large-v3-turbo
- E1: full fine-tune of whisper-small, 1 seed
- Final test evaluation: pooled and per-accent-group WER, paired bootstrap 95% CI over speakers
- Forgetting check on LibriSpeech test-clean and test-other
- Error analysis of the 50 worst utterances, including whether the gain is accent-related or transcript formatting
- Served model: CTranslate2 + FastAPI + Docker on a Hugging Face Space, RTF measured on the deployment CPU
- README and model card

Only if triggered or ahead of schedule:
- E3 (replay): only if E1 raises LibriSpeech WER by more than 10% relative on test-clean or test-other
- E2 (LoRA): only if E1 does not fit in memory, or if Phases 1-3 finish early
- 3 seeds for the final configuration: only if ahead of schedule
- GitHub Actions CI, cloud deployment: only in spare time in week 4

Cut: E4, E5, E6, Common Voice.

## Pre-set rules
- Gate 1 (dataset choice): zero-shot whisper-small on a fixed random sample of 1,000 EdAcc utterances (seed 0). If pooled WER >= 10%, choose EdAcc. Only if EdAcc fails, measure L2-ARCTIC the same way.
- Accent groups are fixed from metadata before any per-group WER is computed.
- One dataset per released model (license reasons).
- All reported WER numbers use the same GPU type (L4) and precision. Speed numbers use the deployment CPU, after warm-up.
- Tune on validation only. Run the test set once per final configuration.

## Schedule checkpoints
- Nov 1: splits frozen, baselines on val done
- Nov 8: E1 val result
- Nov 15: test results, forgetting check, error analysis
- Nov 21: demo live, README and model card done
- If E1 has no val result by Nov 8: cut everything optional; ship E1 + test + forgetting + Docker + Space.

## Log
- 2026-10-24: Phase 0 done. Mac and Colab L4 environments pass. CTranslate2 export path works with transformers 5.20.2. int8 output differed between Mac (ARM) and Colab (x86) CPUs on one clip, so Phase 4 accuracy and speed checks run on the deployment CPU only.
