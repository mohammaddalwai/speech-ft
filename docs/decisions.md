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
- Error analysis of the 50 worst utterances, including whether the gain is accent-related or transcript style
- Served model: CTranslate2 + FastAPI + Docker on a Hugging Face Space, RTF measured on the deployment CPU
- README and model card

Only if triggered or ahead of schedule:
- E3 (replay): only if E1 raises LibriSpeech WER by more than 10% relative on test-clean or test-other
- E2 (LoRA): only if E1 does not fit in memory, or if Phases 1-3 finish early
- Recover segments longer than 30 s for training by cutting them with word timings (see R2)
- 3 seeds for the final configuration: only if ahead of schedule
- GitHub Actions CI, cloud deployment: only in spare time in week 4

Cut: E4, E5, E6, Common Voice.

## Pre-set rules
- Gate 1 (dataset choice): zero-shot whisper-small on the fixed 1,000-utterance sample of our val split (splits/gate1_val_sample.txt). If pooled WER >= 10%, choose EdAcc. Only if EdAcc fails, measure L2-ARCTIC the same way.
- One dataset per released model (license reasons).
- All reported WER numbers use the same GPU type (L4) and precision. Speed numbers use the deployment CPU, after warm-up.
- Tune on validation only. Run the test set once per final configuration.

## EdAcc facts (census, 2026-10-24)
- 61 conversations with 2 speakers each (122 speakers), 19,137 segments, 32.74 h. Audio is 32 kHz.
- Official validation and test splits share no conversation.
- Segments over 30 s: 641 segments, 9.03 h (27.6% of all hours); only 0.65 h of that is IGNORE segments.
- Tags in transcripts: <LAUGH>, <DTMF>, <OVERLAP>, <NO-SPEECH>, <FOREIGN>, <BREATH>, <COUGH>, <RING>, <LIPSMACK>, <CLICK>, plus IGNORE_TIME_SEGMENT_IN_SCORING.
- Transcripts are verbatim: all capitals, no punctuation, repeated words, partial words and fillers kept.

## Data rules (fixed 2026-10-24, before any WER was computed)
- R1: drop segments containing IGNORE_TIME_SEGMENT_IN_SCORING (speech that was never transcribed).
- R1b: drop segments containing <FOREIGN> (foreign-language speech that was never transcribed).
- R2: drop segments longer than 30 s (Whisper's input limit; no word timings to cut them).
- R3: drop segments that are empty after Whisper's English normalizer (tags only).
- R4: keep short clips; report WER by clip length in the error analysis.
- R5: training text is lowercased with all <...> tags removed; nothing else changes.
- R6: split by conversation, 70/10/20 (train/val/test) by usable hours; best of 20,000 random candidates (seed 0): val and test within 15% of target hours, then most group coverage.
- R7: accent groups by first language (configs/accent_groups.yaml); groups with fewer than 4 speakers merge into "other" (by speaker count only). Headline group comparison: English as first language vs not.
- The split files are created once by scripts/make_splits.py and never regenerated after any WER is seen.

Known costs of these rules (report in README and model card):
- R2 removes about 8.4 h of transcribed speech and almost all usable speech of speakers C31-A, C34-A and C34-B. The test set therefore excludes long monologues.
- Groups use first language, not how people sound: several second-language speakers report a "Mainstream US English" accent.
- north_american has only about 0.7 usable hours, so its numbers will be noisy.
- Because of R5 and EdAcc's style, the fine-tuned model will lose casing and punctuation. WER is not affected (normalization removes both), but the demo output will be plain lowercase text.

## Schedule checkpoints
- Nov 1: splits frozen, baselines on val done
- Nov 8: E1 val result
- Nov 15: test results, forgetting check, error analysis
- Nov 21: demo live, README and model card done
- If E1 has no val result by Nov 8: cut everything optional; ship E1 + test + forgetting + Docker + Space.

## Log
- 2026-10-24: Phase 0 done. Mac and Colab L4 environments pass. CTranslate2 export path works with transformers 5.20.2. int8 output differed between Mac (ARM) and Colab (x86) CPUs on one clip, so Phase 4 accuracy and speed checks run on the deployment CPU only.
- 2026-10-24: EdAcc inspected and census run. Data rules R1-R7 fixed. C39-A moved to east_southeast_asian (l1 "English", accent "Indonesian English"); middle_eastern (3 speakers) merged into "other".
