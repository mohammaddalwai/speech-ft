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

## Evaluation settings (fixed 2026-10-24, before any WER was computed)
- Decoding (configs/decode.yaml): English, transcribe, greedy (1 beam), at most 224 new tokens, float16 on the L4. The same for every model and split; serving also uses greedy decoding.
- WER is pooled over words: (S + D + I) / reference words, after the same normalizer on both sides.
- Normalizer v2 (src/text.py), fixed 2026-10-25 after the Gate 1 review (val only) and before any baseline was reported: Whisper's English normalizer, then (a) more filler spellings removed: hm, hmmm, mmmm..., uhh, umm, uhm, er, erm, ah; (b) spelling conventions unified: alright -> all right, ok -> okay, cannot -> can not, kinda -> kind of, sorta -> sort of, dunno -> do not know, lemme -> let me, gimme -> give me, cause/cuz/coz/cos -> because. Same for every system. Its effect is measured by scripts/normalization_effect.py.
- Utterances whose reference is empty after normalization (only fillers or tags) are not scored; they are counted, and so are model outputs on them.
- Per-group WER uses the same pooling inside each group; the macro average is the plain mean of group WERs.
- Hallucination counts (counted, never removed): "capped" = output reached the 224-token limit without ending; "long" = output longer than 2x the reference plus 4 words.
- A run name is never reused; results/results.csv only gets new rows.

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
- R6: split by conversation, 70/10/20 (train/val/test) by usable hours; best of 20,000 random candidates (seed 0): val and test within 15% of target hours, then up to 2 speakers per group in test, then every group in train, then in val, then closest hours. A speaker counts toward a group only with at least 2 minutes of usable audio in that split.
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
- 2026-10-24: First split (commit dbe087e) rejected before any WER was computed. Its scoring counted a group as covered with any utterance at all, so north_american had 12 seconds of test audio (1 speaker) and african had 1 test speaker. R6 scoring changed to require 2 speakers with at least 2 minutes per group in test; dataset revision pinned in the script. Splits regenerated once with the new rule.
- 2026-10-25: Gate 1: zero-shot whisper-small on the 1,000-utterance val sample, pooled WER 30.24% (normalizer v1, commit ae5a5c5). Rule says >= 10%, so EdAcc is the dataset; L2-ARCTIC not measured. Error review of that run: two looping outputs ("blah" x192, "mmmm...") add about 1.5 points; filler spellings and spelling conventions led to normalizer v2. The word-count hallucination check missed the "mmmm..." loop, so outputs that hit the token cap are now counted too. The Gate 1 row in results.csv stays as a decision record (normalizer v1).
