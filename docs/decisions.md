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

## Training setup for E1 (fixed 2026-10-25, before any training run)
- configs/E1.yaml: whisper-small, all weights trained, learning rate 1e-5 (linear decay, 200 warmup steps), effective batch 32 (16 x 2 accumulation), at most 2,000 steps (about 5.4 epochs), bf16, gradient checkpointing, seed 42.
- Training text: R5 (lowercase, tags removed). Training data: the train split only.
- Checkpoint selection: val WER (full val split, greedy, normalizer v2) every 250 steps; the best checkpoint is kept; training stops early after 3 evaluations in a row without improvement.
- Because val picks the checkpoint, E1's val WER is slightly optimistic compared with the zero-shot baselines, which were not selected on val. The test set gives the unbiased comparison.
- The official E1 val number comes from scripts/evaluate.py on the selected checkpoint, with exactly the baseline settings.
- Gate 2: E1's official val WER must be below E0 whisper-small (29.42%). If not, check in this order: normalizer, label tokens, learning rate, data alignment, leakage.
- Before E1: (a) debug run, 200 steps on 1,600 train / 200 val utterances: training loss must fall; (b) overfit test on 32 train clips at learning rate 1e-4: WER on those same clips must approach 0.
- Checkpoints of full runs go to a private Hugging Face repo (<user>/speech-ft-<run>), latest checkpoint kept for resuming; history is squashed at the end to save storage.

## Forgetting check (fixed 2026-10-26, before running it)
- LibriSpeech test-clean and test-other (openslr/librispeech_asr, parquet), same decoding and normalizer as everything else, L4.
- Rule L1: clips longer than 30 s are dropped (Whisper's input limit, as R2 for EdAcc); the number dropped is reported.
- Compared: zero-shot whisper-small (E0) vs E1. E3 (replay) is triggered if E1's WER is more than 10% higher (relative) than E0's on either test set (scripts/forgetting_check.py).

## E3 and the final-model choice (fixed 2026-10-26, before E3 runs)
- E3 = configs/E1.yaml plus ONE change: replay. 15% of all training clips are LibriSpeech train-clean-100 (about 2,077 clips <= 30 s, random with seed 42, from the first 4 parquet shards), lowercased like R5. LibriSpeech train and test speakers do not overlap. Checkpoint chosen by EdAcc val WER, exactly as E1.
- Final model = E3 if (a) its official val WER is at most 1.0 point worse than E1's (24.77%), and (b) its LibriSpeech WER is lower than E1's on both test-clean and test-other. Otherwise the final model is E1, and its forgetting is reported as a limitation.
- E1 and E3 are both reported on test either way, with the forgetting-rule result for each.

## Test session (fixed 2026-10-26, before any test-set number exists)
- Final model: E3 (decided by the rule above; see log). Extra training seeds 43 and 44 of E3 are run first, because the project is ahead of schedule (scope rule). The replay clips stay the same across seeds.
- Every system is evaluated on the test split ONCE, in one L4 session: E0 small, E0b medium, E0c turbo, E1, E3 (seeds 42, 43, 44).
- Headline: E3 test WER as mean +/- standard deviation over the 3 seeds (training randomness), plus the paired speaker bootstrap 95% interval (2,000 resamples) for E3 seed 42 vs E0 small, vs E0c turbo and vs E1 (test-set sampling). "Significant" only if the interval excludes 0.
- Per-group results: point WER for every group; bootstrap intervals only for groups with at least 5 test speakers (fewer gives falsely narrow intervals).
- The served model is E3 seed 42 (chosen now, not by test results).

## Serving and release (fixed 2026-10-27, before conversion)
- Served model: E3 seed 42, converted to CTranslate2 (scripts/export_ct2.py) and run with faster-whisper on CPU. Decoding is set explicitly to match configs/decode.yaml: greedy, temperature 0 with no fallback, no timestamps, no voice-activity filter, no previous-text prompt, at most 224 new tokens. faster-whisper's own defaults differ (beam 5, fallback, timestamps), so src/asr.py sets every one and tests/test_release.py checks it.
- Conversion check (scripts/ct2_check.py): 500 random val clips (seed 0), compared with the official E3_val output on exactly the same clips. Val, not test, because this result decides what ships.
- Where: a Colab x86 CPU. Not the Mac: its ARM CPU gave different int8 output in Phase 0. Not the Space: it cannot hold the dataset. Colab's x86 CPU stands in for the Space's; small differences between x86 CPUs are possible and are not checked.
- Rule: ship int8 if its WER on those clips is at most 1.0 point above the official model's. Otherwise float32 under the same rule. If both fail, stop and debug the conversion; nothing is published.
- Whatever ships, its measured difference is printed in the CTranslate2 model card, so the demo's accuracy is not silently assumed to equal the reported test numbers.
- Speed: the README's speed numbers come from the Space only (real-time factor after one warm-up request, and cold start). Colab speed numbers are for orientation and are not reported.
- Release: public repos <user>/whisper-small-edacc (transformers) and <user>/whisper-small-edacc-ct2 (CTranslate2), license CC BY-SA 4.0 because EdAcc is CC BY-SA (LibriSpeech is CC BY 4.0, whisper-small Apache 2.0). Inference files only, no optimizer state or checkpoints. The training repos (speech-ft-E1, -E3, seeds) stay private. No EdAcc or LibriSpeech audio is published anywhere: not in the repos, the Space, its examples or the README.

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
- 2026-10-25: Zero-shot baselines on full val (normalizer v2): small 29.42%, medium 25.66%, turbo 23.83% WER. Deletions barely shrink with model size (12.5% -> 11.4%) while substitutions do (12.2% -> 9.2%), so most deletions look like a transcription-style gap that a bigger model does not close.
- 2026-10-25: Bug in the capped-output count: it flagged the longest output in every batch, because generate() leaves the end token off that output (counts were 62/121/118). Fixed in evaluate.py to count generated text tokens. Counts for the four earlier runs recounted with scripts/fix_capped.py (heuristic: re-tokenized length or a 40+ character "word"). WER is unaffected. Also: the "-dirty" commit tag no longer counts changes inside results/.
- 2026-10-25: E1 trained: 2,000 steps (no early stop), 38.9 min on an A100 80GB, peak 6.7 GB. Val WER during training was noisy (28.7, 31.6, 25.7, 25.4, 29.3, 26.0, 24.8, 25.3%) while val loss rose from the first check (0.61 -> 0.80): overfitting by loss. Best checkpoint: step 1750, pushed to mddalwai/speech-ft-E1 (private).
- 2026-10-25: Official E1 val (L4): 24.77% WER (sub 14.3%, del 4.8%, ins 5.7%), 0 looping outputs. Gate 2 passed (bar 29.42%). Still about 1 point behind zero-shot turbo (23.83%). The match with step 1750's training-time WER (24.80%, vs 25.25% at step 2000) confirms the Hub copy is the best checkpoint.
- 2026-10-25: Where E1's gain comes from (scripts/compare_runs.py, E0_small_val -> E1_val, % of reference words): deleted -> correct 6.1 pts, mostly short discourse words and repeats; substituted -> correct 3.5 pts; correct -> wrong 3.9 pts. On words the zero-shot model already attempted, recognition is roughly flat; the gain is mostly learning EdAcc's word-for-word transcription style. Claims about "better recognition of accented speech" are not supported by val.
- 2026-10-26: Forgetting check (L4): zero-shot small 3.33% / 7.45% WER on LibriSpeech test-clean / test-other (9 and 7 clips over 30 s dropped); close to the Whisper paper's 3.4% / 7.6%, which also validates the pipeline. E1: 4.70% / 10.22%, i.e. +41% / +37% relative, so E3 is triggered. Looping outputs: E1 1 and 6, E0 none; without them E1 is still 4.67% / 10.07%, so the damage is broad, not just loops. E1 is worse than E0 on 971 of 2,932 test-other utterances and better on 303. Typical failures: stopping after a few words, a short conversational reply (e.g. "yeah sir") for a long read sentence, starting mid-sentence. This is consistent with E1 having learned EdAcc's short conversational segments.
- 2026-10-26: E3 trained: 2,000 steps, 40.5 min on an A100, best checkpoint step 1750 (24.09% val WER during training). Replay: 2,077 LibriSpeech clips = 15% of clips but 7.4 h, i.e. about 32% of training audio (LibriSpeech clips are longer than EdAcc's).
- 2026-10-26: Official E3 (L4): val 23.93% WER (E1 24.77%, turbo 23.83%); LibriSpeech 3.85% / 8.62% (E0 small 3.33% / 7.45%, E1 4.70% / 10.22%). Forgetting rule still FAILED (+15.6% / +15.7% relative), but less than half of E1's damage. Final-model rule: E3 val is not worse than E1's and its LibriSpeech WER is lower on both sets, so E3 is the final model; its remaining forgetting is a reported limitation. No looping outputs on val or LibriSpeech.
- 2026-10-26: Where E3's val gain comes from (compare_runs, E0 small -> E3): deleted -> correct 6.1 pts, substituted -> correct 3.6, correct -> wrong 3.5. Same picture as E1: the gain is mostly transcription style; recognition of already-attempted words is about flat.
- 2026-10-26: E2 (LoRA) stays cut: E3 already addresses forgetting, and the remaining time goes to the test session, error analysis and Phase 4. Bootstrap on val (for checking the code only; val picked the checkpoints, so these are optimistic): E3 vs E0 small -5.5 points [-7.3, -3.7]; E3 vs turbo +0.1 [-1.0, +1.3].
- 2026-10-26: E3 seeds 43 and 44 trained (A100, 38.9 min each; best checkpoint step 1500 in both). Official val (L4): 25.19% and 24.11%, so E3 val over 3 seeds = 24.41% +/- 0.68 (std). Seed 43 is worse than E1 on val: E3's val advantage over E1 is within training-seed noise.
- 2026-10-26: Test session (L4, commit 9e4482d, each system once): E0 small 22.53%, E0b medium 19.82%, E0c turbo 18.23%, E1 17.25%, E3 16.31% / 16.81% / 17.13% (seeds 42/43/44; mean 16.75% +/- 0.41). Paired speaker bootstrap, E3 seed 42, 28 speakers: vs E0 small -6.22 points [-8.66, -4.19], -27.6% relative; vs turbo -1.92 [-3.17, -0.68]; vs E1 -0.94 [-2.30, -0.29].
- 2026-10-26: How to read the test results: (1) E3 vs zero-shot small is large and holds for all seeds. (2) E3 beats turbo on test with all 3 seeds, but only tied turbo on val (23.93% vs 23.83%; seed mean 24.41%), so the advantage depends on which speakers are tested; report "on par with or better than turbo", with both numbers. (3) E3 vs E1: the bootstrap interval excludes 0, but the 0.9-point gap is about 2 seed standard deviations of E3 and E1 has one seed only, so it is not established; E3 was chosen for forgetting (LibriSpeech 3.85%/8.62% vs 4.70%/10.22%), not for EdAcc WER. (4) Gain decomposition on test (E0 small -> E3): deleted -> correct 5.7 pts (22% are repeats), substituted -> correct 2.4, correct -> wrong 2.6: the gain is mostly transcription style. (5) Only east_southeast_asian has >= 5 test speakers (6): E3 vs E0 small -10.4 [-17.0, -4.7], vs turbo -5.8 [-10.2, -1.9]; every other group is reported as points only.
- 2026-10-26: POST-HOC sensitivity check (scripts/loop_sensitivity.py; official numbers unchanged): 14 test clips where any system looped. Re-scored without those clips: E0 small 20.68%, medium 19.13%, turbo 17.90%, E1 16.71%, E3 16.22% / 16.25% / 16.54%. So about 1.8 of E3's 6.2-point gain over zero-shot small comes from zero-shot small's looping outputs on short clips (6 loops, 851 word errors), and most of E3's seed spread comes from one looping clip in seeds 43 and 44 (223 errors each).
- 2026-10-26: Error analysis overview (scripts/error_analysis.py): the 50 worst E3 test utterances hold only 9.6% of its word errors, so errors are spread out. WER by clip length (E0 small / turbo / E3): under 1 s (998 clips) 80.1% / 59.2% / 50.6%; 1-3 s 46.8% / 33.7% / 32.7%; 3-10 s 21.0% / 16.8% / 15.6%; 10-30 s 14.5% / 13.1% / 11.0%. The 73 test clips containing <DTMF> (a word bleeped for privacy) hold 6.3% of E3's errors (WER 28.7%).
- 2026-10-27: Hand review of the 50 worst E3 test utterances (all 50 labelled; 19 of the 50 automatic guesses changed, mostly "names" -> "accent"; scripts/label_errors.py). Share of their 735 word errors: common words misheard (accent or fast speech) 59.5% (30 clips), names / titles / rare words 20.7% (10), bleeped words 6.9% (3), invented or looping text 4.6% (3), speech outside the reference 3.3% (1), reference errors 3.1% (2), disfluency differences 1.9% (1). These 50 clips hold only 9.6% of all errors, so this describes the worst cases, not the whole error budget.
- 2026-10-27: Check of a pattern from the review notes ("start of clip lost", 6 clips): on the whole test set E3 loses FEWER reference words at the start of clips than zero-shot small (0.62 vs 1.81 points; turbo 1.92) and at the end (1.21 vs 2.75). So it is a feature of those worst clips, not a systematic E3 failure.

- 2026-10-28: CTranslate2 int8 check, first run (2026-10-27, Colab CPU, 8 vCPUs, 500 val clips, 495 scored): 23.27% vs official 22.42% WER on the same clips, +0.86 points; 292/500 outputs identical. Within the 1.0-point rule but close to the limit. Its files were lost when the Colab runtime was recycled before download, so the check is re-run; the re-run (files kept) is the official result, and both runs are reported. Fixed now, before the re-run and before any float32 result: float32 is run as a control before publishing. (a) If float32 is also more than 0.5 points worse than the official model, the gap comes from the serving pipeline, not from int8, and nothing is published until the cause is found. (b) Otherwise the original rule applies to the re-run: int8 ships if within 1.0 point, else float32; the CTranslate2 model card reports both numbers.
- 2026-10-28: Re-run, serving v1 (Colab CPU, AMD EPYC 7B12, 8 vCPUs; results/runs/ct2_check_int8 and ct2_check_float32): int8 23.14% vs official 22.42% on the same 495 scored clips (+0.72 points, 307/500 identical); float32 23.23% (+0.81, 314/500). int8 and float32 give the same text on 362/495 clips, more often than either matches the official model, so the gap is not from int8. The two int8 runs differ (+0.86 vs +0.72), so int8 output depends on the CPU. Rule (a) applies: float32 is more than 0.5 points worse, so nothing is published. Where the gap is (a look after the result): one clip where the official output already repeats a phrase and both CTranslate2 versions repeat it longer holds 24-30 extra errors (about 0.4 points); without the 3 worst clips the gap is +0.06 (int8) / +0.25 (float32); clips that got worse and better are about balanced (int8 68/60, float32 68/53). The 495 clips hold only 6,772 reference words, so one looping clip moves WER by about 0.4 points: the 0.5- and 1.0-point limits are coarse for this sample. The limits stay as they are.
- 2026-10-28: Cause found in the serving code (a mistake in src/asr.py v1: its test checked the decoding settings but not the encoder input). faster-whisper 1.2.1 pads the log-mel features with 0.0 up to 30 s. transformers, used for training and every reported WER, pads the audio with silence, so its padded frames hold the log-mel of silence (-0.54 on a test signal), not 0. So every clip under 30 s got encoder input the model never saw. On a test signal the largest feature difference was 1.16 before the fix and 0.0000 after. Fix (serving v2): src/asr.py pads the audio to 30 s before faster-whisper; tests/test_release.py checks the encoder input against transformers. Not shown yet: that this explains the gap (in v1, short clips, which are mostly padding, show no net extra errors).
- 2026-10-28: Fixed before the v2 checks run: float32 first, then int8, serving v2, the same 500 clips, Colab CPU (CPU model recorded). (a) If v2 float32 is within 0.5 points of the official model, the serving code is accepted and the original rule picks the format (int8 if within 1.0 point, else float32). (b) If not, a precision control runs: the original transformers model in float32 on GPU on the full val split (scripts/evaluate.py --dtype float32). If v2 float32 is within 0.5 points of transformers float32 on the same 500 clips, the serving code is accepted (the rest is the model's own sensitivity to float16 vs float32) and the original rule applies; otherwise nothing is published and debugging continues. The v1 numbers stay in this log; the CTranslate2 model card reports v2 numbers.
- 2026-10-28: Serving v2 checks (Colab CPU, AMD EPYC 7B12, 8 vCPUs; results/runs/ct2_check_float32_v2 and ct2_check_int8_v2). Token suppression: faster-whisper and transformers block the same 88 tokens. float32: 22.33% vs official 22.42% on the same 495 scored clips (-0.09 points), 491/500 outputs identical (v1: 314/500); the 9 changed clips are small (3 worse, by at most 2 errors each; 6 better). int8: 22.47% (+0.06 points), 375/500 identical, 44 clips worse and 46 better. The looping clip behind most of the v1 gap now matches the official output (float32) or adds 2 words (int8). So the padding mismatch caused the v1 gap. Rule (a): v2 float32 is within 0.5 points, so the serving code is accepted. Original rule: int8 is within 1.0 point, so int8 ships. scripts/publish_model.py now refuses to publish unless both v2 checks exist and pass.
- 2026-10-28: Released, public, CC BY-SA 4.0: mddalwai/whisper-small-edacc (Hub commit c55fcc9, 971 MB: transformers weights of E3 seed 42, processor files, model card) and mddalwai/whisper-small-edacc-ct2 (Hub commit fc9badc, 253 MB, int8). Checked without login: these are the only public models on the account; mddalwai/speech-ft-E3 is not visible. Model card fixed before upload: loop description (6 clips under 8 s), comparability with the EdAcc paper, seed std label, int8 CPU note.
- 2026-10-28: Serving API (src/serve): POST /transcribe takes the audio file as the request body (any common format; at most 10 MB and 30 s), decoded in memory with the same mono averaging and soxr resampling as src/data.py (tested equal), never written to disk. One transcription at a time: at most 30 s wait (else 503) and 60 s per answer (else 504). GET /health answers only after model load and warm-up, reporting both times. Logs hold timings only, no transcripts or audio.
- 2026-10-28: Docker image, built on the Mac (ARM, so for orientation only): 1.15 GB on disk, 373 MB compressed. Container ready 1.8 s after docker run (model load 0.17 s, warm-up 0.85 s); a 3 s clip took 1.01 s. Runs as user app; /app holds only configs, model, requirements-serve.lock and src.
- 2026-10-28: Space measurement plan (fixed before the Space exists): free CPU Space (cpu-basic), CPU_THREADS=2. Latency: 5 text-to-speech clips of about 3 to 27 s, one warm-up request, then each clip 5 times in a row; per clip, report the median and maximum server processing time and round-trip time (round trip includes the network from the Mac). Cold start: restart the Space 3 times through the API and time until /health answers, reported with the load and warm-up times /health gives. These are the only speed numbers in the README; Mac and Colab speeds are not reported.
- 2026-10-28: Hugging Face refused to create the Space (402: since July 2026, new Gradio and Docker Spaces on free cpu-basic need a PRO subscription). Options considered: PRO ($9/month), Google Cloud Run (same image, free tier, card needed), or no live demo. Chose PRO. The hardware is still cpu-basic, so the measurement plan above is unchanged. If PRO is cancelled the demo may stop; the model repos stay public either way.
- 2026-10-28: Space measurements, run 1 (results/serving/space_latency_1.json and space_cold_start_1.json). Cold start: restart to ready 15.3, 15.3 and 13.2 s (model load about 0.5 s, warm-up about 2.4 s; the rest is Hugging Face scheduling and container start; waking from sleep was not measured). Latency: server time 2.08 s for a 2.7 s clip up to 4.24 s for a 22.2 s clip; round trip 0.5-0.7 s more. About 2 s is a fixed cost (every clip runs a full 30 s window), plus about 0.02 s per output word. Problem: the clips repeat one 9-word sentence 1, 3, 5, 7 and 9 times, and the model wrote 9, 9, 9, 81 and 114 words (spoken: 9, 27, 45, 63, 81). It dropped or added repeats, so these timings mix clip length with odd output length. Added after seeing run 1, and reported next to it: run 2 with 5 clips of ordinary, non-repeating text of similar lengths, plus the repeated clips again, with all transcripts saved.
