# Decisions log

## Constraints
- GPU: Colab Pro, [__] compute units on 2026-10-24. L4 for development and all reported evaluation; A100 for full training runs only.
- Hours per week: [__]
- Start: 2026-10-24. Hard stop: [__]

## Plan changes made before seeing any data (2026-10-24)
- Added baseline E0c: zero-shot whisper-large-v3-turbo (evaluation only).
- Gate 1 uses a fixed sample of about 1,000 val utterances per dataset.
- Common Voice dropped unless EdAcc and L2-ARCTIC both fail the headroom rule.
- One dataset per released model (license reasons).
- All reported numbers use the same GPU type (L4) and the same precision.
