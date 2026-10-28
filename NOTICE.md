# Licenses in this repository

- **Code** (`src/`, `scripts/`, `tests/`, `configs/`): MIT, see `LICENSE`.
- **Text derived from EdAcc** (transcripts and model outputs in `results/`, segment lists in `splits/`):
  CC BY-SA 4.0, as EdAcc itself. Source: Sanabria et al., "The Edinburgh International Accents of English Corpus:
  Towards the Democratization of English ASR", ICASSP 2023 (https://huggingface.co/datasets/edinburghcstr/edacc).
  Changes: segments filtered and split by the rules in `docs/decisions.md`; transcripts normalized for scoring.
- **Text derived from LibriSpeech** (transcripts and model outputs in `results/runs/*libri*`): CC BY 4.0.
  Source: Panayotov et al., "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015.
- **Models** (published separately on Hugging Face, not in this repository): CC BY-SA 4.0, see their model cards.
  Base model: OpenAI Whisper (`openai/whisper-small`), Apache 2.0.

No audio from either dataset is stored in this repository.
