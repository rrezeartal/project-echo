# project-echo

LoRA fine-tuning of `openai/whisper-medium` for Albanian (sq) and Macedonian (mk) ASR. The experiment compares the base model with two LoRA variants: one trained on the uncleaned data and one trained on data curated with LaBSE (contextual) sentence embeddings. A Chainlit app serves all of them.

## Setup (Windows, local GPU)

1. Install PyTorch with CUDA, then the remaining requirements:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```
2. Install FFmpeg and add it to `PATH`. It is needed to decode MP3/WebM uploads, and `datasets>=4` needs it for audio decoding. If `datasets` 4 cannot decode audio on Windows, run `pip install "datasets[audio]<4"`.
3. Create a `.env` file in the repo root containing `HF_TOKEN=<your token>`.
4. Accept the dataset terms on Hugging Face for `Kushtrim/common_voice_24_sq` and `akadriu/albanian-asr-10k-processed` (gated).

## Data

| Language | Train | Test |
|---|---|---|
| sq | `Kushtrim/common_voice_24_sq` (train), `akadriu/albanian-asr-10k-processed` (optional, skipped if it has no raw audio) | Common Voice 24 test |
| mk | `shunyalabs/macedonian-speech-dataset` (train+validation), `google/fleurs` `mk_mk` (train+validation) | shunyalabs test + FLEURS test |

Training sentences that also appear in the test set (after normalization) are removed in both variants.

## Workflow

```powershell
# 1. Optional: inspect data and run LaBSE curation (training runs it automatically if missing)
#    notebooks/01_explore_and_curate.ipynb

# 2. Train (4 runs)
python scripts/train_lora.py --lang sq --arm raw
python scripts/train_lora.py --lang sq --arm labse
python scripts/train_lora.py --lang mk --arm raw
python scripts/train_lora.py --lang mk --arm labse
# Add --load_in_8bit --batch_size 4 --grad_accum 8 on GPUs with < 12 GB VRAM.

# 3. Evaluate (6 runs)
python scripts/evaluate.py --lang sq --model base
python scripts/evaluate.py --lang sq --model raw
python scripts/evaluate.py --lang sq --model labse
python scripts/evaluate.py --lang mk --model base
python scripts/evaluate.py --lang mk --model raw
python scripts/evaluate.py --lang mk --model labse

# 4. Compare: notebooks/02_results.ipynb

# 5. Demo
chainlit run app/app.py
```

Adapters are saved to `outputs/whisper-medium-lora-{lang}-{arm}/`, and metrics (WER, CER, normalized WER/CER and LaBSE semantic similarity) go to `results/`.