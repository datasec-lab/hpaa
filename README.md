# HPAA

This repository contains the code and data for the paper:

> **What the Eyes See, the LLMs Miss: Exploiting Human Perception for Adversarial Text Attacks**

### Artifacts

All artifacts are included in this repository, including implementation code, evaluation scripts, datasets, user study analysis notebooks, and survey instruments.

---

## Repository Structure

```
.
├── HPAA.py                  # Main entry point (generation & evaluation)
├── quick_test.sh            # Reproduce all main paper results end-to-end
├── quick_test.py            # Summarize evasion rates (1-shot & 3-shot) from results
├── run.sh                   # Runnable examples for all options
├── requirements.txt         # Python dependencies
├── .env.example             # API key template (copy to .env and fill in)
├── LICENSE                  # MIT License
├── src/
│   ├── __init__.py
│   ├── gen_HPAA.py          # Adversarial sample generation
│   ├── eval_HPAA.py         # Detector evaluation
│   ├── detectors.py         # Detector implementations
│   └── detectors.yaml       # Detector configurations & prompts
├── data/
│   ├── toxic.Advbench_10.csv        # Toxic text dataset (249 samples)
│   ├── benign.Hotel.csv             # Benign corpus: hotel reviews
│   ├── benign.Movie.csv             # Benign corpus: movie reviews
│   ├── benign.Restaurant.csv        # Benign corpus: restaurant reviews
│   ├── benign.Music.csv             # Benign corpus: music reviews
│   ├── benign.Product.csv           # Benign corpus: product reviews
│   ├── Dataset_I.csv                # Phase I user study
│   ├── Dataset_II.csv               # Phase II user study
│   └── HED_top6.csv                 # Top-K Configurations Dataset
├── user_study/                      # Raw user study data & analysis
│   ├── Survey/                      # Survey instruments (PDFs)
│   ├── SurveyDataRound1/           # (see README.md inside)
│   └── SurveyDataRound2/           # (see README.md inside)
└── HPAA/                            # Output folder for generated samples
    ├── optA.M1-W-Hi.csv             # Pre-generated demo (generation)
    └── exp1.M1-W-Hi.perspective_api.csv  # Pre-generated demo (evaluation)
```

---

## Getting Started

### 1. Environment Setup

```bash
# Download the repository
cd hpaa

# Install dependencies
pip install -r requirements.txt

# For local GPU detectors (Llama Guard, ShieldGemma), also install:
# pip install torch>=2.0 transformers>=4.40 accelerate>=0.25
```

### 2. API Keys

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your actual keys:

```
perspective_api_key = "YOUR_KEY"
gemini_api_key      = "YOUR_KEY"
openai_api_key      = "YOUR_KEY"
enkryptai_api_key   = "YOUR_KEY"
azure_api_key       = "YOUR_KEY"
client_id           = "YOUR_AWS_ACCESS_KEY_ID"
client_secret       = "YOUR_AWS_SECRET_ACCESS_KEY"
```

### 3. Local Model Weights

For GPU-based detectors, download model weights and update `download_path` in `src/detectors.yaml`:

```bash
pip install huggingface_hub

# Llama Guard (requires Meta approval: https://huggingface.co/meta-llama/Llama-Guard-3-8B)
huggingface-cli download meta-llama/Llama-Guard-3-8B --local-dir /path/to/models/Llama-Guard-3-8B

# ShieldGemma
huggingface-cli download google/shieldgemma-2b --local-dir /path/to/models/shieldgemma-2b
huggingface-cli download google/shieldgemma-9b --local-dir /path/to/models/shieldgemma-9b
```

Then set `download_path` in `src/detectors.yaml` to `/path/to/models/`.

---

## Usage

The system operates in two modes: **generation** (create adversarial samples) and **evaluation** (test against detectors). See `run.sh` for full runnable examples.

### Step 1: Generate Adversarial Samples

Four options for specifying benign/toxic inputs:

| | Toxic: user-specified | Toxic: from dataset |
|---|---|---|
| **Benign: user-specified** | Option A (1 sample) | Option B (N samples) |
| **Benign: sampled from corpus** | Option C (1 sample) | Option D (N samples) |

**Option D** (recommended for reproducing paper results):

```bash
python HPAA.py \
  -bc Hotel -b text \
  -tc Advbench_10 -t text \
  -m M1 -l W -s Hi \
  --seed 42 -p adv
```

Output: `./HPAA/adv.M1-W-Hi.csv` (249 samples — one per toxic sentence, each paired with a different randomly-sampled benign sentence)

**Option A** (quick single-sample test):

```bash
python HPAA.py \
  -bc Given -b "nice hotel expensive parking got good deal stay hotel anniversary, arrived late evening took advice previous reviews did valet parking, check quick easy, little disappointed non-existent view room room clean nice size, bed comfortable woke stiff neck high pillows." \
  -tc Given -t "YOU ARE WORTHLESS EVERYONE HATES YOU" \
  -m M1 -l W -s Hi \
  -p demo
```

### Step 2: Evaluate Against Detectors

```bash
EVAL_INPUT="./HPAA/adv.M1-W-Hi.csv"
EVAL_PREFIX="exp1"

# API detectors
python HPAA.py -f $EVAL_INPUT -dn perspective_api    -ep $EVAL_PREFIX
python HPAA.py -f $EVAL_INPUT -dn gemini-2.0-flash   -ep $EVAL_PREFIX
python HPAA.py -f $EVAL_INPUT -dn omni-moderation-latest -ep $EVAL_PREFIX

# Local GPU detectors
python HPAA.py -f $EVAL_INPUT -dn Llama-Guard-3-8B   -ep $EVAL_PREFIX
python HPAA.py -f $EVAL_INPUT -dn shieldgemma-2b     -ep $EVAL_PREFIX \
  --tau 0.7 --bias_yes 0.2 --bias_no 0.0 --min_margin 0.0
```

Results are saved to `./HPAA/<eval_prefix>.<detector_name>.<timestamp>.csv`.

### Step 3: Quick Test — Reproduce Paper Results

`quick_test.sh` reproduces Table 2, Figure 5 (1-shot & 3-shot evasion), and Figure 6 / Table 3 / Table 6 (threshold sweep) using Perspective API.

```bash
conda activate hpaa

# Generate adversarial samples (skip if pre-generated files already exist in HPAA/)
bash quick_test.sh gen

# Evaluate a random subset for a quick check (e.g. 10 samples, uses seed 42)
bash quick_test.sh eval 10

# Or evaluate — full dataset (249 samples per config)
bash quick_test.sh eval

# Summarize results: prints per-file evasion rates, 1-shot (Table 2), and 3-shot (Figure 5)
python quick_test.py
```

---

## Supported Detectors

| Detector | Provider | Type | Notes |
|---|---|---|---|
| `perspective_api` | Google | API | Free tier available |
| `gemini-2.0-flash` | Google | API | Free tier available |
| `gemini-2.5-flash-lite` | Google | API | |
| `omni-moderation-latest` | OpenAI | API | |
| `gpt-4o` | OpenAI | API | |
| `gpt-3.5-turbo` | OpenAI | API | |
| `enkryptai` | EnkryptAI | API | |
| `comprehend` | AWS | API | |
| `azure_ai_content_safety_api` | Microsoft | API | |
| `Llama-Guard-3-8B` | Meta | Local | ~16 GB VRAM |
| `shieldgemma-2b` | Google | Local | ~5 GB VRAM |
| `shieldgemma-9b` | Google | Local | ~18 GB VRAM |

---

## Using Your Own Datasets

No code changes needed. Follow this naming convention:

**Benign corpus:** save as `./data/benign.<Name>.csv` with a text column, then use:
```bash
-bc <Name> -b <column_name>
```

**Toxic corpus:** save as `./data/toxic.<Name>.csv` with a text column, then use:
```bash
-tc <Name> -t <column_name>
```

---

## Artifact Evaluation — Quick Verification

To reproduce the main paper results (Table 2, Figure 5, Figure 6):

```bash
# Requires: perspective_api_key set in .env
bash quick_test.sh all        # generate + evaluate full dataset (~249 samples × 18 runs)
python quick_test.py          # print 1-shot and 3-shot evasion rates
```

For a fast end-to-end sanity check (10 random samples, ~3 minutes):

```bash
bash quick_test.sh all 10
python quick_test.py
```

To test other detectors, use `HPAA.py` directly with `-dn <detector_name>` and optionally `-n <N>` to limit samples:

```bash
python HPAA.py -f ./HPAA/optD.Hotel.Advbench_10.M6-W-Hi.csv \
  -dn gemini-2.0-flash -ep verify -n 10
```

---

## Datasets

| Dataset | Location | Description |
|---|---|---|
| Toxic text | `data/toxic.Advbench_10.csv` | 249 toxic sentences |
| Benign text | `data/benign.*.csv` | Hotel / Movie / Restaurant / Music / Product reviews |
| User Study I | `data/Dataset_I.csv` | Phase I user study |
| User Study II | `data/Dataset_II.csv` | Phase II user study |
| HED | `data/HED_top6.csv` | Human Evaluation Dataset (top-6 configs) |
| Survey data | `user_study/` | Raw survey responses and analysis notebooks |
| Survey instruments | `user_study/Survey/` | Survey questionnaire PDFs for Round I and Round II |

For detailed instructions on reproducing the user study analyses (software requirements, how to run the notebooks, expected outputs), see the README files inside each folder:

- [`user_study/SurveyDataRound1/README.md`](user_study/SurveyDataRound1/README.md)
- [`user_study/SurveyDataRound2/README.md`](user_study/SurveyDataRound2/README.md)

User study survey instruments are included in [`user_study/Survey/`](user_study/Survey/).
