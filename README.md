# HPAA

This repository contains the code and data for the paper:

> **What the Eyes See, the LLMs Miss: Exploiting Human Perception for Adversarial Text Attacks**

---

## Repository Structure

```
.
├── HPAA.py                  # Main entry point (generation & evaluation)
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
│   ├── SurveyDataRound1/           # (see README.md inside)
│   └── SurveyDataRound2/           # (see README.md inside)
└── HPAA/                            # Output folder for generated samples
    └── demo.*.csv                   # Pre-generated demo outputs
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

To quickly verify the main claims of the paper:

```bash
# 1. Generate adversarial samples (Option D, reproduces paper setup)
python HPAA.py -bc Hotel -b text -tc Advbench_10 -t text \
  -m M1 -l W -s Hi --seed 42 -p adv

# 2. Evaluate (e.g., Perspective API — free tier, easiest to set up)
python HPAA.py -f ./HPAA/adv.M1-W-Hi.csv -dn perspective_api -ep verify

```

To sweep all configurations tested in the paper, vary `-m` (M1–M6), `-l` (W, T, Mix), and `-s` (B, Col, Hi, Pre, Cap, Cloze).

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

For detailed instructions on reproducing the user study analyses (software requirements, how to run the notebooks, expected outputs), see the README files inside each folder:

- [`user_study/SurveyDataRound1/README.md`](user_study/SurveyDataRound1/README.md)
- [`user_study/SurveyDataRound2/README.md`](user_study/SurveyDataRound2/README.md)

Additional user study materials (survey instruments, consent forms, and study documentation) are permanently archived on OSF (Open Science Framework): https://osf.io/tn2vw/overview?view_only=af5cc0f70492497ca773b58155a333c2
