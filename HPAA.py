import os, argparse
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

from itertools import product
from src.gen_HPAA import gen_HPAA, M, L, S
from src.eval_HPAA import eval_HPAA


"""
>>> 1. Typographic Cues Set, H = list(product(M, L, S))
>>> 2. (Phase I User Study)  -> H_top21 (Top-21 in H, namely Top-21 Typographic Cues Set)
>>> 3. (Phase II User Study) -> H'      (Top-10 in H_top21, HPAA Configurations Set)
"""

# Built-in dataset names (benign / toxic).
# To use your own dataset, place a CSV in --b_dataset_folder or --t_dataset_folder:
#   Benign: benign.<YourName>.csv  (must contain the column 'text' by --benign)
#   Toxic:  toxic.<YourName>.csv   (must contain the column 'text' by --toxic)
# Then pass --benign_sentence_choice <YourName> or --toxic_sentence_choice <YourName>.
BUILTIN_BENIGN = ["Hotel", "Movie", "Restaurant", "Music", "Product"]
BUILTIN_TOXIC  = ["Advbench_10"]


def get_args():
    parser = argparse.ArgumentParser(
        description="HPAA: Generate or evaluate adversarial samples.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Folders ──────────────────────────────────────────────
    parser.add_argument(
        "-bf", "--b_dataset_folder",
        type=str, default="./data",
        help="Folder for benign dataset (must exist)"
    )
    parser.add_argument(
        "-tf", "--t_dataset_folder",
        type=str, default="./data",
        help="Folder for toxic dataset (must exist)"
    )
    parser.add_argument(
        "-hf", "--hpaa_folder",
        type=str, default="./HPAA",
        help="Folder for HPAA outputs (created if not exists)"
    )

    # ── Benign sentence ──────────────────────────────────────
    parser.add_argument(
        "-bc", "--benign_sentence_choice",
        type=str, default="Given",
        help="'Given' = use the sentence in --benign directly.\n"
             "Otherwise, name of a benign corpus: the code looks for\n"
             "  <b_dataset_folder>/benign.<choice>.csv\n"
             f"Built-in: {BUILTIN_BENIGN}\n"
             "Custom:   place benign.<YourName>.csv in --b_dataset_folder"
    )
    parser.add_argument(
        "-b", "--benign",
        type=str, default=None,
        help="When --benign_sentence_choice is 'Given': the benign sentence.\n"
             "When using a corpus: the column name to read (e.g. 'text')."
    )

    # ── Toxic sentence ───────────────────────────────────────
    parser.add_argument(
        "-tc", "--toxic_sentence_choice",
        type=str, default="Given",
        help="'Given' = use the sentence in --toxic directly.\n"
             "Otherwise, name of a toxic corpus: the code looks for\n"
             "  <t_dataset_folder>/toxic.<choice>.csv\n"
             f"Built-in: {BUILTIN_TOXIC}\n"
             "Custom:   place toxic.<YourName>.csv in --t_dataset_folder"
    )
    parser.add_argument(
        "-t", "--toxic",
        type=str, default="",
        help="When --toxic_sentence_choice is 'Given': the toxic sentence.\n"
             "When using a corpus: the column name to read (e.g. 'str')."
    )

    # ── HPAA configuration (m, l, s) ────────────────────────
    parser.add_argument(
        "-m", "--mode",
        type=str, choices=M, default="M1",
        help=f"Spatial Placement, one of: {M}"
    )
    parser.add_argument(
        "-l", "--granularity",
        type=str, choices=list(L.keys()), default="W",
        help="Granularity of typographic cues: " +
             ", ".join([f"{k} ({v})" for k, v in L.items()])
    )
    parser.add_argument(
        "-s", "--stylistic_transformation",
        type=str, choices=list(S.keys()), default="Hi",
        help="Stylistic transformation: " +
             ", ".join([f"{k} ({v})" for k, v in S.items()])
    )

    # ── Output naming ────────────────────────────────────────
    parser.add_argument(
        "-p", "--adv_prefix",
        type=str, default="adv",
        help="Prefix for the output CSV filename.\n"
             "Output: <hpaa_folder>/<adv_prefix>.<mode>-<gran>-<style>.csv"
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    # ── Evaluation ───────────────────────────────────────────
    parser.add_argument(
        "-f", "--file_eval",
        type=str, nargs="+", default=None,
        help="CSV file(s) to evaluate. If not provided, runs generation mode."
    )
    parser.add_argument(
        "-dn", "--detector_name",
        type=str, default=None,
        help="Detector to use for evaluation. Required with --file_eval."
    )

    # ── Detector hyperparameters ─────────────────────────────
    parser.add_argument("--do_sample", action="store_true", default=None,
                        help="Detector param: Llama Guard")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Detector param: Llama Guard & Gemini")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Detector param: Llama Guard & Gemini")
    parser.add_argument("--tau", type=float, default=None,
                        help="Detector param: ShieldGemma (tau > 0)")
    parser.add_argument("--bias_yes", type=float, default=None,
                        help="Detector param: ShieldGemma")
    parser.add_argument("--bias_no", type=float, default=None,
                        help="Detector param: ShieldGemma")
    parser.add_argument("--min_margin", type=float, default=None,
                        help="Detector param: ShieldGemma")
    parser.add_argument("--top_k", type=float, default=None,
                        help="Detector param: Gemini")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.file_eval is None:
        os.makedirs(args.hpaa_folder, exist_ok=True)
        gen_HPAA(args)
    
    if args.file_eval is not None and args.detector_name is not None:
        eval_HPAA(args)


if __name__ == "__main__":
    main()