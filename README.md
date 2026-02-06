## hpaa

### Modes

- **Generation mode** (default)  
  If `--file_eval` is **not** provided, the script generates HPAA samples and saves them to `--hpaa_folder`.

- **Evaluation mode**  
  If both `--file_eval` **and** `--detector_name` are provided, the script evaluates the specified file(s) using the selected detector.  
  Note that `--file_eval` may include multiple files produced in Generation mode, but run Evaluation mode using a **single** detector specified by `--detector_name`.

---

### Demo

Before using any detectors, make sure to update `.env` with your own API keys and install the dependencies listed in `requirements.txt`. Additionally, configure the `download_path` field in `./src/detectors.yaml` to specify where downloaded open-source models (e.g., Llama Guard 8B, ShieldGemma-2B, ShieldGemma-9B) should be stored.

Run `run.sh` to see examples for generating HPAA samples or evaluating the generated samples.

If you need to specify which GPU to use, you can set the environment variable before running the script.  
For example, `env["CUDA_VISIBLE_DEVICES"] = "0"` specifies that GPU 0 should be used.

The raw detector outputs will be saved, and these results are then aggregated to compute the final detection rate.

---

### Datasets

- **Short Toxic Text Dataset**  
  Location: `./data/toxic.Advbench_10.csv`

- **Benign Text Dataset**  
  Location: `./data/benign.*.csv`

- **User Study Dataset I**  
  Location: `./data/Dataset_I.csv`

- **User Study Dataset II**  
  Location: `./data/Dataset_II.csv`

- **HED**
  Location: `./data/HED_top6.csv`
