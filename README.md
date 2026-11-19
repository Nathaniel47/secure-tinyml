# Secure TinyML: Attacking and Defending Quantized Models

A study platform for adversarial patches, FGSM/PGD and backdoors on small image models, comparing full-precision (fp32) against int8-quantized versions, with cheap defenses (adversarial training, input purification, activation clipping) and a Gradio demo. A static page runs the int8 model in the browser on a phone with ONNX Runtime Web.

Status: repo setup, data pipeline (CIFAR-10, GTSRB), the attacks (FGSM, PGD, patch, backdoor on CIFAR-10, fp32 vs int8), three defenses, the Gradio demo and a browser test page are built. Evaluated on CIFAR-10 and GTSRB with 3 seeds (PlantVillage not run). The real-phone test, deployment and the write-up are still to do. Results so far are one seed and preliminary (int8 now matches fp32 accuracy after a kernel fix): see LEARNING.md (Steps 4-5) and `results/*.csv`. See [PLAN.md](PLAN.md) for the design.

Small models, small images, one phone: this is a study, not a claim about production edge devices. Dataset licences are not yet verified (see PLAN.md); no raw data is redistributed.

## Run locally (laptop: tests and demo, no GPU)

Windows (PowerShell):
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
python app/demo.py
```

macOS / Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
python app/demo.py
```

The demo needs `python app/make_assets.py` once (torch) to build its models, patches and example images; the browser page needs `python web/make_testset.py`. See LEARNING.md (Step 5).

Training and attacks need `requirements-ml.txt` (torch, ART) and run best on Kaggle or Colab.

Docker: `docker compose up` serves the demo on http://localhost:7860.

## Data

```bash
python -m securetinyml.data.fetch      # CIFAR-10 (170 MB) and GTSRB (365 MB) into data/raw
python -m securetinyml.data.prepare    # cache arrays, make seeded train/val splits (seeds 0-2)
python -m securetinyml.data.report     # writes results/data_report.md
```

Raw data, cached arrays and split indices are gitignored; `data/split_digests.json` holds a hash per split so a rebuilt split can be checked. PlantVillage is not downloaded yet (no stable direct link; see `securetinyml/data/fetch.py`). Results so far: [results/data_report.md](results/data_report.md).

## Dataset licences and terms

Checked 2026-11-18. **None of these gives an authoritative, machine-readable licence I could confirm; treat all as research use and do not redistribute the raw data.**

| Dataset | Source used | What the official page says | Status |
|---|---|---|---|
| CIFAR-10 | cs.toronto.edu/~kriz/cifar.html | Asks users to cite the Krizhevsky 2009 tech report. No licence text on the page. Third-party tables say MIT or "unknown". Images were crawled from the web (subset of 80M Tiny Images). | Research use, cite. Archive md5 matched `c58f30108f718f92721af3b95e74349a`. |
| GTSRB | ERDA mirror linked from the INI benchmark repo (the old benchmark.ini.rub.de site now redirects to github.com/houbensebastian/GermanTrafficSignBenchmarks) | Asks to cite Stallkamp et al., IJCNN 2011 and Neural Networks 2012. No licence stated (the repo has none). A Hugging Face mirror says GPL-3.0, which is not the authors' statement. | Research use, cite. No checksum published; sha256 prefixes printed by `fetch`. |
| PlantVillage | not downloaded | Conflicting: CC0 1.0 (Mendeley record per Meta-Album) vs CC BY-SA 3.0 (a Zenodo copy, one paper). Official Mendeley page not opened. | **Unverified.** Decide before using it. |

Code is MIT-style; add a LICENSE file before announcing the repo (not done yet).

## Results and limitations (Step 6)

Charts: `results/chart_pgd_fp32_vs_int8.png`, `results/chart_defenses.png`, `results/chart_patch_backdoor.png`. Tables with intervals: LEARNING.md, "Step 6". Raw rows: `results/*.csv`; what produced them: `results/run_meta.json`.

Headline: after fixing an int8 kernel setting, int8 matches fp32 on accuracy and under attack, and attacks crafted on fp32 transfer to int8 almost completely. Adversarial training is the only defense that clearly works, including against an adaptive attacker. Input purification helps only against an attacker who does not know about it, and activation clipping did nothing here. A BadNets backdoor is weakened by int8 (one seed per rate).

Limitations: 3 seeds and 500 images per grid; backdoor results are one seed per rate; PGD with 10 steps and no AutoAttack, so robust numbers are upper bounds; one tiny ResNet at 32x32; one quantization toolchain; the real-phone test is not done yet; PlantVillage, backdoor detection and TFLite were not run; dataset licences are not yet verified.
