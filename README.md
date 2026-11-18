# Secure TinyML: Attacking and Defending Quantized Models

A study platform for adversarial patches, FGSM/PGD and backdoors on small image models, comparing full-precision (fp32) against int8-quantized versions, with cheap defenses (adversarial training, input purification, activation clipping) and a Gradio demo. A static page runs the int8 model in the browser on a phone with ONNX Runtime Web.

Status: repo setup (step 2) done; no experiments yet. See [PLAN.md](PLAN.md) for the design.

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

Training and attacks need `requirements-ml.txt` (torch, ART) and run best on Kaggle or Colab.

Docker: `docker compose up` serves the demo on http://localhost:7860.

## Dataset licences

To be filled in week 1 from each official page (CIFAR-10, GTSRB, PlantVillage). See PLAN.md section 5.
