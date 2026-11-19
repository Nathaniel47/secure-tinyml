# LEARNING.md: Secure TinyML

Plain-English notes on what was built and why it matters. One section per step.

## Step 2: Repo setup

**What was built.** An empty but working project: a Python package `securetinyml/` with one folder per part of the plan (data, models, attacks, defenses, eval), a `config.py` holding seeds, attack strengths and the dataset order, three dependency files, a Dockerfile, formatting and secret checks, a placeholder demo, and a GitHub Actions workflow that runs lint and tests.

**Why the dependencies are split in three.**
- `requirements.txt` is what the demo needs (Gradio, ONNX Runtime, numpy, pandas, pillow). No PyTorch, so the demo fits free hosting and CI stays fast.
- `requirements-ml.txt` adds PyTorch, ONNX and the Adversarial Robustness Toolbox. Install it only where you train (Kaggle, Colab or a GPU machine).
- `requirements-dev.txt` adds pytest, ruff and pre-commit.

The demo only loads exported `.onnx` files, so the public app never needs the training stack.

**Secrets.** None needed. `.env` is gitignored, `.env.example` lists names with empty values (a test checks both), and the gitleaks hook blocks commits that look like they hold a key.

**How to check it works.** Create a virtual environment, run `pip install -r requirements-dev.txt`, then `pytest` and `ruff check .`. All should pass.

**Reading.** The pre-commit documentation (pre-commit.com) intro page: hooks in five minutes.

## Step 3: Data pipeline

**What was built.** Four small modules in `securetinyml/data/`. `fetch` downloads CIFAR-10 and GTSRB and checks the CIFAR-10 md5. `load` turns the raw archives into small uint8 arrays (CIFAR-10 32 px; GTSRB cropped to the sign box and resized to 32 px) and caches them. `prepare` makes a train/validation split for seeds 0, 1 and 2 and saves it. `report` writes `results/data_report.md` and a grid of sample images per dataset. A PlantVillage loader exists and is tested on synthetic images, but the data is not downloaded yet.

**Why splits are saved with a hash.** A split is just a list of image numbers. If every experiment uses the same list, differences between models come from the models, not from luck in the split. The lists are regenerated from the seed in seconds; `data/split_digests.json` stores a short hash of each, so you can confirm that a rebuild is identical without committing the lists.

**Why GTSRB is split by track.** The training set has 1,307 physical signs, each photographed about 30 times as a car drives past. Those 30 frames are almost the same picture. If frames of one sign land in both training and validation, validation accuracy is inflated because the model has effectively seen the answer. So whole tracks stay together. CIFAR-10 has no such problem. The official test sets are used untouched.

**Problems the report found.**
- GTSRB classes are very uneven (210 to 2,250 images, 10.7x), so accuracy alone can hide a bad rare class; per-class numbers will be reported later.
- 8 GTSRB test images are byte-identical (after resizing) to training images: a tiny leak (0.06% of the test set), recorded rather than removed so the official test set stays official.
- Many GTSRB signs are small (median 43 px wide, down to 25) and become blurry when scaled to 32 px; patch attacks on such images will cover relatively more of the sign.
- CIFAR-10 is perfectly balanced with no duplicates or train/test overlap.

**Not done.** PlantVillage is not downloaded (no stable direct link, and its licence is unverified, see the README). It is the optional third dataset in PLAN.md. Also no LICENSE file for the code yet.

**How to check it.** `pytest` (14 tests: splits are disjoint, complete, stratified, reproducible, keep groups together, change when the seed changes; loaders handle the CIFAR, GTSRB and PlantVillage layouts), then the three commands in the README.

**Reading.** Stallkamp et al. 2012, "Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition" (Neural Networks), section on the dataset: why the track structure matters.

## Step 4: Attacks on full vs quantized models

**What was built.** A small ResNet (0.15M parameters) trained on CIFAR-10, exported to ONNX and quantized to int8, plus four attacks and a runner that measures them on both versions. All numbers below are **one seed (0), CIFAR-10 only, 1,000 test images (200 for the black-box attack)**: a first look, not a result. Repeats with several seeds, GTSRB and the report come in Step 6.
- `securetinyml/attacks/evasion.py`: FGSM and PGD, written by hand, plus `random_search`, a query-only attack for models with no usable gradient.
- `attacks/patch.py`: an adversarial patch, a visible square learned by gradient descent so that, pasted anywhere, it pushes the image towards a chosen class (airplane here).
- `attacks/backdoor.py`: a BadNets-style backdoor: a 4x4 black-and-white checker in the corner, stamped on 5% of the training images, which are relabelled "airplane".
- `models/quantize.py`: ONNX export, static int8 quantization (what would ship to the phone), and a **fake-quant surrogate**.
- `eval/run_attacks.py` writes `results/attacks_cifar10_seed0.csv`; `models/train.py` trains (with checkpoints, so a crash loses at most one epoch).

**Why a surrogate.** Rounding to int8 has zero gradient, so a gradient attack cannot be run through the real int8 model. The surrogate does the same rounding but lets the gradient pass straight through (a "straight-through estimator"). It is not QAT (no retraining) and not the same arithmetic as ONNX Runtime's kernels; it is an approximation the attacker can differentiate.

**Key numbers (CIFAR-10, seed 0)**
- Clean accuracy: fp32 **79.9%**, int8 **72.3%**. Only 83.5% of predictions agree. That is a large quantization loss, probably because the model is tiny, calibration used 500 images with min/max ranges and there was no quantization-aware training. It also confounds everything below that compares the two.
- FGSM/PGD (accuracy under attack, crafted on fp32, eps in 1/255): at eps 2, PGD takes fp32 to 3.9% and int8 to 11.1%; at eps 4 PGD gives 0.0% and 0.8%; at eps 8 both are about 0%. Of the adversarial examples that fool fp32, 91-100% also fool int8 ("transfer rate"), so quantization does **not** protect against attacks crafted on the full-precision model. int8 ends up slightly *higher* in accuracy under attack, but it also starts lower, and I would not read that as robustness.
- Surrogate vs fp32: attacks crafted on the surrogate are almost identical to those crafted on fp32 (e.g. PGD eps 2 gives 11.1% on int8 both ways), so with this model the surrogate adds little.
- Black-box check (query-only, 300 queries, 200 images, int8): accuracy 57.5% at eps 2, 44.0% at eps 4, 21.5% at eps 8. Weaker than white-box but far from "robust": it rules out the idea that int8 simply hides its gradients.
- Adversarial patch (target class airplane, share of non-airplane images sent to airplane, patch covers 4/7/10% of the image): fp32 20% / 49% / 85%; int8 18% / 38% / 76%. A patch trained against fp32 transfers to int8 with a modest drop.
- Backdoor (5% poison): the poisoned fp32 model sends **100%** of triggered images to airplane at 77.9% clean accuracy, but the poisoned int8 model only **48%** at 73.3% clean accuracy. So quantization weakened this backdoor, but the int8 model is also generally worse, I did not measure how often an *un*poisoned model sends triggered images to airplane (a baseline), and it is one seed. Treat as a lead to test in Step 6, not a finding.

**What can go wrong with these results.** One seed, one dataset, 1,000 images (about +-1.3 points of sampling error on 80%). The int8 model loses 7.6 points of clean accuracy, so int8-vs-fp32 comparisons are not like for like. Calibration data were clean training images. The random-search attack is simplified, not the published Square Attack. GTSRB and PlantVillage were not run.

**How to run and check.** `pytest` (31 tests; the torch and ONNX ones are skipped on CI without torch): FGSM with eps 0 changes nothing, perturbations stay within eps and in [0, 1], accuracy falls as eps grows, PGD is at least as strong as FGSM, a trained patch raises the target rate, only the patch region changes, the trigger only touches its corner, poisoning hits the right rate and classes, folded BatchNorm matches the original, the surrogate rounds to 256 levels and passes gradients, and the int8 ONNX file is smaller and runs. To reproduce the numbers (CPU, about 4 minutes per epoch):

```bash
pip install -r requirements-ml.txt
python -m securetinyml.models.train cifar10 --epochs 10 --seed 0
python -m securetinyml.models.train cifar10 --epochs 8 --seed 0 --poison-rate 0.05
python -m securetinyml.eval.run_attacks cifar10 --seed 0 --n 1000 --bd 0.05
```

**Concepts and one reading each**
- *FGSM / PGD*: one or many gradient-sign steps that increase the loss inside a small box. Read Madry et al. 2018, "Towards Deep Learning Models Resistant to Adversarial Attacks" (arXiv).
- *Adversarial patch*: a visible, location-independent sticker. Read Brown et al. 2017, "Adversarial Patch" (arXiv).
- *Backdoor / poisoning*: a hidden trigger planted through the training data. Read Gu et al. 2017, "BadNets" (arXiv).
- *Quantization*: int8 weights and activations. Read Jacob et al. 2018, "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (arXiv).
- *Straight-through estimator*: pretend rounding has gradient 1. Read Bengio et al. 2013, "Estimating or Propagating Gradients Through Stochastic Neurons" (arXiv).
- *Gradient masking*: why a failed gradient attack proves little. Read Athalye et al. 2018, "Obfuscated Gradients Give a False Sense of Security" (arXiv).
