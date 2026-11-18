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
