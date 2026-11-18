# CLAUDE.md: Secure TinyML coding rules

- Follow PLAN.md; if anything conflicts, PLAN.md wins. Work one step at a time.
- Python 3.11. Plain PyTorch for training and attacks. FGSM and PGD are hand-written; ART is a cross-check and source of poisoning helpers only (pin its version).
- One quantized format for evaluation and the browser: ONNX Runtime static int8 (QDQ). PyTorch QAT is the differentiable surrogate for gradient attacks. TFLite is an optional Colab cross-check only.
- Never claim an int8 model is robust because a gradient attack failed on it. Compare against a black-box or QAT-surrogate attack to rule out gradient masking. Label defense results "adaptive" or "non-adaptive".
- Raw datasets are never committed or redistributed. Licences for CIFAR-10, GTSRB and PlantVillage are unverified (see PLAN.md); record exact terms from the official pages in the README before any public example images.
- No secret is required. Never commit `.env`; `.env.example` has names with empty values.
- Training and attack grids run in Kaggle/Colab notebooks (or locally) and write one CSV row per (model, attack, defense, seed); charts and the demo read from files. CI installs no torch; tests needing torch must skip or use NumPy fixtures.
- Fixed seeds. Every run records dataset, model variant, attack, strength, defense, seed, library versions and git commit in `results/run_meta.json` or its row.
- Format and lint with ruff (line length 100). Write tests in `tests/` for every feature; run `pytest` before each commit.
- Code comments only for non-obvious reasons. Plain-English explanations go in LEARNING.md, not chat.
- Commit with the step name in the message, then push (if a remote exists).
