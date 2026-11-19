# Deploying the demo and the phone page

**Status (2026-11-19):** the phone page is live at https://nathaniel47.github.io/secure-tinyml/ (GitHub Pages, deployed from the `pages` workflow). A browser-only version of the demo (all three models and the patches, run client-side) is live at https://nathaniel47.github.io/secure-tinyml/demo.html. The Python Gradio demo is **not hosted**: creating the Space failed with HTTP 402, because Hugging Face now requires a PRO subscription for Gradio and Docker Spaces on CPU (static Spaces stay free). Nothing was bought. Options: pay for PRO; host the Gradio app on another free service (needs its own account); or run it locally (`python app/demo.py`). The Space steps below remain valid if you take the first option.

| What | Where | Built from |
|---|---|---|
| Gradio demo (fp32 vs int8 vs defended int8, patch pasted on) | Hugging Face Space | `python deploy/build.py space out/space` |
| Phone page (ONNX Runtime Web, WASM) | GitHub Pages | `python deploy/build.py site out/site` (the `pages` workflow does this) |

## What is and is not included

- Models (`app/assets/*.onnx`, about 0.5 MB) and the three pre-computed patches (PNG): yes.
- **CIFAR-10 / GTSRB images: no.** Their redistribution terms are unverified (README, "Dataset
  licences"), so the Space has no example images (users upload their own) and the phone page
  has no test set. The accuracy-matches-laptop check therefore runs from the laptop over Wi-Fi
  (below), not from the hosted page.
- No secrets, tokens or `.env` are needed by either. Do not paste a Hugging Face token into
  any file in this repo.

## 1. Gradio demo on a Hugging Face Space

1. Make a free account at huggingface.co and create a new Space: SDK **Gradio**, hardware
   **CPU basic** (free), visibility public (or private first to look at it).
2. Build the upload folder and push it as the Space's own git repo:

   ```bash
   python deploy/build.py space out/space
   cd out/space
   git init -b main
   git remote add space https://huggingface.co/spaces/<your-user>/<space-name>
   git add . && git commit -m "Secure TinyML demo"
   git push space main
   ```

   Hugging Face asks for your username and an access token with write rights when you push
   (type it into the prompt; do not store it in the repo). `README.md` in `out/space` holds
   the Space settings (`sdk_version: 6.32.0`, `app_file: app/demo.py`).
3. The build log appears on the Space page; the first build takes a few minutes. Open the
   Space, upload an image, pick a patch, press Classify, and check that fp32, int8 and
   defended int8 all show predictions.
4. Free Spaces sleep when idle and wake on the next visit (a short delay). The demo is tiny,
   so memory is not a concern.

Before this works you need `app/assets/` (already committed). To rebuild it after retraining:
`python app/make_assets.py` (needs torch).

## 2. Phone page on GitHub Pages

1. In the GitHub repo: Settings > Pages > Build and deployment > Source: **GitHub Actions**.
2. Actions tab > **pages** > Run workflow (it only runs on manual trigger).
3. The page appears at `https://<your-user>.github.io/secure-tinyml/`. Open it on the phone,
   choose the int8 model, press Load model, then take or choose a photo: it shows the top 3
   classes and the time in ms. (It loads ONNX Runtime Web from the jsDelivr CDN, so the phone
   needs internet.)

## 3. The real-phone accuracy check (from the laptop, over Wi-Fi)

This is the check that the phone gives the same answers as the laptop, and it needs the test
images, so it is served from your laptop and never published.

```bash
python web/make_testset.py          # writes web/testset.bin, testset.json, web/models/
cd web
python -m http.server 8000
```

On the phone (same Wi-Fi) open `http://<laptop-ip>:8000`, choose a model, Load model, then
**Run test set**. Copy the JSON it shows (accuracy, `agreement_with_laptop`, median ms per
image, user agent) into LEARNING.md. On a desktop browser this gave 200/200 agreement at about
3.5 ms per image. If the phone page cannot load the int8 model, the int8 operators may be
unsupported on that phone's browser: report that as a result, and try the fp32 model.

## After deploying

- Put the two links in the README header and in PORTFOLIO.md (Step 8).
- Re-run `python deploy/build.py ...` and push again whenever the models change.
- To take things down: delete the Space in its settings; in GitHub, Settings > Pages > unpublish.
