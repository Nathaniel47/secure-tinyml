# Attacking and Defending Quantized Models: a small study

Repo: https://github.com/Nathaniel47/secure-tinyml (code, results CSVs, per-step notes in LEARNING.md)

## 1. Problem

Models that run on phones and microcontrollers are usually compressed, most often by quantization: storing weights and activations as 8-bit integers instead of 32-bit floats. Two things are commonly said about this. One is that a quantized model is harder to attack, because its rounding steps have no useful gradient. The other is that compression is a security-neutral engineering detail. This study checks both on small image classifiers, and asks which cheap defenses actually help once the attacker is allowed to know about them.

Questions:
1. Does int8 change accuracy and robustness compared with the same model in fp32?
2. Do attacks crafted on the fp32 model also work on the int8 model that would ship?
3. Does a planted backdoor survive quantization?
4. Which of three cheap defenses (adversarial training, input purification, activation clipping) help, including against an *adaptive* attacker who knows the defense?

Scope: one tiny ResNet (about 0.15M parameters), CIFAR-10 and GTSRB images at 32x32, one quantization toolchain. This is a measurement study at small scale, not a claim about production systems.

## 2. Method

**Data.** CIFAR-10 (50,000 train, 10,000 test) and GTSRB (39,209 train, 12,630 test, 43 classes). Each dataset gets seeded 90/10 train/validation splits for seeds 0-2, with hashes recorded so a rebuilt split can be checked. GTSRB shows each physical sign in about 30 near-identical frames, so it is split by sign (1,307 tracks), otherwise validation images would be near-copies of training images. Dataset licences could not be verified from official pages (README); no images are redistributed.

**Models.** A small ResNet with normalisation inside the model, so attacks and the exported graph both work in plain [0, 1] pixel space. It trains in minutes per epoch on a laptop CPU (CIFAR-10: 10 epochs, 81.7% mean test accuracy; GTSRB: 6 epochs, 97.2%).

**Quantization.** The int8 model is an ONNX Runtime static quantization (QDQ format, per-channel int8 weights, int8 activations, min/max calibration on 500 clean training images). The same `.onnx` file is evaluated in Python and loaded in the browser with ONNX Runtime Web, so the model that is attacked is the model that ships. Because rounding has zero gradient, gradient attacks cannot go through the real int8 graph. Two substitutes are used: a **fake-quant surrogate** (same rounding, with a straight-through gradient; a post-training approximation, not quantization-aware training) and a **query-only attack** (random search on the margin) run directly on the int8 model. The query-only attack is the guard against gradient masking: if a gradient attack fails on an int8 model, that proves nothing until a gradient-free attack agrees.

**Attacks.** FGSM and 10-step L-infinity PGD (eps 2, 4, 8 out of 255), written by hand and checked against the Adversarial Robustness Toolbox 1.20.1 (FGSM identical, PGD within 0.2 points on CIFAR-10). A targeted adversarial patch (a visible square, 4, 7 or 10% of the image, trained to push images to one class, pasted at random positions). A BadNets-style backdoor: a 4x4 black-and-white corner trigger stamped on 1, 2 or 5% of the training images, which are relabelled to the target class.

**Defenses.** (a) Adversarial training: fine-tuning on one-step attacked images at eps 4/255 with a random start, the cheap variant that a CPU can afford. (b) Input purification: round pixels to 4 bits, or 5 bits plus a 3x3 blur, inside the model graph. (c) Activation clipping: cap every ReLU output at the 99.9th percentile seen on clean data. Each defense is measured against three attackers: **non-adaptive** (attack crafted on the undefended model, then shown to the defended one), **adaptive** (attack crafted on the defended model, with a straight-through gradient through the purifier's rounding) and **query-only** on the defended int8 model.

**Protocol.** Every result is one CSV row per (dataset, seed, weights, defense, attack, strength, crafted-on, evaluated-on). Three training seeds per dataset; 500 test images per grid; tables show the mean and a 95% t-interval over seeds. Library versions and the git commit are recorded in `results/run_meta.json`.

## 3. Results

**int8 versus fp32.**

| | CIFAR-10 | GTSRB |
|---|---|---|
| Clean accuracy, fp32 / int8 | 81.7% / 81.7% | 97.2% / 97.1% |
| Same predicted label, fp32 vs int8 | 98.6% | 99.9% |
| PGD eps 2: accuracy fp32 / int8 | 5.5% / 5.6% | 55.5% / 56.0% |
| PGD eps 4: accuracy fp32 / int8 | 0.1% / 0.1% | 24.0% / 24.3% |
| PGD eps 8: accuracy fp32 / int8 | 0.0% / 0.0% | 3.3% / 3.3% |
| Attacks fooling fp32 that also fool int8 | 99.9-100% | 98-100% |
| Query-only attack on int8, eps 2 / 4 / 8 (accuracy) | 72 / 56 / 23% | 89 / 73 / 45% |

Quantization neither protected nor harmed: the fp32-int8 differences are far inside the seed-to-seed intervals, and attacks crafted on fp32 carry over almost completely. The query-only attack is much weaker than white-box (because it only sees scores, with 300 queries), but it still breaks the model, so the int8 graph is not hiding useful gradients.

**Patches and backdoor.** Patch success varies a lot between seeds (CIFAR-10, 10% area: 74% on fp32 and 66% on int8, with seed intervals of about 35 points), so the small fp32-to-int8 drop is not a finding. GTSRB is much harder to patch (25% at 10% area). The backdoor results are one seed per rate:

| Poisoned share | Clean model's own rate | fp32 success | int8 success | Clean accuracy fp32 / int8 |
|---|---|---|---|---|
| CIFAR-10 1% | 2.4% | 95.8% | 49.2% | 78.7 / 78.7% |
| CIFAR-10 2% | 2.4% | 100% | 17.4% | 78.7 / 78.7% |
| CIFAR-10 5% | 2.4% | 100% | 40.2% | 77.9 / 77.9% |
| GTSRB 5% | 0.02% | 100% | 40.4% | 97.2 / 97.1% |

An unpoisoned model almost never sends triggered images to the target class, so the backdoor is real. int8 keeps it only partly, with unchanged clean accuracy, on both datasets. The pattern is not monotonic in the poison rate. A guess, untested, is that the small trigger is a fragile feature that rounding sometimes erases. With one seed per rate this is suggestive, not established.

**Defenses** (int8 model, accuracy under PGD-10; mean over 3 seeds):

| CIFAR-10 | Clean | Non-adaptive eps 2 | Adaptive eps 2 | Adaptive eps 4 |
|---|---|---|---|---|
| none | 81.7% | 5.6% | 5.6% | 0.1% |
| purify 4-bit | 80.1% | 16.3% | 11.7% | 0.6% |
| purify 5-bit + blur | 44.1% | 38.8% | 13.7% | 3.3% |
| activation clipping | 81.2% | 6.1% | 5.9% | 0.1% |
| adversarial training | 69.9% | 67.3% | 58.0% | 45.9% |

| GTSRB | Clean | Non-adaptive eps 2 | Adaptive eps 2 | Adaptive eps 4 |
|---|---|---|---|---|
| none | 97.1% | 56.0% | 56.0% | 24.3% |
| purify 4-bit | 96.0% | 63.9% | 59.7% | 28.4% |
| purify 5-bit + blur | 96.6% | 78.8% | 66.3% | 36.1% |
| activation clipping | 97.3% | 55.3% | 55.1% | 23.3% |
| adversarial training | 96.5% | 93.4% | 87.0% | 69.9% |

- **Adversarial training is the one defense that clearly works**, on both datasets and against the adaptive attacker. It costs 12 points of clean accuracy on CIFAR-10 and under 1 point on GTSRB. Its query-only accuracy is higher than its white-box accuracy, which is what an honest defense looks like and what gradient masking would not.
- **Input purification** helps against an attacker who does not know about it and mostly disappears against one who does (CIFAR-10: 16-39% non-adaptive, 12-14% adaptive, at eps 4 about 0%). On GTSRB the blur variant keeps clean accuracy and shows a small gain even adaptively, but the intervals nearly touch at 3 seeds, so it is possible, not shown.
- **Activation clipping did nothing** with one cap setting.
- Defenses added at most 6 KB to the 100 KB int8 file and under about 0.2 ms to single-image CPU latency (close to timing noise on a busy laptop).

**On-device.** The int8 and fp32 ONNX models load in ONNX Runtime Web (WASM backend) and agree with the laptop on 200 of 200 CIFAR-10 test images in a desktop browser, at about 3.5 ms per image. A real phone has not been tested yet.

## 4. Lessons learned

1. **A suspicious gap is worth a day.** The first int8 model lost 7.6 points of accuracy. Calibration changes did nothing and a torch simulation of the same rounding lost only 0.1 point, which pointed away from quantization and towards the runtime. The cause was ONNX Runtime's default uint8-activation by int8-weight CPU kernel saturating its intermediate sums on this machine; int8 activations removed the gap. Every fp32-versus-int8 comparison made before that fix was confounded.
2. **Compare against a reference implementation, and read its defaults.** ART's attacks looked far weaker than mine until I passed the true labels: without them ART appears to attack the model's own predictions (the numbers moved to near-identical once I did). Then the two agree.
3. **Check your own reporting.** The first black-box numbers were computed on 200 images while the CSV said 1,000. A column that records how many samples a number came from would have caught it on day one.
4. **A defense claim needs an attacker who knows it.** Purification looks useful until the attack is allowed to differentiate through it. Measuring both is cheap.
5. **Three seeds is thin.** Intervals of 5-35 points on several rows mean some of the interesting patterns (patch transfer, the backdoor trend) cannot be separated from noise yet.

## 5. Limitations and what is next

Limitations (full list in the README): 3 seeds and 500 images per grid; backdoor results one seed per rate; one tiny ResNet at 32x32 on a CPU; 10-step PGD without restarts or AutoAttack, so robust numbers are upper bounds; a simplified query-only attack; one quantization toolchain and a surrogate that is not QAT; cheap one-step adversarial training at one eps; one clipping cap; desktop browser only; PlantVillage and backdoor detection not done; dataset licences unverified.

Next: a real-phone run and latency; AutoAttack and stronger adaptive attacks on purification; QAT and a second int8 toolchain (TFLite on Colab); more seeds for the backdoor, with a test of why int8 weakens it; tuned clipping and full PGD adversarial training on a GPU; activation-clustering backdoor detection.

## References

Madry et al. 2018, Towards Deep Learning Models Resistant to Adversarial Attacks. Athalye, Carlini and Wagner 2018, Obfuscated Gradients Give a False Sense of Security. Brown et al. 2017, Adversarial Patch. Gu et al. 2017, BadNets. Jacob et al. 2018, Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. Wong, Rice and Kolter 2020, Fast is better than free: Revisiting adversarial training. Xu et al. 2018, Feature Squeezing. Stallkamp et al. 2012, Man vs. computer (GTSRB). Krizhevsky 2009, Learning Multiple Layers of Features from Tiny Images (CIFAR-10).
