# Portfolio card: Secure TinyML

**Title:** Secure TinyML: Attacking and Defending Quantized Models

**Card text (120 words):**

Secure TinyML is a study of what happens to adversarial attacks when a small image classifier is squeezed to int8 for a phone or microcontroller. I trained a tiny ResNet on CIFAR-10 and GTSRB, then attacked the full-precision and int8 versions with hand-written FGSM and PGD, adversarial patches and a backdoor, and tested three defenses against attackers who know about them. Over three seeds, int8 matched fp32 on accuracy and robustness, attacks transferred between them almost completely, and only adversarial training clearly helped. A planted backdoor weakened under int8. A local Gradio demo and a live in-browser ONNX Runtime Web page let anyone try it, using small exported ONNX model files. Results are small-scale and the limitations are written down.

**Links:** repo https://github.com/Nathaniel47/secure-tinyml · live browser demo https://nathaniel47.github.io/secure-tinyml/demo.html · phone test page https://nathaniel47.github.io/secure-tinyml/ · write-up `docs/technical-writeup.md`

**Tags:** adversarial machine learning, quantization, edge AI, backdoors, ONNX, PyTorch

---

# One sentence for emailing professors

> I built an open study showing that, on small CIFAR-10 and GTSRB classifiers, int8 post-training quantization neither removed nor added adversarial vulnerability (attacks crafted on the full-precision model fooled the int8 model 98-100% of the time) yet weakened a planted backdoor, which bears on how compression interacts with protecting deployed models on embedded devices.

Notes for using it:
- It states only what this project measured (one tiny ResNet, 3 seeds; the backdoor result is one seed per rate). Keep those qualifiers if you shorten it.
- It makes no claim about any professor's own work. Before sending, read one recent paper from each person and add a half-sentence of your own about how it relates (for example secure embedded AI at Aalborg, AI security at RIT, protection of AI algorithms at UCC Cork); I did not look up their publications, so I have not named any.
