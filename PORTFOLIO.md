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

## Optional tailored second sentences (drafts from a web search on 2026-11-19; check each page is current and read a recent paper before sending)

Each is a draft you can append to the sentence above. I only used what the search results said about the group, not any specific paper, so none of them claims to know a person's results.

- **RIT:** [Cory Merkel](https://www.rit.edu/ai/news/engineering-faculty-member-wins-air-force-research-grant-work-improvements-neuromorphic) is reported to study how hardware characteristics affect AI robustness to adversarial attacks, and [Matthew Wright](https://sites.google.com/site/matthewkwright) has worked on adversarial machine learning and AI security: "My results suggest the hardware-side change (int8 rounding) did not by itself change attack success, so I would value your view on which hardware properties are worth testing next."
- **Guelph:** [Wenjing Zhang](https://www.uoguelph.ca/ceps/people/wenjing-zhang) leads the AI Security Lab, which is reported to cover poisoning and evasion attacks: "I measured both a poisoning (backdoor) and an evasion (PGD, patch) attack across fp32 and int8, and I would like to learn how you evaluate defenses against adaptive attackers."
- **UCC Cork:** the [CyberSecurity Research group](https://security.ucc.ie/) lists edge-security work (for example the BRAINE project): "My phone test shows the quantized models run correctly in a browser at about 8 ms per image, so deployment is cheap; I would like to ask how to protect such models once they sit on a device."
- **Aalborg:** the search did not find a clearly matching group page, so I drafted nothing. Check Aalborg's own research pages for secure embedded AI, or the lines above as a template.

