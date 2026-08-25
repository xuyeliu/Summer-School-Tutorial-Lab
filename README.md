# Privacy Lab: Membership Inference and DP-SGD

## Run on Google Colab

**Participants: [xuyeliu.github.io/Summer-School-Tutorial-Lab](https://xuyeliu.github.io/Summer-School-Tutorial-Lab)**

| Notebook | Link |
| --- | --- |
| Student version (with TODOs) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuyeliu/Summer-School-Tutorial-Lab/blob/main/privacy_lab_student.ipynb) |

Pick a GPU runtime (`Runtime > Change runtime type > T4 GPU`) and run the setup cell at the top, which clones this repo so the Phase 4 checkpoints are available. TPU is not supported, because Opacus relies on PyTorch per-sample gradients.

These links open the notebook read-only from GitHub, so nobody can overwrite the copy in this repo. To keep your own edits, use `File > Save a copy in Drive`.

---

## Sharing the lab in a session

There are two QR targets. Both open a **read-only** copy, so nobody can overwrite the notebook in this repo. Participants keep their work with `File > Save a copy in Drive`.

| Target | What it opens | When to use it |
| --- | --- | --- |
| GitHub Pages | [xuyeliu.github.io/Summer-School-Tutorial-Lab](https://xuyeliu.github.io/Summer-School-Tutorial-Lab) | Short enough to read aloud; the page then sends people into Colab |
| Student notebook | [Open in Colab](https://colab.research.google.com/github/xuyeliu/Summer-School-Tutorial-Lab/blob/main/privacy_lab_student.ipynb) | Scan goes straight into the student notebook |

QR codes live in [`docs/qr/`](docs/qr/) and are also served over Pages:

| File | Opens | Use |
| --- | --- | --- |
| [`qr_huge_labeled.png`](docs/qr/qr_huge_labeled.png) | Pages | Opening slide, URL spelled out underneath |
| [`qr_huge.png`](docs/qr/qr_huge.png) / [`.svg`](docs/qr/qr_huge.svg) | Pages | Opening slide, bare code |
| [`qr_corner.png`](docs/qr/qr_corner.png) / [`.svg`](docs/qr/qr_corner.svg) | Pages | Later slides, top-right corner |
| [`qr_colab_huge_labeled.png`](docs/qr/qr_colab_huge_labeled.png) | Student Colab | Opening slide, caption underneath |
| [`qr_colab_huge.png`](docs/qr/qr_colab_huge.png) / [`.svg`](docs/qr/qr_colab_huge.svg) | Student Colab | Opening slide, bare code |
| [`qr_colab_corner.png`](docs/qr/qr_colab_corner.png) / [`.svg`](docs/qr/qr_colab_corner.svg) | Student Colab | Later slides, top-right corner |

Placement that actually scans from the back of a room:

- **Opening slide:** centre the code at 45% of the slide height or more.
- **Later slides, Pages code:** about 1.1 inch square in the top-right, with a 0.25 inch margin from both edges.
- **Later slides, Colab code:** about 1.3 inch square. The Colab URL is longer, so the same 1.1 inch size that works for the Pages code fails from the back of the room.
- Do not crop the white margin built into the images. That is the QR quiet zone, and most scanners refuse a code without it.
- Prefer the SVG in vector tools. Both formats have an opaque white background, because a transparent QR is unscannable on a dark slide.

To regenerate after changing a URL, edit `PAGES_URL` or `COLAB_URL` in [`make_qr.py`](make_qr.py) and run it. The script decodes its own output and fails loudly if the codes do not round-trip:

```bash
pip install segno
python make_qr.py
```

---

## Overview


This repo provides a practical walkthrough of **Membership Inference Attacks (MIA)** and demonstrates how **Differentially Private (DP) training** reduces leakage.

We use **BreastMNIST** from MedMNIST: 28×28 breast ultrasound images, binary malignant vs normal/benign. The privacy experiment uses only the official **train** split (546 images), cut into members / non-members / hold-out. The main message is that overfitting will have privacy consequences.


---

## What is inside

By running the notebook, you will:

1. Train a baseline classifier on BreastMNIST.
2. Measure generalization gap / overfitting.
3. Implement a membership inference attack that uses the model's per-sample loss as the test statistic.
4. Train the same model with DP (e.g., DP-SGD), first from scratch and then with Opacus.
5. Compare privacy leakage before vs. after DP training.
6. Understand the privacy–utility tradeoff in practice, and why an empirical attack gives a lower bound on leakage rather than a certificate of privacy.
