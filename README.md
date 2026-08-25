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

QR code for the **student notebook** (opens read-only in Colab from GitHub). Participants keep their work with `File > Save a copy in Drive`.

- Colab link: [Open in Colab](https://colab.research.google.com/github/xuyeliu/Summer-School-Tutorial-Lab/blob/main/privacy_lab_student.ipynb)
- QR: [`docs/qr/qr_colab_huge.png`](docs/qr/qr_colab_huge.png)

To regenerate after changing the notebook URL, edit `COLAB_URL` in [`make_qr.py`](make_qr.py) and run it:

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
