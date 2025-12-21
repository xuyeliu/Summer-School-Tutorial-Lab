# Hands-on exploration of membership-inference attack on deep learning models  



## Overview


This repo provides a practical walkthrough of **Membership Inference Attacks (MIA)** and demonstrates how **Differentially Private (DP) training** reduces leakage.

This notebook later became the basis for an Intro to ML assignment originally developed for the University of Toronto, aimed at giving students real intuition for privacy in ML.  
Assignment page: https://modelai.gettysburg.edu/2025/privacy/

We use **MedMNIST (Medical MNIST)**, a lightweight medical-imaging benchmark, to make the privacy risk concrete. The main message is that overfitting will have privacy consequences.


---

## What is inside

By running the notebook, you will:

1. Train a baseline classifier on MedMNIST.
2. Measure generalization gap / overfitting.
3. Implement a simple but strong membership inference attack based on the confidence of the model as the test statistics. 
4. Train the same model with DP (e.g., DP-SGD).
5. Compare privacy leakage before vs. after DP training.
6. Understand the privacy–utility tradeoff in practice.

---


## Contact

For questions and feedback:
- **Mahdi Haghifam** - [haghifam.mahdi@gmail.com](mailto:haghifam.mahdi@gmail.com)
