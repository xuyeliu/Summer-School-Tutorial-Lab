# Instructor Guide: Privacy in Machine Learning Hands-On Lab

## Overview

This 60-minute hands-on lab teaches PhD students to attack a model with membership inference, defend it with DP-SGD, and visualize the privacy-utility trade-off. Students fill in key lines of code (1-3 lines per TODO) while scaffolding and automated validation are provided.

## Materials

| File | Purpose |
|------|---------|
| `privacy_lab_student.ipynb` | Student version with blanks |
| `privacy_lab_solution.ipynb` | Complete solution for instructor |
| `generate_checkpoints.py` | Script to regenerate Phase 4 checkpoints |
| `checkpoints/` | Pre-trained models at various epsilon values |

## Pre-Session Setup

1. Ensure `checkpoints/` directory exists with all model files
2. If missing, run: `python generate_checkpoints.py` (takes ~10 min on GPU)
3. Test the solution notebook end-to-end on the target platform (Colab/local)
4. Verify all validation cells show PASS

## Time Allocation

| Phase | Content | Time | Student Action |
|-------|---------|------|----------------|
| Setup | Install + imports | 2 min | Run cell |
| Phase 0 | Data exploration | 5 min | Run cells, observe |
| Phase 1 | Train baseline | 13 min | Fill 2 lines, observe gap |
| Phase 2 | MIA attack | 15 min | Fill 3 TODOs (1 line each) |
| Phase 3 | DP-SGD defense | 15 min | Fill 2 TODOs (3 lines each) |
| Phase 4 | Trade-off curve | 12 min | Fill 2 TODOs + discussion |
| **Total** | | **~62 min** | |

## Phase-by-Phase Teaching Notes

### Phase 1: Seeing the Leak

**Key concept to emphasize**: Overfitting IS the privacy leak. The gap between train and test accuracy is exactly what an attacker exploits.

**TODO answer**:
```python
loss = criterion(outputs, labels)
loss.backward()
```

**Expected results**: Train acc ~97-99%, Test acc ~88-93%, Gap ~5-10%

**If students struggle**: This is basic PyTorch — most PhD students should know this. If stuck, point them to the hint comments.

---

### Phase 2: Membership Inference Attack

**Key concept to emphasize**: The model "knows" its training data better — it assigns lower loss (higher confidence) to members. This is a direct consequence of the generalization gap.

**TODO 1 answer**:
```python
per_sample_loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
```

**TODO 2 answer**:
```python
mia_auc = roc_auc_score(attack_labels, attack_scores)
```

**TODO 3 answer**:
```python
plt.hist(-member_signals, bins=50, alpha=0.5, label='Members', density=True)
plt.hist(-nonmember_signals, bins=50, alpha=0.5, label='Non-members', density=True)
```

**Expected results**: AUC ~0.70-0.80 for overfit model

**Common mistakes**:
- Forgetting `reduction='none'` (gives scalar instead of per-sample)
- Not negating the loss for scores (lower loss = higher membership)
- Wrong variable names in histogram

**Discussion talking points**:
- AUC = 0.5 means random (no leak), AUC = 1.0 means perfect attack
- The histogram separation IS the generalization gap, just measured differently
- Real attacks can be stronger (shadow models, LiRA)

---

### Phase 3: DP-SGD Defense

**Key concept to emphasize**: DP-SGD has two components that map to the tutorial theory:
1. Gradient clipping → bounds sensitivity (Δf)
2. Gaussian noise → calibrated to sensitivity (like Gaussian mechanism)

**TODO 1 answer**:
```python
model_dp, optimizer_dp, train_loader_dp = privacy_engine.make_private_with_epsilon(
    module=model_dp,
    optimizer=optimizer_dp,
    data_loader=train_loader_dp,
    epochs=EPOCHS_DP,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)
```

**TODO 2 answer**:
```python
member_signals_dp = get_membership_signal(model_dp_eval, member_loader)
nonmember_signals_dp = get_membership_signal(model_dp_eval, nonmember_loader)
mia_auc_dp = roc_auc_score(attack_labels_dp, attack_scores_dp)
```

**Expected results**: DP AUC ~0.50-0.55, test acc ~70-85%

**Common mistakes**:
- Using `model_dp` directly instead of `model_dp._module` for evaluation
- Forgetting to construct `attack_scores_dp` before computing AUC

**Discussion talking points**:
- Clipping ensures no single example dominates the gradient
- Noise makes it impossible to tell if a specific example was present
- epsilon tracks cumulative privacy loss (composition theorem)

---

### Phase 4: Privacy-Utility Trade-off

**Key concept to emphasize**: There's no free lunch. The curve makes this visceral — you see utility dropping as you demand more privacy. The "right" operating point depends on the application.

**TODO 1 answer**:
```python
test_acc = evaluate_accuracy(model_ckpt, test_loader)
auc = roc_auc_score(labels_ckpt, scores_ckpt)
```

**TODO 2 answer**:
```python
ax.scatter(r['mia_auc'], r['test_acc'], s=120, zorder=5)
ax.annotate(f"  eps={eps_label}", (r['mia_auc'], r['test_acc']), fontsize=11)
```

**Expected curve shape**: Roughly from bottom-left (low AUC, low acc) to top-right (high AUC, high acc)

**Discussion talking points**:
- Medical: might choose eps=1-2 (regulatory requirement, high stakes)
- Recommendation system: might choose eps=5-10 (lower stakes)
- GDPR doesn't specify an epsilon value — it's a policy decision
- Larger datasets make privacy "cheaper" (O(1/εn) utility cost)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Switch to `MODEL_TYPE = "FC"` or reduce batch size |
| Opacus version error | Ensure opacus >= 1.4: `pip install opacus --upgrade` |
| Checkpoint not found | Run `python generate_checkpoints.py` first |
| AUC near 0.5 for baseline | Model didn't overfit enough — increase epochs |
| DP model AUC > 0.6 | Try smaller epsilon or more epochs |
| Very slow training | Use GPU (Colab) or reduce epochs |

## Model Architecture Notes

Both architectures are designed to be simple and fast:
- **SimpleFC**: 3-layer fully connected. Trains in ~1.5 min. Less overfitting potential.
- **SimpleCNN**: 2 conv layers + FC. Trains in ~2.5 min. Better accuracy, clearer privacy signal.

Students can choose either. CNN is recommended for clearer demonstration of the privacy leak.

## Connecting to the Tutorial Lecture

| Lab Phase | Tutorial Concept |
|-----------|-----------------|
| Phase 1 (overfitting) | Motivation: models memorize training data |
| Phase 2 (MIA) | MIA slide: attackers exploit confidence gap |
| Phase 3 (DP-SGD) | DP-SGD slide: clip + noise at each step |
| Phase 4 (trade-off) | Privacy-accuracy trade-off: O(1/εn) |

## Optional Extensions (if time allows)

- **ROC curve**: Plot the full ROC curve instead of just AUC
- **Vary clipping norm**: Change MAX_GRAD_NORM and observe effect
- **Per-class analysis**: Check if some classes leak more than others
- **Shadow model attack**: Implement a stronger attack (preview for Challenge session)
