# Instructor Guide: Privacy in Machine Learning Hands-On Lab

## Overview

This hands-on lab teaches PhD students to identify membership leakage, mount a loss-based membership inference attack, implement DP-SGD from scratch, repeat the defense with Opacus, and examine the privacy-utility trade-off.

The student notebook contains eight TODOs. Each code cell is introduced by a numbered Markdown cell with its goal, what runs, expected outcome, and TODO status.

## Materials

| File | Purpose |
|---|---|
| `privacy_lab_student.ipynb` | Student notebook with TODO scaffolding |
| `privacy_lab_solution.ipynb` | Completed reference notebook |
| `generate_checkpoints.py` | Regenerates the Phase 4 checkpoints |
| `checkpoints/` | Pre-trained fully connected models for the epsilon sweep |

## Pre-Session Setup

1. Confirm that `checkpoints/` contains `fc_eps_0.5.pt`, `fc_eps_1.0.pt`, `fc_eps_2.0.pt`, `fc_eps_5.0.pt`, `fc_eps_10.0.pt`, and `fc_eps_inf.pt`.
2. If the checkpoints are missing or incompatible, run `python generate_checkpoints.py`.
3. Run the solution notebook from a clean kernel on the target platform.
4. Confirm that every phase-validation cell reports all checks passing.
5. Test the manual DP-SGD cell in advance. Its explicit per-sample loop is intentionally educational and can be slow.
6. Clear stale notebook outputs before distribution if they do not match the current `SimpleFC` source.

## Time Allocation

| Section | Content | Time | Student action |
|---|---|---:|---|
| Setup | Install packages, import libraries, select device | 2 min | Run the setup cell |
| Phase 0 | Explore and split PneumoniaMNIST | 5 min | Run and inspect |
| Phase 1 | Train the non-private baseline | 13 min | Complete TODO 1 |
| Phase 2 | Mount the membership inference attack | 15 min | Complete TODOs 2–3 |
| Phase 3a | Implement DP-SGD from scratch | 10 min | Complete TODOs 4a–4b |
| Phase 3b | Train with Opacus and repeat the attack | 15 min | Complete TODOs 5–7 |
| Phase 4 | Evaluate the privacy-utility trade-off | 12 min | Complete TODO 8 and discuss |
| **Total** |  | **~72 min** |  |

If only 60 minutes are available, pre-run the baseline training or treat the detailed manual implementation as an instructor-led walkthrough. Keep the conceptual order: students should see clipping and noise in Phase 3a before Opacus automates them in Phase 3b.

## Phase-by-Phase Teaching Notes

### Phase 0: Warm-up and Orientation

**Key concepts**

- PneumoniaMNIST is a binary classification dataset of normalized 28×28 chest X-rays.
- The training pool is split into disjoint member, non-member, and validation groups.
- Only the member split is used to train the target model.

**Teaching emphasis**

The member/non-member distinction is the foundation of the attack. A member was used to update the model; a non-member was held out and gives the attacker a comparison group.

**Validation target**

- Member and non-member datasets are nonempty.
- A compute device is selected.
- Phase 0 reports 3/3 checks passing.

---

### Phase 1: Seeing the Leak

**Key concept**

Overfitting creates a useful attack signal. When the model fits member data more closely than unseen data, its loss and confidence distributions can differ between the two groups.

**TODO 1 answer**

```python
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

**Expected behavior**

| Metric | Validation range | Typical value |
|---|---|---|
| Member (train) accuracy | > 90% | 96–100% |
| Test accuracy | 70–98% | 80–92% |
| Member/test gap | > 2 points | 8–20 points |

The Phase 1 validation cell reports `[WARN]` instead of `[FAIL]` when a number lands outside its range, and the phase can continue regardless. The seed fixes initialization and shuffling, but results still move by a few points across hardware (CPU vs GPU, different GPU models, cuDNN kernels) and library versions, so correct student code can legitimately miss a threshold. Focus on the existence of a gap rather than one expected number.

**Common mistakes**

- Forgetting `optimizer.zero_grad()`.
- Passing logits or labels with incompatible shapes.
- Calling `optimizer.step()` before `loss.backward()`.

---

### Phase 2: Mounting a Membership Inference Attack

**Key concept**

The attack asks whether a particular example was part of the training set. This lab uses per-sample loss as the membership signal:

- Lower loss means higher model confidence.
- Higher confidence makes the example more likely to be a member.
- Negating the loss converts it into a score where larger values indicate membership.

**TODO 2 answer**

```python
per_sample_loss = F.binary_cross_entropy_with_logits(
    outputs, labels, reduction='none'
)
```

**TODO 3 answer**

```python
attack_labels = np.concatenate([
    np.ones(len(member_signals)),
    np.zeros(len(nonmember_signals)),
])
attack_scores = np.concatenate([
    -member_signals,
    -nonmember_signals,
])
mia_auc = roc_auc_score(attack_labels, attack_scores)
loss_ratio = np.mean(nonmember_signals) / (
    np.mean(member_signals) + 1e-8
)
```

**Expected behavior**

- Members have lower mean loss than non-members.
- The loss ratio exceeds 1.3 in the validation cell.
- AUC may remain close to 0.5 for this binary task because most individual samples in both groups are classified confidently.

**Discussion guidance**

- AUC measures how well the attack ranks individual members above non-members.
- The loss ratio compares aggregate mean losses and can reveal a population-level difference even when AUC is modest.
- A more sophisticated attack, such as a shadow-model attack or LiRA, may amplify a weak signal.
- Ask students how the loss-distribution separation relates to the Phase 1 generalization gap.

**Common mistakes**

- Omitting `reduction='none'`, which produces one averaged loss instead of one loss per sample.
- Forgetting to negate the loss when constructing attack scores.
- Reversing member and non-member labels.

---

### Phase 3a: Implementing DP-SGD From Scratch

**Key concept**

Students first implement the essential DP-SGD operations directly:

1. Compute a gradient for each sample.
2. Clip each sample's gradient before aggregation.
3. Sum the clipped gradients.
4. Add Gaussian noise once to the summed gradient.
5. Average the private gradient and update the model.

The Opacus accountant is used only to obtain a noise multiplier for the requested privacy parameters. The training mechanism itself is implemented manually.

**TODO 4a answer**

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(), max_grad_norm
)
```

**TODO 4b answer**

```python
noise = torch.normal(
    0.0,
    noise_multiplier * max_grad_norm,
    size=param.shape,
    device=param.device,
)
```

**Teaching emphasis**

- Clipping must happen inside the per-sample loop, before the gradient is accumulated.
- Clipping limits how much one patient can influence an update.
- Noise is added once to the summed clipped gradient, not independently to each sample. Per-sample noise would inject `batch_size` times more noise than the accountant assumed, so the reported epsilon would no longer describe the run.
- The manual Python loop favors clarity over speed.

The Understanding Check cell just before this code asks students why clipping must come first. The answer to draw out: the noise scale is calibrated to the clipping bound `C`, so without a per-sample bound there is no finite noise level that hides an individual contribution.

**Common mistakes**

- Clipping only after gradients have been aggregated.
- Adding noise before clipping.
- Forgetting to scale the noise by both the noise multiplier and clipping norm.
- Forgetting to divide the noisy summed gradient by the batch size.

**Transition to Phase 3b**

Tell students that Opacus will automate these same core operations and add privacy accounting while preserving a familiar PyTorch training loop.

---

### Phase 3b: DP-SGD with Opacus

**Key concept**

Opacus is an open-source library for training PyTorch models with differential privacy. Its `PrivacyEngine` wraps the model, optimizer, and data loader to support per-sample clipping, Gaussian noise, and privacy accounting.

#### TODO 5: Configure private training

The student cell ships each keyword argument set to `None`, so running it unfilled raises a runtime error from Opacus rather than a `SyntaxError`. Students replace the placeholders one by one.

```python
privacy_engine = PrivacyEngine()
model_dp, optimizer_dp, train_loader_dp = (
    privacy_engine.make_private_with_epsilon(
        module=model_dp,
        optimizer=optimizer_dp,
        data_loader=train_loader_dp,
        epochs=EPOCHS_DP,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )
)
```

#### TODO 6: Complete the private training step

```python
optimizer_dp.zero_grad()
outputs = model_dp(images)
loss = criterion_dp(outputs, labels)
loss.backward()
optimizer_dp.step()
```

Emphasize that the loop resembles standard PyTorch. The Opacus-wrapped optimizer performs clipping and noise injection during `optimizer_dp.step()`.

#### TODO 7: Repeat the attack on the DP model

```python
member_signals_dp = get_membership_signal(
    model_dp_eval, member_loader
)
nonmember_signals_dp = get_membership_signal(
    model_dp_eval, nonmember_loader
)
attack_labels_dp = np.concatenate([
    np.ones(len(member_signals_dp)),
    np.zeros(len(nonmember_signals_dp)),
])
attack_scores_dp = np.concatenate([
    -member_signals_dp,
    -nonmember_signals_dp,
])
mia_auc_dp = roc_auc_score(
    attack_labels_dp, attack_scores_dp
)
loss_ratio_dp = np.mean(nonmember_signals_dp) / (
    np.mean(member_signals_dp) + 1e-8
)
```

**Expected behavior**

- The DP loss ratio is lower than the baseline loss ratio.
- DP test accuracy remains above 55%.
- The DP accuracy gap is smaller than the baseline gap.
- The final epsilon remains within approximately 10% of the requested budget.

**Discussion guidance**

- Compare the manual and Opacus results only after both models have been trained.
- A loss ratio closer to 1 means members and non-members have more similar average loss.
- The empirical attack results illustrate reduced leakage; the formal privacy guarantee comes from the DP mechanism and its accountant, not from the attack metric itself.
- Epsilon tracks cumulative privacy loss over training.

**Common mistakes**

- Evaluating the wrapped object incorrectly; use `model_dp._module` when necessary.
- Calling `make_private_with_epsilon` but continuing to train with the original optimizer or loader.
- Forgetting to construct DP attack labels and scores before calculating AUC.

---

### Phase 4: The Privacy-Utility Trade-off

**Key concept**

Smaller epsilon provides stronger privacy but may reduce model utility. There is no universal operating point; the choice depends on the application, threat model, and acceptable harms.

**TODO 8 answer**

```python
test_acc_ckpt = evaluate_accuracy(model_ckpt, test_loader)
signals_m = get_membership_signal(model_ckpt, member_loader)
signals_nm = get_membership_signal(model_ckpt, nonmember_loader)
labels_ckpt = np.concatenate([
    np.ones(len(signals_m)),
    np.zeros(len(signals_nm)),
])
scores_ckpt = np.concatenate([-signals_m, -signals_nm])
auc_ckpt = roc_auc_score(labels_ckpt, scores_ckpt)
lr_ckpt = np.mean(signals_nm) / (
    np.mean(signals_m) + 1e-8
)
```

**Expected behavior**

- Six epsilon settings are evaluated.
- Accuracy generally improves as epsilon becomes less restrictive.
- The accuracy range exceeds 1 percentage point.
- The infinite-epsilon checkpoint represents non-private training.

**Final discussion guidance**

For a pneumonia diagnosis system, ask students to justify an epsilon using both privacy and clinical utility:

- What harm could result from revealing that a patient was in the training data?
- What minimum accuracy is required for the intended clinical role?
- How do AUC, loss ratio, and subgroup performance change across epsilon values?
- What attacker access and knowledge are assumed?
- Will repeated training runs or model releases compose additional privacy loss?
- What policies, laws, clinicians, and patient perspectives should inform the decision?

There is usually no single correct epsilon. A defensible choice should state its assumptions, document the trade-off, and be validated empirically.

## Troubleshooting

| Issue | Suggested response |
|---|---|
| CUDA out of memory | Reduce batch size or use the CPU fallback |
| Opacus API/version error | Install or upgrade Opacus and restart the kernel |
| Checkpoint not found | Confirm the `checkpoints/` path or regenerate checkpoints |
| `MODEL_TYPE` is undefined | Define `MODEL_TYPE = "FC"` before loading the `fc_eps_*` checkpoints |
| Checkpoint state-dict mismatch | Regenerate checkpoints using the current `SimpleFC` architecture |
| Phase 1 validation prints `[WARN]` | Expected variance across hardware; continue as long as train accuracy stays clearly above test accuracy |
| Baseline loss ratio is too small | Confirm TODOs 1–3, then consider increasing baseline epochs |
| DP loss ratio does not decrease | Confirm that the wrapped model, optimizer, and loader are used |
| Manual DP-SGD is very slow | Explain that the per-sample Python loop is educational; reduce epochs for a live demo |
| Stored outputs mention CNN | Clear and rerun the notebook; the current source uses `SimpleFC` |

## Model Architecture Notes

The current notebooks use one architecture:

- **SimpleFC:** a three-layer fully connected binary classifier with a single output logit.

Phase 4 should therefore load the `fc_eps_*` checkpoints. If the architecture changes, regenerate the checkpoints before teaching the lab.

## Connecting to the Tutorial Lecture

| Lab phase | Tutorial concept |
|---|---|
| Phase 1 | Memorization and generalization gaps |
| Phase 2 | Membership inference from confidence or loss |
| Phase 3a | Per-sample clipping and Gaussian noise |
| Phase 3b | Practical DP-SGD and privacy accounting with Opacus |
| Phase 4 | Privacy-utility trade-offs and selecting epsilon |

## Optional Extensions

- Plot the full ROC curve instead of reporting only AUC.
- Vary `MAX_GRAD_NORM` and observe the privacy-utility effect.
- Analyze leakage and accuracy separately for each class.
- Replace the manual per-sample loop with `torch.func.vmap`.
- Implement a stronger attack using shadow models or LiRA.
