# Instructor Guide: Privacy in Machine Learning Hands-On Lab

## Overview

This hands-on lab teaches PhD students to train a baseline classifier, mount a loss-based membership inference attack, implement DP-SGD from scratch, repeat the defense with Opacus, and examine the privacy-utility trade-off.

The student notebook contains eight TODOs. Each code cell is introduced by a numbered Markdown cell with its goal, what runs, expected outcome, and TODO status.

## Materials

| File | Purpose |
|---|---|
| `privacy_lab_student.ipynb` | Student notebook with TODO scaffolding |
| `privacy_lab_solution.ipynb` | Completed reference notebook |
| `generate_checkpoints.py` | Regenerates the Phase 4 checkpoints |
| `checkpoints/` | Pre-trained fully connected models for the epsilon sweep |

## How to run the session

Do **not** walk the room through the notebook cell by cell. Your role is facilitator, not live-coding instructor.

1. **Open with a 1-2 minute overview.** Do not let anyone start the notebook until you have said what the lab is about and what the four phases are. The Lab Roadmap cell (cell 3) exists for exactly this and mirrors what you should say out loud. In the last review, several people opened the notebook without knowing what they were about to do or why each step existed.
2. **Release one phase at a time.** Give roughly 15 to 20 minutes per phase and let students read the instructions and code themselves.
3. **Stop at the phase checkpoint.** Ask who finished. Hand out hints, or the solution cell, to anyone stuck.
4. **Spend a few minutes on the takeaway** using the checkpoint discussion cells before releasing the next phase.

Phases 1, 2 and 4 end with an explicit "Checkpoint — Discussion" markdown cell holding the questions to run.

## Time Allocation

| Section | Content | Time | Student action |
|---|---|---:|---|
| Opening | Roadmap and framing, delivered verbally | 2 min | Listen, do not open the notebook yet |
| Setup | Install packages, import libraries, select device | 2 min | Run the setup cell |
| Phase 0 | Explore and split PneumoniaMNIST | 5 min | Run and inspect |
| Phase 1 | Train the non-private baseline | 15 min | Complete TODO 1, then discuss |
| Phase 2 | Mount the membership inference attack | 15 min | Complete TODOs 2-3, then discuss |
| Phase 3a | Implement DP-SGD from scratch | 20 min | Complete TODOs 4a-4b |
| Phase 3b | Train with Opacus and repeat the attack | 15 min | Complete TODOs 5-7 |
| Comparison | Baseline vs manual DP vs Opacus, accuracy and leakage together | 3 min | Run and read the figure |
| Phase 4 | Evaluate the privacy-utility trade-off | 15 min | Complete TODO 8 and discuss |
| **Total** |  | **~92 min** |  |

Phase 3a was previously budgeted at 10 minutes, which was too short: students have to implement both per-example clipping and Gaussian noise addition. Budget 20 minutes.

If you only have 60 minutes, pre-run the baseline training and treat the manual DP-SGD implementation as an instructor-led walkthrough. Keep the conceptual order: students should see clipping and noise in Phase 3a before Opacus automates them in Phase 3b.

## Pre-Session Setup

1. Confirm that `checkpoints/` contains `fc_eps_0.5.pt`, `fc_eps_1.0.pt`, `fc_eps_2.0.pt`, `fc_eps_5.0.pt`, `fc_eps_10.0.pt`, `fc_eps_50.0.pt`, `fc_eps_200.0.pt`, `fc_eps_inf.pt`, and `fc_control.pt`.
2. If the checkpoints are missing or incompatible, run `python generate_checkpoints.py` (about 20 minutes on a CPU).
3. Run the solution notebook from a clean kernel on the target platform.
4. Confirm that every phase-validation cell reports all checks passing.
5. Test the manual DP-SGD cell in advance. Its explicit per-sample loop is intentionally educational and can be slow.

## Phase-by-Phase Teaching Notes

### Phase 0: Warm-up and Orientation

**Key concepts**

- PneumoniaMNIST is a binary classification dataset of normalized 28x28 chest X-rays.
- The training pool is split into disjoint member, non-member, and validation groups.
- Only the member split is used to train the target model.

**Teaching emphasis**

The member/non-member distinction is the single biggest comprehension barrier in this lab, and explaining it once in cell 4 is not enough. Say it at the split, say it again at the start of Phase 2, and point back to cell 4 when you do. A member was used to update the model; a non-member was held out and gives the attacker a comparison group. Members and non-members come from the same distribution, so the *only* difference between them is whether the optimizer ever saw them.

**Validation target**

Member and non-member datasets are nonempty, a compute device is selected, and Phase 0 reports 3/3 checks passing.

---

### Phase 1: Train the Non-Private Baseline

**Key concept**

Phase 1 only trains the baseline classifier and looks at how it performs on data it has seen versus data it has not. Resist the temptation to talk about membership leakage here. In the last review, framing Phase 1 around leakage made people repeatedly ask what this step had to do with members and non-members, when the honest answer is "nothing yet, we are training the model the attack will target."

Say instead: Phase 1 trains the non-private baseline model and examines its train/test behavior. We will use this trained model in the next phase to study membership inference.

**TODO 1 answer**

```python
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

**Expected behavior**

Measured on this recipe (Adam, lr 1e-3, 20 epochs, member split of 1883 samples):

| Metric | Typical value |
|---|---|
| Member (train) accuracy | 99% |
| Test accuracy | 82% |
| Member/test gap | 17 points |

The validation cell no longer checks a numeric range for test accuracy. It previously advertised 85-93% in the phase header and 70-98% in the validation cell, while the recipe actually produces about 81%, which made students think they had broken something. Only two soft checks remain: the model fit its training data, and train accuracy exceeds test accuracy. Focus on the existence of a gap rather than on any specific number, and say out loud that results move by a few points across hardware and library versions.

**Checkpoint discussion**

- What do the train and test accuracies tell us?
- Is there a generalization gap?
- Why might a model with different behavior on seen versus unseen samples later be vulnerable to membership inference?

**Common mistakes**

- Forgetting `optimizer.zero_grad()`.
- Passing logits or labels with incompatible shapes.
- Calling `optimizer.step()` before `loss.backward()`.

---

### Phase 2: Mounting a Membership Inference Attack

**Key concept**

Lead with the mechanism, not the metric:

> The model was optimized to fit the members, so it is usually more confident on them. Member loss is therefore lower than non-member loss, and that difference is the membership inference signal.

Quantify it with the loss ratio, `mean(non-member loss) / mean(member loss)`. Do not lead with AUC.

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
loss_ratio = np.mean(nonmember_signals) / (
    np.mean(member_signals) + 1e-8
)
mia_auc = roc_auc_score(attack_labels, attack_scores)
```

**Expected behavior**

- Members have lower mean loss than non-members: about `0.025` versus `0.158`.
- The loss ratio is roughly 6x, and the validation cell requires it to exceed 1.3.
- The AUC stays near 0.5, measured at 0.497.

**Why the AUC is useless here, and what to say about it**

This is not a bug and it will not improve. The percentile table printed by cell 12 shows why. On the baseline model:

| Percentile | Member loss | Non-member loss | Ratio |
|---|---|---|---|
| 50% | 3.6e-05 | 3.0e-05 | 0.8x |
| 75% | 9.4e-04 | 1.1e-03 | 1.1x |
| 90% | 1.2e-02 | 2.9e-02 | 2.4x |
| 95% | 4.9e-02 | 3.4e-01 | 6.9x |
| 99% | 6.2e-01 | 4.5e+00 | 7.2x |

The two distributions are indistinguishable up to the 75th percentile, where both are essentially zero, and separate only in the top 10%. Because a single global threshold cannot separate the tied bulk, AUC stays at chance no matter how much the model memorizes; in a stress test it still read 0.498 for a model whose mean loss ratio was 600x.

That is why the cell 12 plot now uses a logarithmic loss axis plus a tail survival curve, and prints the table above. On the old linear axis the entire signal was squashed against zero, and students correctly complained that they could not tell whether the difference was meaningful.

**Checkpoint discussion**

- Why do member samples tend to have lower loss than non-member samples?
- What does the loss ratio represent, and what value means no leakage?
- If the distributions became more similar, what would that imply about membership leakage?
- How does this relate to the Phase 1 train/test gap?

**Common mistakes**

- Omitting `reduction='none'`, which produces one averaged loss instead of one loss per sample.
- Forgetting to negate the loss when constructing attack scores.
- Reversing member and non-member labels.

---

### Phase 3a: Implementing DP-SGD From Scratch

**Key concept**

Use this one sentence, which is more precise than a list of steps:

> DP-SGD first clips each example's gradient contribution, aggregates the clipped gradients, and then adds Gaussian noise to the aggregated gradient before the parameter update.

Two follow-up questions come up every time, so answer them pre-emptively:

- **Is clipping per example or per batch?** Per example. Each sample's own gradient is rescaled to norm at most `C` before it is added to anything.
- **Where does the noise go?** Once, on the aggregated gradient, before the optimizer step. Never per example.

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
- Per-sample noise would inject `batch_size` times more noise than the accountant assumed, so the reported epsilon would no longer describe the run.
- The manual Python loop favors clarity over speed.

The Understanding Check cell asks why clipping must come first. The answer to draw out: the noise scale is calibrated to the clipping bound `C`, so without a per-sample bound there is no finite noise level that hides an individual contribution.

**Common mistakes**

- Clipping only after gradients have been aggregated.
- Adding noise before clipping.
- Forgetting to scale the noise by both the noise multiplier and clipping norm.
- Forgetting to divide the noisy summed gradient by the batch size.

---

### Phase 3b: DP-SGD with Opacus

#### TODO 5: Configure private training

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

The student cell ships each keyword argument set to `None`, so running it unfilled raises a runtime error from Opacus rather than a `SyntaxError`.

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
member_signals_dp = get_membership_signal(model_dp_eval, member_loader)
nonmember_signals_dp = get_membership_signal(model_dp_eval, nonmember_loader)
attack_labels_dp = np.concatenate([
    np.ones(len(member_signals_dp)),
    np.zeros(len(nonmember_signals_dp)),
])
attack_scores_dp = np.concatenate([
    -member_signals_dp,
    -nonmember_signals_dp,
])
mia_auc_dp = roc_auc_score(attack_labels_dp, attack_scores_dp)
loss_ratio_dp = np.mean(nonmember_signals_dp) / (
    np.mean(member_signals_dp) + 1e-8
)
```

**Expected behavior**

- The DP loss ratio drops from about 6.2x to about 0.86x, so it lands near 1.0.
- DP test accuracy remains above 55%; it typically lands around 84%.
- The DP accuracy gap is smaller than the baseline gap.
- The final epsilon remains within approximately 10% of the requested budget.

There is no longer a target AUC. The old `MIA AUC < 0.58` target was vacuous, since the baseline already measures about 0.50.

**Discussion guidance**

- A loss ratio closer to 1 means members and non-members have more similar average loss.
- The empirical attack results illustrate reduced leakage. The formal guarantee comes from the DP mechanism and its accountant, not from the attack metric. A failed attack does not prove privacy.
- Epsilon tracks cumulative privacy loss over training.

**The comparison figure**

The cell after the manual-versus-Opacus table draws accuracy and leakage side by side for all three models, because a text table gets skimmed past.

Be ready for a result that looks wrong. On the current recipe, DP costs essentially no *test* accuracy (about 82% to 84%) while the loss ratio collapses from 6.2x to 0.86x. The visible cost is in *train* accuracy, 99% down to about 94%. Say this out loud rather than glossing over it: clipping and noise also act as regularizers, so a model that memorizes less can generalize just as well, and the training-accuracy drop *is* the defense, because the memorization it removed is exactly what the attack was exploiting. The cell prints this interpretation automatically and adapts if test accuracy does fall on your hardware.

Phase 4 is where a real utility cost appears, since epsilon = 0.5 gives up more than 10 points of test accuracy.

**Common mistakes**

- Evaluating the wrapped object incorrectly; use `model_dp._module` when necessary.
- Calling `make_private_with_epsilon` but continuing to train with the original optimizer or loader.
- Forgetting to construct DP attack labels and scores before calculating AUC.

---

### Phase 4: The Privacy-Utility Trade-off

**How the checkpoints are built, and why it matters**

Every checkpoint uses one matched recipe: SGD, lr 0.1, batch 64, 60 epochs, clipping norm C = 1.0. Only epsilon varies. Two checkpoints anchor the sweep:

- `fc_eps_inf.pt` runs the same recipe with clipping and noise switched off. It is the non-private endpoint.
- `fc_control.pt` is trained on the validation split, so it has seen neither members nor non-members and cannot leak about either. Its measured membership signal is pure measurement noise, which gives the class an empirical "no leakage" reference. This matters because the loss ratio of a non-leaking model is only approximately 1.0, not exactly 1.0.

An earlier version of this lab produced a genuinely confusing plot, and the cause was the recipe, not the plotting. The DP checkpoints trained for 20 epochs, a budget at which this model cannot memorize the member split **even with zero noise** (measured loss ratio 1.36 at 20 epochs, 5.5 at 50, 116 at 100). There was therefore no leakage for epsilon to modulate. Meanwhile the non-private checkpoint used Adam for 25 epochs, so the apparent cliff between epsilon = 10 and epsilon = inf was an optimizer artifact, and its overfitting was why its test accuracy fell *below* epsilon = 5 and epsilon = 10. Do not shorten the epoch budget when regenerating checkpoints.

**TODO 8 answer**

```python
test_acc = evaluate_accuracy(model_ckpt, test_loader)
signals_m = get_membership_signal(model_ckpt, member_loader)
signals_nm = get_membership_signal(model_ckpt, nonmember_loader)
labels = np.concatenate([np.ones(len(signals_m)), np.zeros(len(signals_nm))])
scores = np.concatenate([-signals_m, -signals_nm])
```

**Expected behavior**

Measured on the current checkpoints:

| Epsilon | Test accuracy | Loss ratio | Tail attack rate | Verdict |
|---|---|---|---|---|
| 0.5 | 71.6% | 0.95x | 0.007 | at the no-leakage floor |
| 1.0 | 74.4% | 0.92x | 0.010 | at the no-leakage floor |
| 2.0 | 78.5% | 0.99x | 0.011 | at the no-leakage floor |
| 5.0 | 82.1% | 1.04x | 0.011 | at the no-leakage floor |
| 10.0 | 83.3% | 1.04x | 0.009 | at the no-leakage floor |
| 50.0 | 82.4% | 1.04x | 0.010 | at the no-leakage floor |
| 200.0 | 82.4% | 1.03x | 0.010 | at the no-leakage floor |
| inf | 81.9% | 12.18x | 0.051 | **leaks** |
| control | -- | 0.97x | 0.011 | no leakage by construction |

Random guessing gives a tail attack rate of 0.010, so the non-private model at 0.051 is the only point that registers as a real attack.

Note that epsilon = inf has slightly *lower* test accuracy than epsilon = 10. That is not a bug: without clipping the model overfits to 99.6% training accuracy and generalizes marginally worse. It is worth pointing out, because it is the same overfitting that makes it the only model that leaks.

**How to read the leakage panel**

The leakage panel has a broken y-axis. The non-private point at 12.18x is an order of magnitude above everything else, so on a single axis the eight DP points collapse into one flat line pinned to the bottom and students cannot see them at all. The lower part of the panel therefore zooms into the no-leakage band, and the upper part carries the non-private point.

The zoom makes the DP points legible, and it will make them look like a rising trend from 0.92x to 1.04x. Head that reading off using the shaded band: the control model, which provably cannot leak, measures 0.97x, and the band spanning 0.73x to 1.29x is the range that same measurement covers. The entire DP spread sits inside it, so the apparent trend is measurement noise, not a privacy difference between epsilon values. This is the point of showing the control model at all, and the tail attack rate column confirms it independently, since every DP checkpoint scores at the random-guessing baseline of 0.010.

**The question you will be asked: why doesn't a smaller epsilon improve the membership metric?**

Have this answer ready, because the plot invites the question and the honest answer is the best lesson in the lab.

The empirical attack is saturated at its floor for every finite epsilon, for two reasons:

1. **Clipping alone does most of the work.** Bounding every per-sample gradient to norm 1.0 prevents any individual X-ray from moving the model enough to be memorized. The non-private model reaches 99.6% accuracy on its training data; every clipped model stops near 95%, which is about what it scores on data it has never seen. With no memorization there is no membership signal, whatever the noise level. At epsilon = 200 the noise multiplier is only 0.33 and leakage is still at the floor, which isolates clipping as the cause.
2. **The attack is weak and the task is easy.** A loss-threshold attack on a confidently-classified binary task is close to the weakest attack in the literature.

Land the conclusion:

> An empirical attack gives a lower bound on leakage. It never certifies privacy.

Epsilon does not promise that *this* attack fails; it bounds what *any* adversary could learn, including attacks nobody has run. A flat leakage curve is evidence that the measuring instrument ran out of resolution, not evidence that epsilon = 200 is as safe as epsilon = 0.5.

**Why the tail attack rate was added, and which tail it uses**

AUC cannot see this leak at all: it reads 0.4976 for the non-private model, which is chance. The tail metric restricts the attacker to a 1% false-positive rate, so it looks only at the most confident guesses, which is where the at-risk individuals are. It registers the non-private model at 0.051 against a random baseline of 0.010.

One subtlety worth knowing before a student asks. The metric thresholds the **high**-loss tail, so the positive class is the non-member: "allowing at most 1% of members to be flagged by mistake, what fraction of non-members does the attacker correctly identify?" The usual membership inference convention is the opposite direction, members as positives at low FPR, but on this task that direction is dead: the low-loss end is saturated, both groups are equally confident there, and the metric returns exactly 0.010 for every checkpoint including the leaking one. That was in fact a bug in the first draft of this revision.

Distinguishing the two groups in either direction is a privacy failure, so the metric is legitimate. Confidently ruling a patient *out* of the training set leaks membership information just as much as ruling them in. If a student raises it, that is a good sign, and the honest framing is that the direction of the attack is dictated by where the signal lives, which here is the tail of samples the model gets wrong.

**Final discussion guidance**

For a pneumonia diagnosis system, ask students to justify an epsilon. Point out that they cannot do it from the leakage panel, because it is flat, which forces them to reason about:

- What harm could result from revealing that a patient was in the training data?
- What minimum accuracy is required for the intended clinical role?
- What attacker access and knowledge are assumed?
- Will repeated training runs or model releases compose additional privacy loss?
- Do some patient subgroups lose more accuracy than others?
- What policies, laws, clinicians, and patient perspectives should inform the decision?

There is usually no single correct epsilon. A defensible choice states its assumptions, documents the trade-off, and is validated empirically.

Close by asking what would change on a harder task, such as a larger model, a rarer condition, or a language model trained on documents that appear only once. That is the bridge to the challenge.

## Connecting to the Challenge

Say this explicitly at the end, because the lab and the challenge use opposite lenses and the switch is otherwise jarring:

> In this lab, we first learn what membership signals look like and how DP can suppress them. In the challenge, you will switch to the attacker's perspective and try to exploit membership signals in an LLM.

The challenge supplies the model, so no retraining is required; the task is to identify members. Akbar's lecture introduces other attack families, so covering only one simple attack hands-on is fine. The transferable intuition: find a quantity that behaves differently on data the model has seen, and look at the tail of its distribution rather than the average.

## Troubleshooting

| Issue | Suggested response |
|---|---|
| CUDA out of memory | Reduce batch size or use the CPU fallback |
| Opacus API/version error | Install or upgrade Opacus and restart the kernel |
| Checkpoint not found | Confirm the `checkpoints/` path or regenerate checkpoints |
| `MODEL_TYPE` is undefined | Define `MODEL_TYPE = "FC"` before loading the `fc_*` checkpoints |
| Checkpoint state-dict mismatch | Regenerate checkpoints using the current `SimpleFC` architecture |
| Phase 1 numbers differ from a neighbour's | Expected variance across hardware; continue as long as train accuracy stays clearly above test accuracy |
| Baseline loss ratio is too small | Confirm TODOs 1-3, then consider increasing baseline epochs |
| DP loss ratio does not decrease | Confirm that the wrapped model, optimizer, and loader are used |
| Manual DP-SGD is very slow | Explain that the per-sample Python loop is educational; reduce epochs for a live demo |
| Phase 4 leakage curve looks flat | This is the expected and intended result; see the prepared answer above |

## Model Architecture Notes

The notebooks use one architecture: **SimpleFC**, a three-layer fully connected binary classifier with a single output logit. Phase 4 loads the `fc_*` checkpoints. If the architecture changes, regenerate the checkpoints before teaching the lab.

## Connecting to the Tutorial Lecture

| Lab phase | Tutorial concept |
|---|---|
| Phase 1 | Memorization and generalization gaps |
| Phase 2 | Membership inference from confidence or loss |
| Phase 3a | Per-sample clipping and Gaussian noise |
| Phase 3b | Practical DP-SGD and privacy accounting with Opacus |
| Phase 4 | Privacy-utility trade-offs, selecting epsilon, and the limits of empirical attack evaluation |

## Optional Extensions

- Plot the full ROC curve instead of reporting only AUC and the tail attack rate.
- Vary `MAX_GRAD_NORM` and observe how much of the defense comes from clipping alone.
- Analyze leakage and accuracy separately for each class.
- Replace the manual per-sample loop with `torch.func.vmap`.
- Implement a stronger attack using shadow models or LiRA and check whether it beats the clipping floor.
