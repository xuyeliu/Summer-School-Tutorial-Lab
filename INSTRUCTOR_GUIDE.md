# Instructor Guide: Privacy in Machine Learning Hands-On Lab

## Overview

This hands-on lab teaches PhD students to train a baseline classifier, mount a loss-based membership inference attack, implement DP-SGD from scratch, repeat the defense with Opacus, and examine the privacy-utility trade-off.

The student notebook contains eight TODOs. Each code cell is introduced by a numbered Markdown cell with its goal, what runs, expected outcome, and TODO status.

## Materials

| File | Purpose |
|---|---|
| `privacy_lab_student.ipynb` | Student notebook with TODO scaffolding |
| `privacy_lab_solution.ipynb` | Completed reference notebook (kept local; not published in this repo) |
| `generate_checkpoints.py` | Regenerates the Phase 4 checkpoints |
| `checkpoints/` | Pre-trained fully connected models for the epsilon sweep |

## How to run the session

Do **not** walk the room through the notebook cell by cell. Your role is facilitator, not live-coding instructor.

1. **Open with a 1-2 minute overview.** Do not let anyone start the notebook until you have said what the lab is about and what the four phases are. The Lab Roadmap cell (cell 3) exists for exactly this and mirrors what you should say out loud. In the last review, several people opened the notebook without knowing what they were about to do or why each step existed.
2. **Release one phase at a time.** Give roughly 15 to 20 minutes per phase and let students read the instructions and code themselves.
3. **Stop at the phase checkpoint.** Ask who finished. Hand out hints, or the solution cell, to anyone stuck.
4. **Spend about three minutes on the checkpoint** before releasing the next phase. Run the script in each phase below. Do not invent extra questions.

Phases 1, 2, 3a, and 4 have a discussion cell. Phase 3b does not — do not open one. Takeaways sit behind an "After you discuss" disclosure. Do not read that disclosure aloud first.

## Time Allocation

| Section | Content | Time | Student action |
|---|---|---:|---|
| Opening | Roadmap and framing, delivered verbally | 2 min | Listen, do not open the notebook yet |
| Setup | Install packages, import libraries, select device | 2 min | Run the setup cell |
| Phase 0 | Explore and split BreastMNIST | 5 min | Run and inspect |
| Phase 1 | Train the non-private baseline | 15 min | Complete TODO 1, then discuss |
| Phase 2 | Mount the membership inference attack | 15 min | Complete TODOs 2-3, then discuss |
| Phase 3a | Implement DP-SGD from scratch | 20 min | Complete TODOs 4a-4b, then discuss |
| Phase 3b | Train with Opacus and repeat the attack | 15 min | Complete TODOs 5-7; no discussion |
| Comparison | Baseline vs manual DP vs Opacus, accuracy and leakage together | 3 min | Run and read the figure |
| Phase 4 | Evaluate the privacy-utility trade-off | 15 min | Complete TODO 8 and discuss |
| **Total** |  | **~92 min** |  |

Phase 3a was previously budgeted at 10 minutes, which was too short: students have to implement both per-example clipping and Gaussian noise addition. Budget 20 minutes.

If you only have 60 minutes, pre-run the baseline training and treat the manual DP-SGD implementation as an instructor-led walkthrough. Keep the conceptual order: students should see clipping and noise in Phase 3a before Opacus automates them in Phase 3b.

## How to run a checkpoint discussion

Each checkpoint is built so the room cannot stall on "any thoughts?"

- **Start with a number, not a concept.** Ask two people for their train/test, loss ratio, or chosen epsilon before you ask why.
- **Force a side.** "Agree or disagree." "Train or test in the abstract." Then take one sentence from each side. Do not ask whether Phase 1 is "already privacy." Train/test is not member/non-member.
- **You talk last.** The student cell hides the takeaway. Do not lecture it first.
- **If nobody speaks after five seconds,** make the slightly wrong claim written in that phase and let them correct you. Do not rephrase the question.
- **One primary question, then sit down.** Use the second question only if the first dies or you have leftover time.
- **Three minutes is enough.** Land one sentence and release the next phase. A partial answer is fine; a second lecture is not.

## Pre-Session Setup

1. Confirm that `checkpoints/` contains `fc_eps_0.5.pt`, `fc_eps_1.0.pt`, `fc_eps_2.0.pt`, `fc_eps_5.0.pt`, `fc_eps_10.0.pt`, `fc_eps_50.0.pt`, `fc_eps_200.0.pt`, `fc_eps_inf.pt`, and `fc_control.pt`.
2. If the checkpoints are missing or incompatible, run `python generate_checkpoints.py` (about 20 minutes on a CPU).
3. Run the solution notebook from a clean kernel on the target platform.
4. Confirm that Phases 0–3 report all checks passing. Phase 4 may `[WARN]` on the tail-attack check (TPR@1%FPR is noisy on 218/218 samples); that is expected, not a broken setup.
5. Test the manual DP-SGD cell in advance. Its explicit per-sample loop is intentionally educational and can be slow.

## Phase-by-Phase Teaching Notes

### Phase 0: Warm-up and Orientation

**Key concepts**

- BreastMNIST is a binary classification dataset of normalized 28x28 breast ultrasound images (malignant vs normal/benign).
- Only the official **train split** (546 images) is used as the membership pool; it is further split into disjoint member, non-member, and validation groups (218 / 218 / 110 with seed 500).
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

Measured on this recipe (Adam, lr 1e-3, 60 epochs, 218 members from the BreastMNIST train split):

| Metric | Typical value |
|---|---|
| Member (train) accuracy | ~100% |
| Hold-out accuracy (`test_loader` = 20% of the train pool) | ~69% |
| Train/hold-out gap | ~31 points |

The validation cell uses two soft checks only: the model fit its training data (train accuracy > 90%), and train accuracy exceeds hold-out accuracy. Focus on the existence of a gap rather than on any specific number, and say out loud that results move by a few points across hardware and library versions. Hold-out accuracy is lower than a typical MedMNIST test number because the membership pool is small (218 training examples).

**Checkpoint discussion**

The student cell asks which accuracy goes in a paper abstract, and what you would still need before calling the gap a membership leak. Do not let anyone name Phase 1 a leak. Train accuracy is measured on the member split; the other number is a **hold-out carved from the official train split**, not the official BreastMNIST test set and not the held-out non-members.

- **Open.** "Read out your two accuracies. I need two people." After two numbers: "Abstract gets train or test? And is this already a membership leak — yes or no."
- **Good answers.** Abstract: the hold-out / test number. The gap is overfitting: the model fits the training split better than unseen data from the same pool. That is not a leak. A leak needs members versus a matched non-member holdout, and a per-sample signal, not two aggregate accuracies. Some will say "the hold-out is also unseen, so it is the same idea." Let them say it, then correct the sets.
- **If the room freezes.** "The test set is unseen, so this already is membership inference. We can skip Phase 2." Then wait.
- **Land.** "This is a generalization gap. Leakage starts in Phase 2, when we compare members to the held-out non-members."
- **Do not** ask "what do the accuracies tell us," "is there a gap," or "is this already a privacy problem." They can see the gap. Privacy language waits for Phase 2.

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

- Members have much lower mean loss than non-members (often ~0.002 versus ~1.4 on this recipe).
- The loss ratio is typically hundreds of times (validation requires > 1.3x).
- AUC is typically around **0.70**, not chance. That is enough to rank many samples, but it still averages over the whole distribution.
- **Cell 12b** turns that ranking into a yes/no attack: Youden threshold \(t\) on loss, predict MEMBER if `loss < t`. Expect roughly **70% accuracy** (~305/436 correct) — clearly **YES** vs 50% random. That is the demo that AUC was not only a ranking curiosity.

**How to talk about AUC versus the loss ratio**

AUC is a global ranking metric. On this train-only split the model overfits hard, so AUC rises above 0.5. Cell 12b makes the pedagogical point concrete: one threshold \(t\) yields more correct membership calls than coin-flip. The loss ratio and the percentile / log-loss plots still matter: a large mean gap can be driven by a high-loss tail of non-members even when many samples look similar.

The cell 12 plot uses a logarithmic loss axis plus a tail survival curve for that reason.

**Checkpoint discussion**

The student cell puts an auditor on record: "AUC 0.70 is not that high, so this model barely leaks." Do not unpack the metrics before they take a side. They should also have the cell 12b correct-count in front of them.

- **Open.** "I need three numbers from the same person: loss ratio, AUC, and threshold-attack accuracy." Write them up. "Agree or disagree with the auditor."
- **Good answers.** Disagree: a moderate AUC can sit next to a huge loss ratio and ~70% threshold accuracy. The at-risk people are in the high-loss tail, not the typical patient. Reporting only AUC would hide them.
- **If the room freezes.** "I'm with the auditor. 0.70 is only a bit above chance, so Phase 1 was just overfitting." Then wait.
- **Land.** Reveal the disclosure, or say: "AUC averages over everyone. The leak is concentrated in the tail. Cell 12b already beat random with one cut on loss. Phase 4 also measures a tail rate."
- **Do not** ask "what does the loss ratio mean." Do not unpack the percentile table first.

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

**The Phase 3a figure**

The cell immediately after the from-scratch training loop draws the same two-panel log-CDF as Phase 3b, with the baseline on the left and the hand-written DP model on the right. Students should see the membership gap close *before* Opacus enters the picture: the loss ratio typically drops from hundreds of times toward ~1x, and the two curves overlap.

**Checkpoint discussion**

Run this immediately after the manual DP figure, **before** releasing Phase 3b. The student cell asks whether test accuracy fell, what they actually paid, and what DP-SGD adds that ordinary regularization does not.

- **Open.** "Three numbers versus Phase 1: test acc, train acc, loss ratio. Did test accuracy fall — yes or no?" Then: "So what did you pay?"
- **Good answers.** Test / hold-out accuracy may fall modestly on this small set, or hold roughly steady. Train accuracy drops sharply (memorization). Loss ratio collapses toward 1. The cost is memorization, which was the attack surface. Regularization can do some of this; it does not give (ε, δ).
- **If the room freezes.** "DP failed: we paid nothing, test accuracy went up slightly, so the privacy must be fake." Then wait.
- **Land.** "A failed attack is not a proof. The accountant is. We come back to that when the Phase 4 curve is flat. Clip-then-noise is doing the work, not the library."
- **Do not** ask why the CDFs overlap. They can see it.

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

- The DP loss ratio drops from hundreds of times toward ~1.0.
- DP hold-out accuracy remains above 55%; on this recipe it often lands around 70–75%.
- The DP accuracy gap (members vs non-members) is smaller than the baseline gap.
- The final epsilon remains within approximately 10% of the requested budget (ε = 2).
- **Cell 18b** repeats the Youden threshold attack on the DP model. Baseline threshold accuracy (~70%) should fall toward chance (~50%). Say out loud: DP-SGD makes this concrete attack harder, even when the attacker re-picks the best threshold for the DP losses.

There is no pass/fail AUC target. The live Phase 2 baseline is typically around 0.70; after DP it should move toward 0.5. A remaining moderate AUC still does not certify the defense.

**No discussion here**

Say one sentence while they run the comparison figure: "Same two steps you wrote, vectorized. Clip caps sensitivity; noise is what the accountant uses." If someone asks about vmap or functorch, "Opacus uses vectorized per-sample gradients; we will not go into the autograd internals." Do not open a new question. If they finished early, send them back to the Phase 3a prompt.

**The comparison figure**

Phase 3a already showed that the hand-written clip-then-noise loop closes the CDF gap. The matching Opacus figure in this phase should look the same. The cell after the manual-versus-Opacus table then draws accuracy and leakage side by side for all three models, because a text table gets skimmed past.

Be ready for mixed utility numbers. On this small BreastMNIST split, DP usually collapses the loss ratio toward 1x and cuts train accuracy a lot (often ~100% down to the 70s). Hold-out accuracy may drop a few points or more; that is a real utility cost, not a bug. Clipping and noise also act as regularizers: the training-accuracy drop *is* part of the defense, because the memorization they remove is what the attack was using. The comparison cell prints this interpretation automatically.

Phase 4 is where the epsilon sweep makes the utility cost explicit: ε = 0.5 typically sits well below the best DP setting on hold-out accuracy.

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

Phases 1–3 train **Adam** live. Phase 4 loads **SGD** checkpoints (same architecture, matched recipe). Do not expect the inf checkpoint to match the live Phase 2 AUC (~0.70) or loss ratio (hundreds of times). On the current BreastMNIST train-only checkpoints:

| Epsilon | Hold-out acc | Loss ratio | MIA AUC | Tail attack |
|---|---|---|---|---|
| 0.5 | 60.9% | 0.95x | 0.50 | 0.005 |
| 1.0 | 65.5% | 0.93x | 0.51 | 0.005 |
| 2.0 | 72.7% | 0.83x | 0.53 | 0.014 |
| 5.0 | 68.2% | 0.90x | 0.53 | 0.009 |
| 10.0 | 68.2% | 0.93x | 0.53 | 0.018 |
| 50.0 | 68.2% | 0.94x | 0.52 | 0.014 |
| 200.0 | 68.2% | 0.93x | 0.52 | 0.014 |
| inf | 60.9% | 1.23x | 0.56 | 0.018 |
| control | -- | 0.91x | 0.48 | 0.009 |

Utility still moves with epsilon: ε = 0.5 is weakest (~61%), the best DP setting is ε = 2 (~73%). The inf SGD model does **not** dominate on hold-out accuracy here. The Phase 4 validation cell may `[WARN]` that the non-private tail rate is not clearly above control (0.018 vs 0.009). That is high-variance 1% FPR on 218 samples, not a broken lab. Phases 0–3 should still be all PASS.

**How to read the leakage panel**

On this split the SGD inf point is not an order of magnitude above the DP points. Treat the control band as the no-leakage reference: DP loss ratios sit near 1x, same neighborhood as control. Apparent wiggles of 0.83x–0.95x are measurement noise on a tiny hold-out, not a ranking of epsilons. The live Adam baseline (Phase 2) is the place where membership signal is obvious; the sweep is about utility vs ε and the limits of a weak attack.

**The question you will be asked: why doesn't a smaller epsilon always look "more private" on this plot?**

Have this answer ready.

1. **Clipping already kills most memorization in the SGD recipe.** Finite-ε models stay near the control floor on loss ratio / AUC. Extra noise (smaller ε) mainly costs hold-out accuracy (ε = 0.5 vs ε = 2).
2. **The tail metric is a weak, high-variance instrument here.** TPR@1%FPR on 218+218 examples can fail to separate inf from control even when Phase 2 (Adam) showed a large leak.

Land the conclusion:

> An empirical attack gives a lower bound on leakage. It never certifies privacy.

A noisy or flat leakage panel is not evidence that ε = 200 is as safe as ε = 0.5.

**Why the tail attack rate was added, and which tail it uses**

The tail metric restricts the attacker to a 1% false-positive rate. It looks at the most confident guesses. On this lab the high-loss tail is the informative end: the positive class is the **non-member** ("allowing at most 1% of members to be flagged by mistake, what fraction of non-members does the attacker correctly identify?"). Random guessing scores 0.01.

On 218 non-members, one extra true positive moves the rate by ~0.005, so the Phase 4 `[WARN]` is unsurprising. Distinguishing the two groups in either direction is still a privacy failure.

**Final discussion guidance**

The student cell asks them to write an epsilon first, then to say which signals they trust for *measurement* versus the *guarantee* when the tail check is noisy. Do not lecture "clipping does the work" until after they have chosen a number.

- **Open.** "Thirty seconds. Write an epsilon you would ship. Do not talk yet." Then: "Hands: who wrote 2 or below? Who wrote 10 or above?" Pick one from each side.
- **Good answers.** Utility clearly changes with ε. Tail@1%FPR may not. Trust the accountant for the guarantee; use attacks as a lower bound. ε = 200 is not certified by a failed or noisy attack. On documents that appear once, expect real memorization.
- **If everyone writes the same number.** You write a different one and ask them to talk you out of it.
- **If the room freezes.** "Ship ε = 200. The tail check is inconclusive, so privacy is free." Then wait.
- **If someone thinks Phase 4 is broken because of the WARN.** "It is not. The live attack in Phase 2 was the clear leak. This sweep is a weak metric on a tiny set." Then land the lower-bound line.
- **Land.** "An attack is a lower bound. It never certifies privacy. Tomorrow, on text that appears once, do not expect this instrument to stay blind."
- **Do not** ask them to list factors. If a defense is only about accuracy, prompt once: "You only talked about accuracy. What about a patient whose membership is the sensitive fact?" Then stop.

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
| Phase 4 leakage curve looks flat or Phase 4 `[WARN]`s on tail vs control | Expected on this tiny train-only split; live Phase 2 is the clear leak. See the prepared answer above |

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
