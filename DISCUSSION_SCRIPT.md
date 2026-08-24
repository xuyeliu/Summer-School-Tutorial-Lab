# Discussion 上场脚本

对照学生 notebook。Phase 1 **没有** discussion。只在 Phase 2、3a、4 停下来。不要先点开 `After you discuss`。

每段大约三分钟：先要数字，再逼站边，你最后收一句。五秒没人开口，念 **Freeze**，然后闭嘴等。有人说话了，只跟一句 *Anyone on the other side?*，不要自己展开。时间到了念 **Land**，放下一 phase。

---

## Phase 1 · 不开讨论

训完 TODO 1、确认模型跑完，直接放 Phase 2。不要问 train/test、不要说 generalization gap、不要说 leak。

---

## Phase 2 · 约 3–4 分钟

他们刚看到：loss ratio 大约 6×，AUC 大约 0.50。**不要先解释 AUC。**

**Open**

> Stop here. I need two numbers from the same person: your loss ratio and your AUC.
>
> An auditor writes: “The AUC is no better than random guessing, so this model does not leak.”
>
> Looking at your own two numbers — agree or disagree?
> Who is actually at risk: a typical patient, or someone in the tail of the loss plot?

**If they talk**

- 同意审计：*Then what is the 6× ratio?*
- 只说 overfitting：*Can an outsider see that per example?*

第二问只在还有时间时用：

> Which of those two numbers would you report in a paper, and what would that choice hide?

**Freeze**

> I’m with the auditor. AUC is the standard metric. About 0.5 means we failed to find leakage, so Phase 1 was just overfitting.

**Land**

> AUC needs a single global threshold. Most losses are near zero on both sides, so they tie. The groups separate only in the tail, which is too thin to move the AUC. An AUC near 0.5 is compatible with a large ratio. The identifiable people live in the tail. Phase 4 will measure that tail directly.

不要问：loss ratio 是什么意思；不要在他们站边之前解释为什么 AUC 是 0.5。

---

## Phase 3a · 约 3 分钟

必须在放 Opacus 之前做。对照三个数：test accuracy、train accuracy、loss ratio。

**Open**

> Three numbers versus the Phase 1 baseline: test accuracy, train accuracy, and the loss ratio.
>
> Did *test* accuracy fall — yes or no?
>
> Privacy is supposed to cost utility. If test accuracy did not fall, what did you actually pay — and is that a failure of the method, or the method working?
>
> A colleague says this is just regularization. What is the one thing DP-SGD adds that ordinary regularization does not?

**If they talk**

- 觉得 DP 没效果：*What happened to the loss ratio?*
- 只说 noise：*Did you need noise to get this plot, or did clipping already do a lot?*

**Freeze**

> DP failed. We paid nothing. Test accuracy went up slightly, so the privacy must be fake.

**Land**

> The visible cost is usually train accuracy — memorization — not test accuracy. That drop can *be* the defense: it is the overfitting the attack was using. A failed attack is not a proof. The accountant is. Ordinary regularization does not give you an (ε, δ) bound. Clip-then-noise is doing the work, not the library.

---

## Phase 3b · 不开讨论 · 约 15 秒

他们跑 comparison figure 时说一句即可。

**Say**

> Same two steps you wrote, vectorized. Clip caps sensitivity; noise is what the accountant uses.

有人问 vmap / functorch：

> Opacus uses vectorized per-sample gradients. We are not going into the autograd internals.

做完早的人：拉回 3a 那题。不要新开一轮。

---

## Phase 4 · 约 4–5 分钟

先写下一个 ε。图是平的是预期，不是坏图。

**Open**

> Thirty seconds. No talking. Write one epsilon you would ship for a pneumonia model trained on real patient X-rays.
>
> Hands: who wrote 2 or below? Who wrote 10 or above?
> I want one person from each side.
>
> A hospital wants to release ε = 200 because the leakage panel is flat, so tighter privacy buys nothing. Using the number you just wrote, what do you tell them?

第二问只在还有时间时用：

> Tomorrow the setting changes: documents that appear once, and a language model. Do you still expect this curve to stay flat?

**If they talk**

- 所有人写了同一个数：你写一个不同的，让他们劝你改。
- 只谈准确率：*You only talked about accuracy. What about a patient whose membership is the sensitive fact?*
- 觉得图坏了：*It is not broken. This attack is on the floor for every finite epsilon. That is the lesson.*

**Freeze**

> Ship ε = 200. The attack panel is the measurement. It is flat, so we are done.

**Land**

> An empirical attack is a lower bound on leakage. It never certifies privacy. A flat panel means this instrument ran out of resolution, not that ε = 200 is as safe as ε = 0.5. Epsilon bounds what any adversary could learn, including attacks we did not run.
>
> Tomorrow, on text that appears once, do not expect this instrument to stay blind.

---

## 一张纸

| 停 | 先要 | 问 | 没人说话时你说 | 收束 |
|---|---|---|---|---|
| 1 | — | 不开 | — | 训完直接进 Phase 2 |
| 2 | 同一人的 ratio 和 AUC | 审计说 AUC 随机所以没漏。同意吗？ | AUC 是标准指标。随机就是没漏。 | 泄漏在尾部 |
| 3a | test / train / ratio | test acc 掉了吗？你付了什么？ | DP 失败了，test acc 没掉，隐私是假的。 | 付的是记忆。攻击失败不是证明 |
| 3b | — | 不开 | — | clip + noise，只是向量化了 |
| 4 | 每人先写下 ε | 医院要发 ε=200，因为图是平的 | 发 ε=200。图是平的，所以结束了 | 攻击是下界，从不证明隐私 |
