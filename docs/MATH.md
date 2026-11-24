# The math behind voxguard

One framework covers every plugin: treat the phone line plus recognizer as a noisy channel, add a prior from context, and act by minimizing expected cost.

## 1. Recognition as Bayesian inference

```
v* = argmax_v  P(audio | v) · P(v | context)
```

`P(audio | v)` is the acoustic evidence. `P(v | context)` is the prior. Each failure mode is a problem with one of the two terms.

| Failure | Term at fault |
|---|---|
| Rare names | Likelihood cannot separate candidates on 8 kHz audio; generic prior favors common names |
| Confusable digits | Likelihood is ambiguous for some pairs; the prior over digit strings is uniform, so it cannot help |
| STS padding a 9-digit number to 10 | Learned prior ("phone numbers have 10 digits") overrides the evidence |
| Accents | Likelihood is mis-estimated for speech unlike the training data |

## 2. Combining two recognizers

The agent and the shadow recognizer each produce a value with error probabilities `p_m` and `p_s`. Treating them as conditionally independent:

```
agree:     P(wrong) ≈ p_m · p_s
disagree:  P(model value wrong) = p_m (1 - p_s) / (p_m (1 - p_s) + p_s (1 - p_m))
```

Independence is optimistic: recognizers err together on the same bad audio. The benchmark measures the real joint behavior, and the mock backends simulate 40% shared errors for that reason.

## 3. Digits: noisy channel decoding

Estimate a confusion matrix `C[d, h] = P(heard h | spoken d)` from aligned pairs (`fit_confusion`, Laplace smoothed). With a heard digit `h` and recognizer confidence `c`:

```
P(d | evidence) ∝ c · 1[d = h] + (1 - c) · P(d | h),     P(d | h) ∝ C[d, h]
score(string) = Σ_i log P(d_i | evidence_i)
```

Candidates are restricted to a valid set (format check, Luhn, or known records). Against a list of known order IDs, this is an exact maximum a posteriori decision.

Unheard digits: if the caller said 9 digits, the posterior for a 10th is uniform over 10 values. Any filled-in digit is a guess that is wrong 90% of the time, so `NumberGuard` treats padding as an error to confirm, never something to repair.

## 4. Names: context prior

```
P(c | e_1..e_k) ∝ Π_j L(e_j | c) · P(c)
L(e | c) = exp(beta · (sim(e, c) - 1))
P(c) = alpha / |roster|                      for c in the roster
P(c) = (1 - alpha) / open_vocab_size         for a specific name outside it
```

`sim` mixes phonetic-key similarity with spelling similarity. Roster entries can carry respellings (`"Siobhan|Shivawn"`) because spelling-based phonetics cannot read every orthography.

`alpha` is the bias strength. Too high hallucinates roster names that were never said; too low misses real ones. Tune it on held-out data and report both the miss rate and the false-alarm rate.

## 5. When to ask: expected cost

```
confirm  iff  P(error) · cost_error  >  cost_confirm
```

A wrong phone number loses the customer (high cost); a slightly wrong topping does not. One rule gives different behavior per field.

The rule needs calibrated probabilities. `TemperatureScaler` fits `T` in `sigmoid(logit / T)` by minimizing negative log-likelihood on held-out data; check with expected calibration error (ECE).

## 6. Turn-taking: optimal stopping

Probability the caller has finished after `t` ms of silence, in state `s` (complete, filler, mid-number, spelling):

```
P(done | t, s) = 1 - exp(-t / tau_s)
```

Commit when waiting costs more than cutting in:

```
P(done) · C_deadair > (1 - P(done)) · C_cutin,s
⇔  P(done) > p*_s = C_cutin,s / (C_cutin,s + C_deadair)
⇔  t > t*_s = -tau_s · ln(1 - p*_s)
```

Cutting in during a number corrupts an entity, so `C_cutin` is larger in that state and the optimal wait is longer. After a complete sentence the wait is shorter than a fixed threshold, so average latency drops too.

The state classifier is rule-based for transparency. A semantic end-of-turn model can replace it without changing the decision rule.

## 7. The limit

Information theory bounds what any decoder can do. Phone audio removes energy above about 3.4 kHz, where much of the difference between s, f, and th lives. When `H(value | audio)` is high, Fano's inequality gives a lower bound on error probability that no algorithm beats without more information. The only sources of more information are a better prior (records) or the caller. The decision rule picks between them.
