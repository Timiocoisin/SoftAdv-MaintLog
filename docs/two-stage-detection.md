# Two-stage detection: what a flag is worth

A note for the manuscript, candidate 1. Every symbol is defined here; nothing is
imported from the paper's prose.

## 1. The object the paper was missing

The manuscript argues that an adjacent slip cannot be caught because "an inspector
comparing the sheet with the scene has little to separate stage g from stage g+3". That
sentence is doing work that no symbol in the model does. Introduce the symbol.

For class `c` and threshold `theta`:

- `Det_c(theta)`  probability the verifier flags a pass whose log is wrong in class `c`
- `FA(theta)`     probability it flags a pass whose log is correct
- `delta_c`       probability a supervisor who receives a flag on a genuinely wrong
                  record of class `c` confirms the error and corrects the record

`delta_c` is a property of the human and the record, not of the detector. It is near 0
for adjacent slips and near 1 for a fifteen-stage offset, which is the whole of the
paper's premise.

**Assumption A1.** Confirmation is independent of how the flag arose,
`P(confirm | flag, wrong, c) = delta_c`, independent of `theta`.

A1 is the substantive assumption. It fails if a reviewer is more likely to confirm a
flag they expected (confirmation bias) or less likely after many false alarms (alarm
fatigue). Section 5 states what the data say about it.

## 2. Three propositions

### P1. The qualifying condition carries delta

A class is worth delegating when the expected recovery exceeds the expected review
cost:

    p w_c c_miss * Det_c * delta_c  >  c_FA * FA

which is the manuscript's Eq. (3) with `Det_c` replaced by `Det_c * delta_c`. In margin
form the bar becomes

    Det_c * delta_c - FA > tau(p),      tau(p) = FA (c_FA/(p c_miss) - 1).

So the bar itself is unchanged by the reviewer, but the quantity that must clear it is
attenuated.

### P2. A class the reviewer cannot confirm is unrecoverable

If `delta_c = 0` then `Det_c * delta_c - FA = -FA < 0 <= tau(p)` for every `theta`.

No improvement in the detector moves the class across the bar. Raising `Det_c`,
lowering `FA`, or enlarging the training set all leave the quantity at `-FA`. The only
route is to raise `delta_c`, which is a change to the record or to the review, not to
the model.

This is the proposition the paper currently states in words ("the misses in excluded
classes are never recovered") and never proves. It also explains why a site cannot buy
its way out of the problem: the excluded classes are excluded by the human stage.

### P3. Lower confirmation shifts the optimal threshold towards fewer flags

Let the value of running at `theta` for a class be

    V(theta) = p w_c c_miss * Det(theta) * delta_c - c_FA * FA(theta),

with `Det' < 0`, `FA' < 0`, and the ROC concave, so that `phi(FA) = dDet/dFA` is
positive and decreasing in `FA`. The first-order condition is

    delta_c * p w_c c_miss * phi(FA*) = c_FA,

that is `phi(FA*) = r / delta_c` with `r = c_FA / (p w_c c_miss)`. As `delta_c` falls,
`r / delta_c` rises, so `phi(FA*)` must rise, and since `phi` decreases in `FA`, the
optimal `FA*` falls.

**A class the reviewer cannot confirm should be flagged less, not more.** The instinct
on seeing a hard class is to cast a wider net; the arithmetic says the opposite,
because each unconfirmed flag is a pure cost. This is not in the manuscript and it is
the result worth having.

Corollary. Because P1 attenuates detection, a low-`delta` class needs a *better*
detector to qualify, `Det_c > (tau + FA)/delta_c`, not merely a louder one.

## 3. What the shadow run can and cannot say about delta

The shadow record holds, per REVIEW decision, the recommended action and the response
(follow, change, dismiss). It does not hold whether the flagged record was actually
wrong, because the natural passes were never independently verified.

    change rate = P(change | REVIEW)
    delta       = P(confirm and correct | flag AND the record is wrong)

These differ by `P(true | REVIEW)`, which is unmeasured, and by whether "follow" means
confirmed or merely not contradicted. The counts:

| action | REVIEW | follow | change | dismiss | change rate |
|---|---|---|---|---|---|
| R2 re-check log | 8 | 4 | 2 | 2 | 0.25 |
| R3 inspect gate | 7 | 1 | 6 | 0 | 0.86 |
| R4 escalate | 2 | 1 | 0 | 1 | 0.00 |

The aggregate change rate is 0.471. Under the assumption that a change requires a true
flag, `delta >= 0.471` follows, so the data give a floor and not an estimate. The split
across actions also shows that the response depends on the *action*, not on the class,
which is a violation of the per-class reading of `delta` rather than support for it.

**Conclusion for the theory.** `delta_c` is a parameter that P1 to P3 use and that this
campaign cannot identify. Identifying it needs a sample in which flag truth is verified
independently of the response, and in which "follow" is resolved into confirmed and
not-checked. That is a design, not a reanalysis, and the paper should say so.

## 4. What this buys, and what it costs

| | |
|---|---|
| Buys | a reason the adjacent class is unrecoverable (P2), and a counter-intuitive operating rule (P3) |
| Buys | an honest identification statement instead of a change rate dressed as a confirmation rate |
| Costs | one more parameter that the campaign cannot estimate |
| Risk | `delta` resembles verification bias (Begg and Greenes 1983) and the alert-override literature in clinical decision support; the increment must be P2 and P3, not the two-stage model |

## 5. Where it goes in the manuscript

- Section 3.6, after the injectability row: P1 and the attenuated bar.
- Section 6.2, new subsection: P2 and P3, with the ROC-concavity condition stated.
- Section 6.3.2, the action audit: the identification statement, replacing the current
  reading of the 47.1% as a confirmation rate.
- Section 8: one sentence that `delta` is unestimated, with the design that would.
