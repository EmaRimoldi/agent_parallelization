# Validation Strategy

## Objective

Resolve the status of the AutoResearch theorem by iterating over:

1. a **formal audit**,
2. a **protocol-compliance audit**,
3. **high-information experiments** that test the theorem's most fragile assumptions,
4. and, if needed, a **reformulation** of the theory and regenerated PDF.

## Why This Strategy Is Optimal For This Theory

This theory can fail in two very different ways:

- the algebra may not follow from BP as claimed,
- or the algebra may be fine but the empirical protocol may fail to identify the required terms.

If these are mixed together, the diagnosis becomes noisy and misleading.
So the optimal strategy is to separate them.

## Highest-Value Initial Experiments

### 1. Verifier-noise assay

Why first:

- the AutoResearch paper introduces latent loss specifically because the verifier is noisy,
- if verifier noise is negligible, that extension is unnecessary,
- if verifier noise is large, then the incumbent re-evaluation protocol is not optional.

This experiment measures repeated `val_bpb` on the same `train.py` to estimate:

- variance,
- range,
- and whether pilot-level differences are smaller than noise.

### 2. Protocol-compliance audit

Why second:

- the theorem may be sound while the pilot is incapable of testing it,
- a false negative due to missing mode labels or wrong cost axis is an experimental failure, not a refutation.

This audit checks:

- were mode labels produced for all pilot runs?
- were `phi`, `G`, and `epsilon` actually estimable?
- was incumbent re-evaluation performed?
- was cost equalization implemented?
- does the empirical cost axis match the paper's definition?

### 3. Context-pressure identifiability audit

Why third:

- the paper claims context pressure is one of the three essential extensions,
- but H5 only makes sense if the experiment reaches enough `c/K` range.

If context fill never leaves the first quartile, H5 is not falsified; it is untested.

### 4. Empirical prediction audit on the 2x2 pilot

Why fourth:

- once the measurement issues are understood, the pilot can be read honestly,
- this tells us whether the theory is supported, unsupported, or simply under-measured.

## Most Likely Break Points

These are the failure points I currently consider most probable, in descending order.

1. **Loss/reward mismatch**
   The AutoResearch theorem is written in terms of latent loss and threshold success, while BP Appendix A is written in terms of continuous rewards.

2. **`kappa_bar` substitution**
   BP assumes fixed per-step cost; AutoResearch uses context-dependent cost and then averages it.

3. **Mode estimator collapse**
   If accepted edits are sparse or mode labels are missing, `G` and `epsilon` become numerically degenerate even if the theory is correct.

4. **Protocol mismatch on the cost axis**
   The current substrate uses CPU and token-only cost, while the PDF's motivating definition uses GPU-hours plus tokens.

5. **Context pressure never entering the intended regime**
   If `c/K` stays low, the mechanism may be real but unidentifiable.

## Iteration Structure

Each iteration is documented with the same schema:

- **Claim tested**
- **Why this claim matters**
- **Procedure**
- **Input files / runs**
- **Raw outputs**
- **Interpretation**
- **Diagnosis**
- **Action**

This structure makes failures as valuable as confirmations:

- if the experiment is broken, fix the experiment;
- if the claim is under-specified, refine the theory;
- if the data contradict a theorem-level prediction under a compliant protocol, count that against the theory.

## Documentation Rules For Traceability

For each iteration I keep:

- a human-readable markdown summary,
- machine-readable JSON where possible,
- raw logs,
- and exact source files used for the analysis.

An external model should be able to reconstruct the conclusion without rerunning the repo.

## Criteria For "Proved" vs "Confuted"

I will only call the theorem **validated / effectively proved for this setting** if all of the following hold:

1. The reduction from AutoResearch to BP is explicit and gap-free.
2. The empirical protocol measures all four terms in non-degenerate form.
3. The key estimators are stable under bootstrap / reruns.
4. The pilot predictions have consistent direction across repetitions.
5. No simpler scalar explanation subsumes the decomposition.

I will call the theorem **confuted for the current formulation** if any of the following happen:

1. A required formal bridge to BP fails.
2. A theorem-level consequence is contradicted under a compliant experimental protocol.
3. A claimed essential mechanism is unidentifiable in the intended setup and the theorem depends on it.

There is also an intermediate outcome:

> **Theory not yet validated because the current formulation or protocol is insufficient.**

This is the most likely outcome when the mathematics is close but not fully closed, or when the pilot is informative but under-instrumented.

If that happens, the next step is not to stop. It is to reformulate the theorem and regenerate the PDF.
