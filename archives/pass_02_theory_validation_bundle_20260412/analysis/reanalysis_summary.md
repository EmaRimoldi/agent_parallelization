# Reanalysis Summary

## Outcome

After rerunning the theory audit with the narrowed theorem, upgraded protocol, corrected estimators, and targeted follow-up experiments, the project now sits in this state:

- mathematically cleaner than before
- experimentally more honest than before
- still not empirically identified well enough for a full theorem claim

## What Changed

1. **The theorem changed**
   The strongest defensible statement is now the single-axis decomposition with a Jensen remainder, not the older shared two-axis theorem.

2. **The protocol changed**
   There is now a real reevaluation path, structured reevaluation logs, and stable candidate identity.

3. **The estimators changed**
   `phi`, `G`, and `epsilon` are no longer placeholder formulas. They are now structurally aligned proxies with explicit failure modes.

4. **The experiment picture changed**
   Repeated incumbent evaluations show `d10` still looks best on mean, but the confidence intervals overlap too heavily to support a decisive ranking claim.

## Narrowest Defensible Theorem Now

The narrowest defensible theorem is:

> For a fixed axis, if AutoResearch admits an architecture-indexed packed family, if success is defined on latent-loss thresholding, and if the arithmetic mean cost is used only with an explicit Jensen remainder, then the BP-style decomposition applies conditionally on that axis.

This is a real improvement over the earlier statement because:

- the remainder is explicit,
- the theorem no longer leans on broken estimators,
- and axis sharing is no longer assumed for free.

## Why The Status Did Not Upgrade To “Rigorous”

Three blockers remain:

1. accepted-mode overlap is too sparse for stable `phi` estimation
2. some cells still have no accepted posterior, so `G` and `epsilon` remain undefined in practice
3. context pressure is still not experimentally controllable enough to test H5

## Practical Reading

The framework is now in a better research state than before:

- theorem: cleaner
- code: cleaner
- logging: cleaner
- empirical claim: still weak

That is why the final classification is **promising but not yet rigorous** rather than **empirically unsupported in current form** or **rigorous under explicit assumptions**.
