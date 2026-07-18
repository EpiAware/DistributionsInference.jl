## Unreleased

Added the fit protocol (`parameter_rows`, `estimated_rows`, `flat_dimension`,
`reconstruct`) generalising ComposedDistributions' `params_table` shape, and
the PPL-neutral log-density engine built on it (`FitLogDensity`,
`as_logdensity`, `logdensity`) with a direct `LogDensityProblems`
implementation.

Added the dotted-name `FlexiChains` readback (`to_flexichain`, `readback`,
`readback_draws`): build a chain from raw draws of any
`LogDensityProblems`-compatible sampler, keyed by the estimated parameter
rows' dotted names, and read it back onto a fitted object, with no PPL
involved anywhere. Generalises ComposedDistributions'
`chain_to_params`/`param_draws` (closes #3).

This file tracks notes for major releases and significant milestones; GitHub
Releases (auto-generated from merged PRs) cover every release in between.
