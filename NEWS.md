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

Added the `DynamicPPL` weakdep extension (`DistributionsInferenceDynamicPPLExt`):
`as_turing` builds a DynamicPPL model over a fittable object's estimated
parameters, a light wrapper on `as_logdensity` whose `~` sites are named to
match the readback's dotted names, so its total log-density equals the
engine's `logdensity` at the corresponding point. `readback`/`readback_draws`
gain a `VarName`-keyed dispatch, so a chain sampled from `as_turing` (e.g.
with `chain_type = FlexiChains.VNChain`) reads back unchanged. Generalises
ComposedDistributions' `as_turing`/FlexiChains `VarName` readback (closes #4).

This file tracks notes for major releases and significant milestones; GitHub
Releases (auto-generated from merged PRs) cover every release in between.
