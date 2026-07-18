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

Added the `Bijectors` weakdep extension (`DistributionsInferenceBijectorsExt`):
`to_constrained(prob, z)` maps an unconstrained flat vector to the constrained
ESTIMATED parameters plus a log-Jacobian, built per row from
`FitLogDensity`'s stored `flat_priors` via `Bijectors.bijector`, so
`logdensity(prob, x) + logjac` is the unconstrained-space log-target a
sampler works with. Generalises ComposedDistributions' `to_constrained`
(closes #6).

Added the `ComposedDistributions` weakdep extension
(`DistributionsInferenceComposedDistributionsExt`): `parameter_rows`,
`estimated_rows`, `flat_dimension`, `reconstruct` and `extra_logprior` for a
composed distribution, over CD's existing public codec (`params_table`,
`flat_dimension`, `unflatten`, `reconstruct`) rather than new inference logic —
a centred pooled parameter's population-dependent prior is scored through
`extra_logprior`, and every other estimated row (stick-breaking `Resolve`
coordinates, pooled z-latents/hyperparameters, shared tags) maps straight onto
a dotted-name row. `to_flexichain`/`readback`/`readback_draws`/`as_turing` need
no ComposedDistributions-specific code at all once the row protocol is
correct. Generalises ComposedDistributions' `as_logdensity`/`ComposedLogDensity`
(closes #5).

Added the `ModifiedDistributions` weakdep extension
(`DistributionsInferenceModifiedDistributionsExt`): `parameter_rows` and
`reconstruct` for a STANDALONE modifier distribution (`affine(Gamma(...))`,
`thin(...)`, ...), not only as a leaf inside a composed tree. A modifier's
fixed structure (an `Affine`'s scale/shift, a `Weighted`'s weight, a
`Modified`'s hazard effect/link) is peeled through and not reported as a row,
mirroring ComposedDistributions' own leaf-protocol precedent; a `thin`
factor is the one modifier-owned parameter reported as an extra row. Every
row is fixed by design (a bare `Distributions.jl` leaf carries no prior
anywhere to mark one estimated) — `distribution_priors` plus a
caller-supplied `loglik` rebuilding the concrete modifier from row values is
the generic path to fitting one (closes #17).

This file tracks notes for major releases and significant milestones; GitHub
Releases (auto-generated from merged PRs) cover every release in between.
