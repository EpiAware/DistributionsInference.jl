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

`extra_logprior` gains a fourth argument, `state`: `extra_prior_state(obj)`,
computed once when `as_logdensity` assembles a `FitLogDensity` and threaded
into every subsequent `extra_logprior` call, mirroring how `flat_priors` is
already collected once rather than re-derived per evaluation. Profiling
(#28) found the ComposedDistributions extension's `extra_logprior` was
paying a full `params_table` walk on every single evaluation to find which
rows carry a centred-pooled prior, whether or not the tree actually has any
— contradicting its own "no extra cost in the common case" claim. That walk
now runs once, in a new `extra_prior_state` method, and `extra_logprior`
reads the cached rows instead. A type overriding `extra_logprior` updates its
method to the new four-argument signature (breaking; the package is
unreleased, so this ships without a deprecation path); most existing
overrides need no precomputed state and can add an ignored trailing
argument.

Documented the rule for a conditionally available exact likelihood in
`as_logdensity`'s `loglik` docstring: choose between the exact and
approximate branch with an explicit predicate or by dispatch, never by
catching an exception thrown from the exact path, and refuse loudly with a
named structural reason where the exact form does not apply. Concluded no
dedicated helper is needed for this: the existing `loglik` reducer hook plus
this convention already cover it, so the helper question is closed rather
than built (closes #44).

Added `as_optimisation_objective` (the `Bijectors` extension, alongside
`to_constrained`): a plain `AbstractVector -> Real` callable — the negative
unconstrained log-posterior — that composes `to_constrained` with the core
`logdensity`, so a standard external optimisation package (`Optim.jl`,
`Optimization.jl`, or any package accepting a callable and an initial vector)
finds the maximum-likelihood or maximum-a-posteriori point directly, with
gradients through the existing AD wrapper. No estimator method and no
optimisation package are added anywhere: this is the thin transform/objective
wiring the fitting layer needed to make an external optimiser usable
out of the box (closes #46).

Fixed: a `reconstruct` method whose ESTIMATED field is typed to a concrete
number (e.g. `shape::Float64`) rather than left generic used to fail a
gradient-based sampler with an opaque `MethodError` from inside the struct's
own constructor. `logdensity` and the `DynamicPPL` extension's turing model
now guard this ahead of time, raising a clear, named `ArgumentError` before
`reconstruct` runs (closes #48). Also documented the custom likelihood-reducer
route in `reconstruct`'s docstring: an ESTIMATED field's type must stay
GENERIC so a tracer number can flow through it.

The ComposedDistributions extension's `extra_prior_state`/`extra_logprior`
now call ComposedDistributions' `centred_pool_rows`/`pool_centred_logprior`,
dropping the leading underscore: the org's naming convention reserves a
leading underscore for internal-only names, and these two are `public`
(ComposedDistributions#212). ComposedDistributions keeps the old
`_centred_pool_rows`/`_pool_centred_logprior` names as `public` transitional
aliases, so this is a no-op release-wise; it only needs ComposedDistributions'
rename to have landed first.

This file tracks notes for major releases and significant milestones; GitHub
Releases (auto-generated from merged PRs) cover every release in between.
