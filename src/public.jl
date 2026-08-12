# The public API surface: `export` for the names a caller writes, `public`
# (Julia 1.11+) for the ones a caller implements or that a package loaded
# alongside this one already owns.

# The fit protocol. A fittable object implements `parameter_rows` and
# `reconstruct`; the rest is generic on top of them. `parameter_rows` is
# exported because a caller reads it to see what a distribution declares;
# `reconstruct` and `flat_dimension` are the engine's side of the protocol, so
# they stay `public`.
export parameter_rows
public estimated_rows, flat_dimension, reconstruct,
       extra_logprior, extra_prior_state

# Default-prior assembly over the protocol. `default_prior` is the per-row
# hook `with_priors` calls, and `ComposedDistributions` exports a function of
# that name, so exporting it here would make `using DistributionsInference,
# ComposedDistributions` ambiguous on first use.
export with_priors
public default_prior

# The PPL-neutral log-density engine. `logdensity` stays unexported: it is a
# second spelling of `LogDensityProblems.logdensity`, which `FitLogDensity`
# implements already, and a session with either `LogDensityProblems` or
# `Turing` in it owns that name too. `template`/`observations`/`flat_priors`
# are the accessors an engine author reaches for instead of a `FitLogDensity`
# struct field directly.
export distribution_to_logdensity
public FitLogDensity, logdensity, template, observations, flat_priors

# `draws_to_chain` keys raw sampler draws (a `dim x niter` matrix, or a
# vector of `dim`-length vectors) into a dotted-name `FlexiChain`
# (`FlexiChains` is a hard dependency; implemented directly in
# `readback.jl`). Public because a sampling verb whose package hands back raw
# draws rather than a chain of its own (`distribution_to_advancedmh`, #94)
# needs it to build one; it stays `public` rather than exported since it is a
# sampler-extension-author tool, not an everyday call.
public draws_to_chain

# The posterior-output API, and the only way this package reads a chain (or
# raw draws) back onto a fitted object: `inference_to_distribution` is the
# equal-weight `MixtureModel` over selected draws (2-argument form, needs
# `reconstruct(obj, x)` to return a `Distribution` — mixing non-distributions
# is meaningless) or the marginal plug-in (3-argument form, the reduction
# positional; generic, like the plural form below, since it never mixes);
# `inference_to_distributions` is the vectorised, every-draw form, also
# generic. Both also dispatch on a `VarName`-keyed chain once `DynamicPPL` is
# loaded alongside this package
# (`DistributionsInferenceDynamicPPLFlexiChainsExt`).
# `inference_to_dist`/`inference_to_dists` are aliases for the two,
# documented as such with the full names canonical.
export inference_to_distribution, inference_to_distributions,
       inference_to_dist, inference_to_dists

# The raw-draws access verb (#90): `inference_to_parameters` is exported,
# same tier as `inference_to_distributions` above, since it is an everyday
# call a caller writes directly (the parameter draws themselves, e.g. to
# build a `DataFrame`), not an implementer- or engine-facing tool.
# `inference_to_parameter_distribution` is exported for the same reason —
# a caller reaches for it directly for a joint-posterior Gaussian
# approximation (e.g. Markov melding) — even though its only method lives
# behind the `Bijectors` extension; `distribution_to_objective` sets the same
# precedent (exported, extension-backed).
export inference_to_parameters, inference_to_parameter_distribution

# DynamicPPL forms over a fittable object's estimated parameters. Stubs here
# (docstrings in `turing.jl`); both methods live in
# `DistributionsInferenceDynamicPPLExt` — `distribution_to_turing(obj, data)`
# builds the model, `distribution_to_turing(obj, data, sampler, nsamples)`
# samples it and returns a `FlexiChain` (#94), one function with two
# concrete return types picked by arity, the same pattern
# `inference_to_distribution` uses.
export distribution_to_turing

# The unconstrained <-> constrained transform and the objective round trip
# built on it. Stubs here (docstrings in `bijectors.jl`); every method lives in
# `DistributionsInferenceBijectorsExt`. Both transforms are what an engine
# author calls rather than fitting entry points, so they stay unexported, while
# the two ends of the objective round trip a caller writes are exported.
# `distribution_to_objective` is the composed route to the same objective
# straight from a distribution and data, so it is exported alongside them by
# the same logic; it is written in core, calling the other two, rather than
# having a method of its own in the extension.
export logdensity_to_objective, objective_to_distribution,
       distribution_to_objective
public to_constrained, to_unconstrained

# The one-call optimiser fit, and the single optimiser-package-specific step it
# calls (docstrings in `optimise.jl`). `optimise_distribution` is written in
# core over the transforms above; `minimise` is a hook an optimiser package's
# extension fills in, so it is documented and public without being exported.
# The `Optim` method lives in `DistributionsInferenceOptimExt`.
export optimise_distribution
public minimise

# `distribution_to_advancedmh` (#94): samples `AdvancedMH`'s random-walk
# Metropolis and returns a `FlexiChain`, the same naming convention
# `distribution_to_turing` sets — a verb per sampling package rather than an
# engine-object layer on top of them. Stub here (docstring in
# `advancedmh.jl`); the method lives in `DistributionsInferenceAdvancedMHExt`.
export distribution_to_advancedmh
