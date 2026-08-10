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
# `Turing` in it owns that name too.
export distribution_to_logdensity
public FitLogDensity, logdensity

# The dotted-name `FlexiChains` readback (`FlexiChains` is a hard dependency;
# implemented directly in `readback.jl`/`inference.jl`). All five also
# dispatch on a `VarName`-keyed chain once `DynamicPPL` is loaded alongside
# this package (`DistributionsInferenceDynamicPPLFlexiChainsExt`).
#
# `distribution_params` reads a chain's estimated values by name;
# `point_estimate` and `distribution_draws` rebuild the distribution from them,
# reduced to one object or kept one per draw. Building a chain out of a
# sampler's raw draws is not on the surface: that conversion belongs to
# `FlexiChains` or to the inference package that produced them (#104).
export distribution_params, point_estimate, distribution_draws

# The posterior-output API (additive; `point_estimate`/`distribution_draws`
# above are unaffected). `inference_to_distribution` is the equal-weight
# `MixtureModel` over selected draws (2-argument form) or the marginal
# plug-in `Distribution` (3-argument form, the reduction positional);
# `inference_to_distributions` is the vectorised, every-draw form.
# `inference_to_dist`/`inference_to_dists` are aliases for the two,
# documented as such with the full names canonical.
export inference_to_distribution, inference_to_distributions,
       inference_to_dist, inference_to_dists

# A DynamicPPL model over a fittable object's estimated parameters. Stub here
# (docstring in `turing.jl`); the model lives in
# `DistributionsInferenceDynamicPPLExt`.
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
