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

# The dotted-name `FlexiChains` readback. Declared here as chain-free stubs
# (with their docstrings, in `readback.jl`); every method lives in
# `DistributionsInferenceFlexiChainsExt`, and all three also dispatch on a
# `VarName`-keyed chain once `DynamicPPL` is loaded alongside `FlexiChains`.
#
# `distribution_params` reads a chain's estimated values by name;
# `point_estimate` and `distribution_draws` rebuild the distribution from them,
# reduced to one object or kept one per draw. Building a chain out of a
# sampler's raw draws is not on the surface: that conversion belongs to
# `FlexiChains` or to the inference package that produced them (#104).
export distribution_params, point_estimate, distribution_draws

# The pooled posterior trace (#90), core only (no `FlexiChains`): a caller
# builds one with `draws_to_trace` and reads it with `parameter_draws`,
# `trace_to_distribution`, or `point_estimate` (a second method on the name
# just exported above — it dispatches on a trace instead of a chain, so the
# two never collide). `PosteriorTrace` itself stays `public`, not exported,
# matching `FitLogDensity`: a caller mostly holds an instance without
# spelling the type name, the way a caller who never annotates a return type
# does not need it imported.
export draws_to_trace, parameter_draws, trace_to_distribution
public PosteriorTrace

# A DynamicPPL model over a fittable object's estimated parameters. Stub here
# (docstring in `turing.jl`); the model lives in
# `DistributionsInferenceDynamicPPLExt`.
export distribution_to_turing

# The unconstrained <-> constrained transform and the objective round trip
# built on it. Stubs here (docstrings in `bijectors.jl`); every method lives in
# `DistributionsInferenceBijectorsExt`. Both transforms are what an engine
# author calls rather than fitting entry points, so they stay unexported, while
# the two ends of the objective round trip a caller writes are exported.
export logdensity_to_objective, objective_to_distribution
public to_constrained, to_unconstrained

# The one-call optimiser fit, and the single optimiser-package-specific step it
# calls (docstrings in `optimise.jl`). `optimise_distribution` is written in
# core over the transforms above; `minimise` is a hook an optimiser package's
# extension fills in, so it is documented and public without being exported.
# The `Optim` method lives in `DistributionsInferenceOptimExt`.
export optimise_distribution
public minimise
