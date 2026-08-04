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
# `DistributionsInferenceFlexiChainsExt`, and the last three also dispatch on a
# `VarName`-keyed chain once `DynamicPPL` is loaded alongside `FlexiChains`.
#
# `readback_draws` keeps the `readback` stem the `point_estimate` rename gave
# up. It returns one object per draw, so neither a plural of `point_estimate`
# nor a `distribution_*` spelling describes it; the two are related in their
# docstrings instead of in their names.
export to_flexichain, distribution_params, point_estimate, readback_draws

# A DynamicPPL model over a fittable object's estimated parameters. Stub here
# (docstring in `turing.jl`); the model lives in
# `DistributionsInferenceDynamicPPLExt`.
export distribution_to_turing

# The unconstrained <-> constrained transform and the negative unconstrained
# log-posterior built on it. Stubs here (docstrings in `bijectors.jl`); both
# methods live in `DistributionsInferenceBijectorsExt`. `to_constrained` is a
# transform an engine author calls, not a fitting entry point, so it stays
# unexported.
export logdensity_to_objective
public to_constrained
