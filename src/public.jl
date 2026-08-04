# Public API declarations for Julia 1.11+ (public but not exported).

# The fit protocol. A fittable object implements `parameter_rows` and
# `reconstruct`; the rest is generic on top of them.
public parameter_rows, estimated_rows, flat_dimension, reconstruct,
       extra_logprior, extra_prior_state

# Default-prior assembly over the protocol.
public default_prior, distribution_priors

# The PPL-neutral log-density engine.
public FitLogDensity, as_logdensity, logdensity

# The dotted-name `FlexiChains` readback. Declared here as chain-free stubs
# (with their docstrings, in `readback.jl`); every method lives in
# `DistributionsInferenceFlexiChainsExt`, and the last three also dispatch on a
# `VarName`-keyed chain once `DynamicPPL` is loaded alongside `FlexiChains`.
public to_flexichain, distribution_params, readback, readback_draws

# A DynamicPPL model over a fittable object's estimated parameters. Stub here
# (docstring in `turing.jl`); the model lives in
# `DistributionsInferenceDynamicPPLExt`.
public as_turing

# The unconstrained <-> constrained transform and the negative unconstrained
# log-posterior built on it. Stubs here (docstrings in `bijectors.jl`); both
# methods live in `DistributionsInferenceBijectorsExt`.
public to_constrained, as_optimisation_objective
