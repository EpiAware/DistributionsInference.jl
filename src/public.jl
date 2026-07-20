# Public API declarations for Julia 1.11+ (public but not exported).

# The fit protocol: a fittable object implements `parameter_rows` (and
# `reconstruct`) with its own method; `estimated_rows`/`flat_dimension` are
# generic on top of `parameter_rows`. `extra_logprior` is the neutral-default
# hook for an object-dependent prior term (a hierarchical population, say)
# that cannot be scored per-row against the flat vector alone. Public but not
# exported (like ComposedDistributions' matching codec surface), reached by
# qualified name.
public parameter_rows, estimated_rows, flat_dimension, reconstruct,
       extra_logprior, extra_prior_state

# Default-prior assembly over the protocol above (CD#195/DI#20): `default_prior`
# picks a support-derived prior for one `parameter_rows` row (mirroring
# ComposedDistributions' own `default_prior`, a parallel implementation since
# DI cannot depend on CD's copy); `distribution_priors` assembles a fully
# prior'd row set for a whole fittable object, the generic estimate-everything
# path ComposedDistributions' `param_priors`/`uncertain(tree)` generalises.
public default_prior, distribution_priors

# The PPL-neutral log-density engine built on the protocol above: assembles a
# `FitLogDensity` from any fit-protocol object and data, and evaluates its
# log-posterior over the estimated flat parameters. `LogDensityProblems` is a
# hard dependency here, so the interface is implemented directly on
# `FitLogDensity` with no glue extension needed.
public FitLogDensity, as_logdensity, logdensity

# The dotted-name `FlexiChains` readback: `to_flexichain` builds a chain from
# raw sampler draws keyed by the estimated rows' dotted names;
# `distribution_params` is the params-first primitive (the estimated values,
# keyed by dotted name, before any rebuild); `readback` and `readback_draws`
# read the chain back onto a fitted object (point summary/draw, and every
# draw respectively) — `readback` is a thin layer over `distribution_params`,
# `readback_draws` its own optimised implementation. `FlexiChains` is a hard
# dependency, so this needs no PPL and no glue extension. All three also
# dispatch on a `VarName`-keyed chain (e.g. sampled from `as_turing`) once
# `DynamicPPL` is loaded (see `ext/DistributionsInferenceDynamicPPLExt.jl`).
public to_flexichain, distribution_params, readback, readback_draws

# `as_turing`: a DynamicPPL model over a fittable object's estimated
# parameters, a light wrapper on `as_logdensity`. Declared here (with its
# docstring, in `turing.jl`) as a Turing-free stub; the model itself lives in
# the `DistributionsInferenceDynamicPPLExt` package extension, loaded only
# when `DynamicPPL` is present.
public as_turing

# `to_constrained`: the prior-driven unconstrained <-> constrained transform
# over a `FitLogDensity`'s estimated parameters. Declared here (with its
# docstring, in `bijectors.jl`) as a Bijectors-free stub; the transform itself
# lives in the `DistributionsInferenceBijectorsExt` package extension, loaded
# only when `Bijectors` is present.
public to_constrained
