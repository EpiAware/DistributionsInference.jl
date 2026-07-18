# Public API declarations for Julia 1.11+ (public but not exported).

# The fit protocol: a fittable object implements `parameter_rows` (and
# `reconstruct`) with its own method; `estimated_rows`/`flat_dimension` are
# generic on top of `parameter_rows`. `extra_logprior` is the neutral-default
# hook for an object-dependent prior term (a hierarchical population, say)
# that cannot be scored per-row against the flat vector alone. Public but not
# exported (like ComposedDistributions' matching codec surface), reached by
# qualified name.
public parameter_rows, estimated_rows, flat_dimension, reconstruct,
       extra_logprior

# The PPL-neutral log-density engine built on the protocol above: assembles a
# `FitLogDensity` from any fit-protocol object and data, and evaluates its
# log-posterior over the estimated flat parameters. `LogDensityProblems` is a
# hard dependency here, so the interface is implemented directly on
# `FitLogDensity` with no glue extension needed.
public FitLogDensity, as_logdensity, logdensity

# The dotted-name `FlexiChains` readback: `to_flexichain` builds a chain from
# raw sampler draws keyed by the estimated rows' dotted names; `readback` and
# `readback_draws` read it back onto a fitted object (point summary/draw, and
# every draw respectively). `FlexiChains` is a hard dependency, so this needs
# no PPL and no glue extension. Both also dispatch on a `VarName`-keyed chain
# (e.g. sampled from `as_turing`) once `DynamicPPL` is loaded (see
# `ext/DistributionsInferenceDynamicPPLExt.jl`).
public to_flexichain, readback, readback_draws

# `as_turing`: a DynamicPPL model over a fittable object's estimated
# parameters, a light wrapper on `as_logdensity`. Declared here (with its
# docstring, in `turing.jl`) as a Turing-free stub; the model itself lives in
# the `DistributionsInferenceDynamicPPLExt` package extension, loaded only
# when `DynamicPPL` is present.
public as_turing
