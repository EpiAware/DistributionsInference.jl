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
