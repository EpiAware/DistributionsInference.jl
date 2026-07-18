"""
    DistributionsInference

The inference layer for the EpiAware composable-modelling stack: a
PPL-neutral fit protocol (parameter rows, flat dimension, reconstruct
hook) and a `LogDensityProblems`-based log-density engine, together with a
dotted-name `FlexiChains` readback that works without a probabilistic
programming language. Extension packages layer `DynamicPPL`,
`ComposedDistributions`, `Bijectors`, and `Mooncake` support on top of this
core (see `ComposedDistributions#185`).

The protocol (`parameter_rows`, `reconstruct`), the engine (`as_logdensity`,
`logdensity`, `FitLogDensity`), and the dotted-name `FlexiChains` readback
(`to_flexichain`, `readback`, `readback_draws`) are implemented, together with
the `DynamicPPL` extension (`as_turing`, and a `VarName`-keyed dispatch of
`readback`/`readback_draws`) and the `Bijectors` extension (`to_constrained`,
the prior-driven unconstrained <-> constrained transform); the remaining
extension packages land in follow-up issues.

```@example
using DistributionsInference
```
"""
module DistributionsInference

# All genuine module-scope `using`/`import` statements live here, in
# the main module file, rather than scattered across included files.
using Distributions: Distributions
using DocStringExtensions: @template, DOCSTRING, EXPORTS, IMPORTS,
                           TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using FlexiChains: FlexiChains
using LogDensityProblems: LogDensityProblems
using Statistics: mean

# Register the standard EpiAware docstring conventions before any
# docstrings are defined (see src/docstrings.jl).
include("docstrings.jl")

# The fit protocol (`parameter_rows`, `estimated_rows`, `flat_dimension`,
# `reconstruct`) and the PPL-neutral log-density engine built on it
# (`FitLogDensity`, `as_logdensity`, `logdensity`, the `LogDensityProblems`
# interface).
include("protocol.jl")
include("engine.jl")

# The dotted-name `FlexiChains` readback: build a chain from raw sampler
# draws (`to_flexichain`) and read it back onto a fitted object
# (`readback`, `readback_draws`). `FlexiChains` is a hard dependency, so
# this needs no PPL and no glue extension.
include("readback.jl")

# `as_turing`: a DynamicPPL model over a fittable object's estimated
# parameters, a Turing-free stub whose method lives in the weakdep
# `DistributionsInferenceDynamicPPLExt` extension (`ext/`).
include("turing.jl")

# `to_constrained`: the unconstrained <-> constrained transform, a
# Bijectors-free stub whose method lives in the weakdep
# `DistributionsInferenceBijectorsExt` extension (`ext/`).
include("bijectors.jl")

include("public.jl")

end # module DistributionsInference
