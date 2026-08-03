"""
    DistributionsInference

The inference layer for the EpiAware composable-modelling stack: a
PPL-neutral fit protocol (parameter rows, flat dimension, reconstruct
hook) and a `LogDensityProblems`-based log-density engine, with no
probabilistic programming language and no chain type in the core package.
Extension packages layer `FlexiChains`, `DynamicPPL`,
`ComposedDistributions`, `Bijectors`, and `Mooncake` support on top of this
core (see `ComposedDistributions#185`).

The protocol (`parameter_rows`, `reconstruct`), default-prior assembly over it
(`default_prior`, `distribution_priors`) and the engine (`as_logdensity`,
`logdensity`, `FitLogDensity`) are implemented here. Everything else arrives
through an extension: `FlexiChains` (the dotted-name readback,
`to_flexichain`, `distribution_params`, `readback`, `readback_draws`),
`DynamicPPL` (`as_turing`), `DynamicPPL` x `FlexiChains` (the `VarName`-keyed
dispatch of `distribution_params`/`readback`/`readback_draws`), `Bijectors`
(`to_constrained`, the prior-driven unconstrained <-> constrained transform,
and `as_optimisation_objective` built on it), `ComposedDistributions` (the fit
protocol over a composed tree's own codec, `extra_logprior` included) and
`Mooncake` (gradient rules for `xlogy`/`xlog1py`, which the engine reaches
through a Gamma log-density). The `ModifiedDistributions` extension is parked
until that package registers in General (#17).

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
using LogDensityProblems: LogDensityProblems

# Register the standard EpiAware docstring conventions before any
# docstrings are defined (see src/docstrings.jl).
include("docstrings.jl")

# The fit protocol (`parameter_rows`, `estimated_rows`, `flat_dimension`,
# `reconstruct`) and the PPL-neutral log-density engine built on it
# (`FitLogDensity`, `as_logdensity`, `logdensity`, the `LogDensityProblems`
# interface).
include("protocol.jl")
include("engine.jl")

# Default-prior assembly over the fit protocol (`default_prior`,
# `distribution_priors`): the generic, params-first analogue of
# ComposedDistributions' `build_priors`/`param_priors` (CD#195/DI#20), over
# any `parameter_rows`-implementing object rather than a composed-distribution
# tree specifically.
include("priors.jl")

# The dotted-name `FlexiChains` readback: build a chain from raw sampler
# draws (`to_flexichain`) and read it back onto a fitted object
# (`readback`, `readback_draws`). Chain-free stubs whose methods live in the
# weakdep `DistributionsInferenceFlexiChainsExt` extension (`ext/`); no PPL is
# involved on either side of that split.
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
