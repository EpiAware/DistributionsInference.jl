"""
    DistributionsInference

The inference layer for the EpiAware composable-modelling stack: a
PPL-neutral fit protocol (parameter rows, flat dimension, reconstruct
hook) and a `LogDensityProblems`-based log-density engine, with no
probabilistic programming language and no chain type in the core package.

The protocol (`parameter_rows`, `reconstruct`), default-prior assembly over it
(`default_prior`, `with_priors`), the engine (`distribution_to_logdensity`,
`logdensity`, `FitLogDensity`) and the pooled posterior trace
(`PosteriorTrace`, `draws_to_trace`, `parameter_draws`,
`trace_to_distribution`, `point_estimate`) are implemented here, with no
`FlexiChains` dependency. Everything else arrives through an extension:
`FlexiChains` (the dotted-name chain readback,
`distribution_params`, `point_estimate`, `distribution_draws`),
`DynamicPPL` (`distribution_to_turing`), `DynamicPPL` x `FlexiChains` (the
`VarName`-keyed dispatch of
`distribution_params`/`point_estimate`/`distribution_draws`), `Bijectors`
(`to_constrained`, `to_unconstrained` and the `logdensity_to_objective` /
`objective_to_distribution` pair built on them),
`ComposedDistributions` (the fit protocol over a composed tree's own codec),
`Optim` (the `minimise` step of `optimise_distribution`) and `Mooncake`
(gradient rules for `xlogy`/`xlog1py`). The
`ModifiedDistributions` extension is parked until that package registers in
General (#17).

```@example
using DistributionsInference
```
"""
module DistributionsInference

using Distributions: Distributions
using DocStringExtensions: @template, DOCSTRING, EXPORTS, IMPORTS,
                           TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using LogDensityProblems: LogDensityProblems
using EpiAwareADTools: EpiAwareADTools
using LogExpFunctions: LogExpFunctions
using Statistics: Statistics

include("docstrings.jl")

include("extensions.jl")
include("protocol.jl")
include("engine.jl")
include("priors.jl")
include("readback.jl")
include("trace.jl")
include("turing.jl")
include("bijectors.jl")
include("optimise.jl")
include("public.jl")

end # module DistributionsInference
