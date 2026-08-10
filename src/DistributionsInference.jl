"""
    DistributionsInference

The inference layer for the EpiAware composable-modelling stack: a
PPL-neutral fit protocol (parameter rows, flat dimension, reconstruct
hook), a `LogDensityProblems`-based log-density engine, and a `FlexiChains`
readback, with no probabilistic programming language in the core package.

The protocol (`parameter_rows`, `reconstruct`), default-prior assembly over it
(`default_prior`, `with_priors`), the engine (`distribution_to_logdensity`,
`logdensity`, `FitLogDensity`) and the dotted-name `FlexiChains` readback
(`distribution_params`, `point_estimate`, `distribution_draws`) are
implemented here; `FlexiChains` is a hard dependency. Everything else arrives
through an extension: `DynamicPPL` (`distribution_to_turing`), `DynamicPPL` x
`FlexiChains` (the `VarName`-keyed dispatch of the readback functions above),
`Bijectors` (`to_constrained`, `to_unconstrained` and the
`logdensity_to_objective` / `objective_to_distribution` pair built on them;
`distribution_to_objective` is written in core over that pair, so it needs
`Bijectors` loaded the same way), `ComposedDistributions` (the fit protocol
over a composed tree's own codec), `Optim` (the `minimise` step of
`optimise_distribution`) and `Mooncake` (gradient rules for
`xlogy`/`xlog1py`). The `ModifiedDistributions` extension is parked until that
package registers in General (#17).

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
using FlexiChains: FlexiChains
using Statistics: mean

include("docstrings.jl")

include("extensions.jl")
include("protocol.jl")
include("engine.jl")
include("priors.jl")
include("readback.jl")
include("turing.jl")
include("bijectors.jl")
include("optimise.jl")
include("public.jl")

end # module DistributionsInference
