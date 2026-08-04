"""
    DistributionsInference

The inference layer for the EpiAware composable-modelling stack: a
PPL-neutral fit protocol (parameter rows, flat dimension, reconstruct
hook) and a `LogDensityProblems`-based log-density engine, with no
probabilistic programming language and no chain type in the core package.

The protocol (`parameter_rows`, `reconstruct`), default-prior assembly over it
(`default_prior`, `distribution_priors`) and the engine (`as_logdensity`,
`logdensity`, `FitLogDensity`) are implemented here. Everything else arrives
through an extension: `FlexiChains` (the dotted-name readback,
`to_flexichain`, `distribution_params`, `readback`, `readback_draws`),
`DynamicPPL` (`as_turing`), `DynamicPPL` x `FlexiChains` (the `VarName`-keyed
dispatch of `distribution_params`/`readback`/`readback_draws`), `Bijectors`
(`to_constrained` and `as_optimisation_objective` built on it),
`ComposedDistributions` (the fit protocol over a composed tree's own codec)
and `Mooncake` (gradient rules for `xlogy`/`xlog1py`). The
`ModifiedDistributions` extension is parked until that package registers in
General (#17).

```@example
using DistributionsInference
```
"""
module DistributionsInference

# All module-scope `using`/`import` statements live here, not in the
# included files.
using Distributions: Distributions
using DocStringExtensions: @template, DOCSTRING, EXPORTS, IMPORTS,
                           TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using LogDensityProblems: LogDensityProblems

# Must precede any docstring: a `@template` only applies to docstrings
# written after it.
include("docstrings.jl")

include("protocol.jl")
include("engine.jl")
include("priors.jl")
include("readback.jl")
include("turing.jl")
include("bijectors.jl")
include("public.jl")

end # module DistributionsInference
