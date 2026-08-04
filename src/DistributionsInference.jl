"""
    DistributionsInference

The inference layer for the EpiAware composable-modelling stack: a
PPL-neutral fit protocol (parameter rows, flat dimension, reconstruct
hook) and a `LogDensityProblems`-based log-density engine, with no
probabilistic programming language and no chain type in the core package.

The protocol (`parameter_rows`, `reconstruct`), default-prior assembly over it
(`default_prior`, `with_priors`) and the engine (`distribution_to_logdensity`,
`logdensity`, `FitLogDensity`) are implemented here. Everything else arrives
through an extension: `FlexiChains` (the dotted-name readback,
`to_flexichain`, `distribution_params`, `point_estimate`, `readback_draws`),
`DynamicPPL` (`distribution_to_turing`), `DynamicPPL` x `FlexiChains` (the
`VarName`-keyed dispatch of
`distribution_params`/`point_estimate`/`readback_draws`), `Bijectors`
(`to_constrained` and `logdensity_to_objective` built on it),
and `ComposedDistributions` (the fit protocol over a composed tree's own
codec). The `ModifiedDistributions` extension is parked until that package
registers in General (#17).

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
# Loaded for effect, not for names: together these trigger
# `EpiAwareADToolsLogExpFunctionsMooncakeExt`, which registers the corrected
# `xlogy`/`xlog1py` Mooncake rules the engine needs for Gamma shape-gradients
# at `shape == 1`. Hosted in the kit so this package and
# `ComposedDistributions` cannot load two copies (#73).
using EpiAwareADTools: EpiAwareADTools
using LogExpFunctions: LogExpFunctions

# Must precede any docstring: a `@template` only applies to docstrings
# written after it.
include("docstrings.jl")

include("extensions.jl")
include("protocol.jl")
include("engine.jl")
include("priors.jl")
include("readback.jl")
include("turing.jl")
include("bijectors.jl")
include("public.jl")

end # module DistributionsInference
