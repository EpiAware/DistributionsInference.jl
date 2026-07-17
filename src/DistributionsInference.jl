"""
    DistributionsInference

The inference layer for the EpiAware composable-modelling stack: a
PPL-neutral fit protocol (parameter table, flat dimension, reconstruct
hook) and a `LogDensityProblems`-based log-density engine, together with a
dotted-name `FlexiChains` readback that works without a probabilistic
programming language. Extension packages layer `DynamicPPL`,
`ComposedDistributions`, `Bijectors`, and `Mooncake` support on top of this
core (see `ComposedDistributions#185`).

This is a bare scaffold; the protocol and engine land in follow-up issues.

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

# Register the standard EpiAware docstring conventions before any
# docstrings are defined (see src/docstrings.jl).
include("docstrings.jl")

include("public.jl")

end # module DistributionsInference
