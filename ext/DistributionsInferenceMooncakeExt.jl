# Mooncake has no rule for `xlogy`/`xlog1py`, so it derives one from the
# primal, whose `iszero(x)` branch gives `d/dx = 0` at `x == 0` instead of
# `log(y)`. That reaches Gamma log-densities through `gammalogpdf`'s
# `xlogy(shape - 1, x / scale)`, so the shape-gradient is wrong at
# `shape == 1`. `LogExpFunctionsChainRulesCoreExt` ships a correct `rrule` and
# `frule` for both, so import those (both directions) rather than re-derive.
# Deliberate, narrowly-scoped type piracy on functions this package does not
# own; remove once Mooncake ships its own rule (chalk-lab/Mooncake.jl#1241).
module DistributionsInferenceMooncakeExt

using LogExpFunctions: xlogy, xlog1py
using Mooncake: Mooncake

Mooncake.@from_chainrules Mooncake.DefaultCtx Tuple{
    typeof(xlogy), Base.IEEEFloat, Base.IEEEFloat}
Mooncake.@from_chainrules Mooncake.DefaultCtx Tuple{
    typeof(xlog1py), Base.IEEEFloat, Base.IEEEFloat}

end # module
