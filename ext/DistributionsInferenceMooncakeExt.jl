# DistributionsInference x Mooncake: imports Mooncake primitives for
# `LogExpFunctions.xlogy`/`xlog1py` on `Base.IEEEFloat` arguments, fixing
# ComposedDistributions#99 for the engine this package now owns. Mooncake has
# no rule for either function, so it derives one from the primal
# implementation
#
#     xlogy(x, y) = iszero(x) && !isnan(y) ? zero(x * log(y)) : x * log(y)
#
# whose `iszero(x)` branch returns a constant, giving `d/dx = 0` at `x == 0`
# instead of the correct `log(y)`. This surfaces through
# `Distributions.gammalogpdf`, which computes `xlogy(shape - 1, x / scale)`,
# so any Gamma log-density differentiated at `shape == 1` gets a wrong
# shape-gradient under Mooncake — a real evaluation point for this package's
# own engine (`as_logdensity`/`logdensity`, `src/engine.jl`), not only for a
# downstream tree that happens to land a parameter there: `parameter_rows`/
# `reconstruct` are generic over any fittable object, so any Gamma-family
# estimated parameter can land on 1.0 at whatever point a sampler or
# gradient check probes. `LogExpFunctionsChainRulesCoreExt` already ships
# correct `ChainRulesCore.rrule`s AND `frule`s for both functions, so
# `@from_chainrules` (Mooncake's default `mode = Mode`, i.e. both directions)
# imports them directly rather than re-deriving the maths.
#
# ComposedDistributions' own `ComposedDistributionsMooncakeExt`
# (ComposedDistributions#99, the fix this generalises) uses the narrower
# `@from_rrule` (reverse only). Widened to `@from_chainrules` here after
# confirming `LogExpFunctionsChainRulesCoreExt` ships an `frule` for both
# functions too: `AutoMooncakeForward` differentiates this package's own
# engine just as validly as reverse mode, and the same `shape == 1` edge
# case reproduces under forward mode with an uncorrected `frule`-derived
# rule (caught by this extension's own AD-fixture scenario, `test/
# ADFixtures`) — there is no reason to leave that gap open when the
# forward rule is already available upstream for free. Worth porting back
# to ComposedDistributions' extension too (a separate PR there).
#
# This is intentional, narrowly-scoped type piracy on functions this package
# does not own, matching the workflow Mooncake's own `@from_rrule`/
# `@from_chainrules` documentation endorses for closing such gaps from a
# downstream package. It should be removed once Mooncake ships its own rule
# (reported upstream, chalk-lab/Mooncake.jl#1241).
module DistributionsInferenceMooncakeExt

using LogExpFunctions: xlogy, xlog1py
using Mooncake: Mooncake

Mooncake.@from_chainrules Mooncake.DefaultCtx Tuple{
    typeof(xlogy), Base.IEEEFloat, Base.IEEEFloat}
Mooncake.@from_chainrules Mooncake.DefaultCtx Tuple{
    typeof(xlog1py), Base.IEEEFloat, Base.IEEEFloat}

end # module
