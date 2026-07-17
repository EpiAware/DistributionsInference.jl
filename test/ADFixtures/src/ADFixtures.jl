# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Minimal AD-fixture registry implementing the EpiAwarePackageTools `ADRegistry`
# contract: scenarios (each with a ForwardDiff reference), a backend list, and
# broken/skip bookkeeping. The shared harness (driven from `test/ad/setup.jl`)
# consumes these. Replace the placeholder scenario with the package's own
# differentiable log densities and add the backends it supports.
#
# If the package's log densities use EpiAwareADTools' AD-safe hooks
# (`cdf_ad_safe`, `primal`, ...) to stay differentiable, add scenarios here that
# exercise those paths, so the per-backend matrix covers them. See
# https://github.com/EpiAware/EpiAwareADTools.jl.
module ADFixtures

using ADTypes: AutoForwardDiff, AutoReverseDiff, AutoMooncake,
               AutoMooncakeForward, AutoEnzyme
using DifferentiationInterface: DifferentiationInterface, Constant
import DifferentiationInterfaceTest as DIT
import ForwardDiff, ReverseDiff, Enzyme, Mooncake
using DistributionsInference
using Distributions: Distributions, Gamma, LogNormal, logpdf

export scenarios, backends, broken_scenario_names,
       backend_broken_scenarios, backend_skip_scenarios

# A minimal fit-protocol object whose engine log-density the scenarios
# differentiate: a Gamma leaf with the shape estimated (LogNormal prior)
# and the scale fixed, mirroring the test suite's toy.
struct GammaFit{T <: Real}
    shape::T
    scale::T
end

Distributions.logpdf(d::GammaFit, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::GammaFit)
    return [
        (name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::GammaFit, x::AbstractVector)
    return GammaFit(x[1], oftype(x[1], d.scale))
end

# ForwardDiff reference gradient for a scenario function.
function _reference(f, θ, contexts)
    return DifferentiationInterface.gradient(
        f, AutoForwardDiff(), θ, contexts...)
end

"""
    scenarios(; with_reference = false, category = :marginal)

The AD gradient scenarios. Each is a `DIT.Scenario{:gradient, :out}` whose
`res1` carries a ForwardDiff reference when `with_reference = true`. Replace the
placeholder with the package's own differentiable log densities; group them by
`category` if the package distinguishes scenario groups (e.g. `:marginal` vs
`:latent`).
"""
function scenarios(; with_reference::Bool = false, category::Symbol = :marginal)
    out = DIT.Scenario{:gradient, :out}[]

    # The engine's own hot path: the gradient of `logdensity(prob, θ)` for a
    # fit-protocol object with one estimated parameter, flowing through the
    # per-row prior sum, `reconstruct`, and the data likelihood.
    leaf = GammaFit(2.0, 1.0)
    prob = DistributionsInference.as_logdensity(
        leaf, [0.5, 1.2, 2.5, 3.8, 5.1])
    f = (θ, p) -> DistributionsInference.logdensity(p, θ)
    θ = [2.0]
    contexts = (Constant(prob),)
    prep_args = (; x = θ, contexts = contexts)
    res1 = with_reference ? _reference(f, θ, contexts) : nothing
    push!(out,
        res1 === nothing ?
        DIT.Scenario{:gradient, :out}(f, θ, contexts...;
            prep_args = prep_args,
            name = "fit-protocol engine logdensity") :
        DIT.Scenario{:gradient, :out}(f, θ, contexts...;
            res1 = res1, prep_args = prep_args,
            name = "fit-protocol engine logdensity"))
    return out
end

"""
    backends()

The AD backends to test, as `(; name, backend)` named tuples. Seeded to match
every backend `test/ad/scenarios.jl` emits a testitem for, so a fresh package
passes its AD suite out of the box; trim to the subset the package actually
supports.
"""
function backends()
    return [
        (name = "ForwardDiff", backend = AutoForwardDiff()),
        (name = "ReverseDiff (tape)", backend = AutoReverseDiff(compile = false)),
        (name = "Enzyme forward",
            backend = AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Forward))),
        (name = "Enzyme reverse",
            backend = AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))),
        (name = "Mooncake reverse", backend = AutoMooncake(config = nothing)),
        (name = "Mooncake forward", backend = AutoMooncakeForward())
    ]
end

"Scenario names broken on every backend."
broken_scenario_names() = String[]

"Per-backend broken scenario names (`Dict{String, Set{String}}`)."
backend_broken_scenarios() = Dict{String, Set{String}}()

"Per-backend scenario names too unstable to run at all."
backend_skip_scenarios() = Dict{String, Set{String}}()

end # module ADFixtures
