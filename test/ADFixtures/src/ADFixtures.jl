# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# AD-fixture registry implementing the EpiAwarePackageTools `ADRegistry`
# contract: scenarios (each with a ForwardDiff reference), a backend list, and
# broken/skip bookkeeping. The shared harness, driven from `test/ad/setup.jl`,
# consumes these, and the AD-backends docs page renders its support table and
# benchmark from the same set.
#
# The scenarios cover the four public differentiable paths, so a
# backend-specific regression in any of them reds the matrix rather than only
# the engine's own hot path (DI#33): the engine's `logdensity` over several
# fit-protocol objects; `to_constrained` composed with it, the
# unconstrained-scale target a sampler differentiates (identical to
# `-logdensity_to_objective(prob)`), covering the `Bijectors` extension's
# log, identity and logit links; both paths over a `ComposedDistributions`
# tree; and `distribution_to_turing`'s model through `LogDensityProblems`, the
# `DynamicPPL` extension's `Vector{Real}` path into `reconstruct`. `point_estimate`
# and friends are absent: they map an already-sampled chain onto a
# reconstructed object, so there is no parameter vector to differentiate
# (their coverage is in `test/readback.jl`).
#
# `Mooncake` is deliberately not a dependency here even though `backends()`
# lists both its configurations: this registry loads `ComposedDistributions`,
# and `DistributionsInferenceMooncakeExt` and `ComposedDistributionsMooncakeExt`
# define the same `LogExpFunctions.xlogy`/`xlog1py` rules (DI ported CD#99's
# fix), which Julia refuses as a method overwrite during precompilation. The
# `ADTypes` backend entries need no import here, and `test/ad/setup.jl` does
# the `using Mooncake` at test time, where the duplicate rules are only a load
# warning. Drop this once the duplication is resolved upstream (DI#73).
module ADFixtures

using ADTypes: AutoForwardDiff, AutoReverseDiff, AutoMooncake,
               AutoMooncakeForward, AutoEnzyme
using DifferentiationInterface: DifferentiationInterface, Constant
import DifferentiationInterfaceTest as DIT
import ForwardDiff, ReverseDiff, Enzyme
using DistributionsInference
using Distributions: Distributions, Beta, Gamma, LogNormal, Normal, logpdf,
                     truncated
# Loading `Bijectors`, `ComposedDistributions` and `DynamicPPL` here is what
# puts `DistributionsInferenceBijectorsExt`,
# `DistributionsInferenceComposedDistributionsExt` and
# `DistributionsInferenceDynamicPPLExt` in the session, so the scenarios below
# differentiate through the extensions themselves rather than through
# core-only code paths.
using Bijectors: Bijectors
using ComposedDistributions: ComposedDistributions, compose, uncertain
using DynamicPPL: DynamicPPL
using LogDensityProblems: LogDensityProblems

export scenarios, backends, broken_scenario_names,
       backend_broken_scenarios, backend_skip_scenarios

# --- Fit-protocol fixtures -------------------------------------------------
#
# Every estimated field is generically typed (`shape::S`, not
# `shape::Float64`): a concrete field cannot hold a tracer number and is
# rejected up front by the engine's own guard.

# One estimated parameter: a Gamma leaf with the shape estimated and the
# scale fixed.
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

# TWO estimated parameters, both on the positive line: the shape and the
# scale of the same Gamma. The two rows share a family but not a value, so a
# backend that silently collapses a multi-parameter gradient onto one
# coordinate (or transposes the two) disagrees with the ForwardDiff
# reference.
struct GammaShapeScaleFit{S <: Real, C <: Real}
    shape::S
    scale::C
end

function Distributions.logpdf(d::GammaShapeScaleFit, y::Real)
    return logpdf(Gamma(d.shape, d.scale), y)
end

function DistributionsInference.parameter_rows(d::GammaShapeScaleFit)
    return [
        (name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.3), support = (0.0, Inf)),
        (name = :scale, value = d.scale,
            prior = LogNormal(log(1.5), 0.3), support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(
        d::GammaShapeScaleFit, x::AbstractVector)
    return GammaShapeScaleFit(x[1], x[2])
end

# Two estimated parameters under DIFFERENT support constraints: a location on
# the whole line (an unconstrained `Normal` prior, whose bijector is the
# identity) and a positive scale under a truncated-`Normal` prior (the family
# `default_prior` itself produces, whose bijector is a log link). This is the
# one scenario family whose unconstrained transform mixes an identity row
# with a linked row.
struct NormalFit{M <: Real, S <: Real}
    mu::M
    sigma::S
end

Distributions.logpdf(d::NormalFit, y::Real) = logpdf(Normal(d.mu, d.sigma), y)

function DistributionsInference.parameter_rows(d::NormalFit)
    return [
        (name = :mu, value = d.mu, prior = Normal(0.0, 2.0),
            support = (-Inf, Inf)),
        (name = :sigma, value = d.sigma,
            prior = truncated(Normal(1.0, 1.0); lower = 0.0),
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::NormalFit, x::AbstractVector)
    return NormalFit(x[1], x[2])
end

# A `[0, 1]`-support estimated parameter: a Beta likelihood in its mean /
# concentration parameterisation, so `p` lives on the unit interval under a
# `Beta` prior (a logit link) while `kappa` stays positive under a
# `LogNormal` prior (a log link). Both the likelihood's own `logbeta` terms
# and the logit transform are paths no Gamma or Normal scenario reaches.
struct UnitIntervalFit{P <: Real, K <: Real}
    p::P
    kappa::K
end

function Distributions.logpdf(d::UnitIntervalFit, y::Real)
    return logpdf(Beta(d.p * d.kappa, (1 - d.p) * d.kappa), y)
end

function DistributionsInference.parameter_rows(d::UnitIntervalFit)
    return [
        (name = :p, value = d.p, prior = Beta(2.0, 2.0),
            support = (0.0, 1.0)),
        (name = :kappa, value = d.kappa,
            prior = LogNormal(log(5.0), 0.3), support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(
        d::UnitIntervalFit, x::AbstractVector)
    return UnitIntervalFit(x[1], x[2])
end

# --- Scenario construction -------------------------------------------------

# ForwardDiff reference gradient for a scenario function.
#
# Every scenario's reference comes from this one backend, so a green matrix is
# evidence about BACKENDS, not a correctness proof for the target: a wrong but
# differentiable target (a dropped `logjac` or prior term) would have all six
# agree on the same wrong function. The values are pinned by the ordinary
# suite — `test/bijectors_ext.jl`, `test/engine.jl`, `test/turing_ext.jl`.
function _reference(f, θ, contexts)
    return DifferentiationInterface.gradient(
        f, AutoForwardDiff(), θ, contexts...)
end

# One `DIT.Scenario`, with the ForwardDiff reference attached only when the
# caller asked for it (the docs page renders the scenario list without paying
# for a reference gradient it never compares against).
function _scenario(f, θ, contexts, name::String; with_reference::Bool)
    prep_args = (; x = θ, contexts = contexts)
    res1 = with_reference ? _reference(f, θ, contexts) : nothing
    return res1 === nothing ?
           DIT.Scenario{:gradient, :out}(f, θ, contexts...;
        prep_args = prep_args, name = name) :
           DIT.Scenario{:gradient, :out}(f, θ, contexts...;
        res1 = res1, prep_args = prep_args, name = name)
end

# The engine's own hot path: the gradient of `logdensity(prob, x)`, flowing
# through the per-row prior sum, `reconstruct`, and the data likelihood.
_engine_target(x, prob) = DistributionsInference.logdensity(prob, x)

# The unconstrained-scale target a sampler or optimiser differentiates:
# `to_constrained` (the `Bijectors` extension) composed with the core
# `logdensity`, plus the transform's log-Jacobian. Identical to
# `-logdensity_to_objective(prob)(z)`, so covering this covers both.
function _unconstrained_target(z, prob)
    x, logjac = DistributionsInference.to_constrained(prob, z)
    return DistributionsInference.logdensity(prob, x) + logjac
end

# The turing model `distribution_to_turing` builds (the `DynamicPPL` extension), reached
# through the same `LogDensityProblems` interface a gradient-based sampler
# uses. The model threads the draw through an abstractly-typed
# `Vector{Real}` before `reconstruct`, a path no other scenario takes.
#
# Its reference gradient is ForwardDiff over this target, like every other
# scenario, so the matrix only checks that the six backends agree on the
# turing model. What ties the model to `logdensity(prob, x)` is
# `test/turing_ext.jl`, whose "model log-density equals logdensity" items
# compare both the value and the ForwardDiff gradient of the two.
_turing_target(x, ldf) = LogDensityProblems.logdensity(ldf, x)

# Gamma-distributed observations, shared by the Gamma-family scenarios.
const _GAMMA_DATA = [0.5, 1.2, 2.5, 3.8, 5.1]
# Real-line observations for the Normal-family scenarios.
const _NORMAL_DATA = [-0.8, 0.3, 1.4, 0.9, -0.2]
# Unit-interval observations for the Beta-family scenarios.
const _UNIT_DATA = [0.21, 0.44, 0.63, 0.37, 0.55]
# Per-edge records for the two-leaf composed tree.
const _COMPOSED_DATA = [[0.5, 2.0], [1.0, 3.0], [0.8, 1.7]]

# A two-leaf `ComposedDistributions` tree with BOTH leaves uncertain, so the
# extension's `parameter_rows`/`reconstruct` translation is differentiated
# through at a genuine multi-parameter flat vector rather than a single
# coordinate.
function _composed_tree()
    return compose((
        onset_admit = uncertain(
            Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
        admit_death = uncertain(
            Gamma(3.0, 1.5); shape = LogNormal(log(3.0), 0.2))))
end

"""
    scenarios(; with_reference = false, category = :marginal)

The AD gradient scenarios. Each is a `DIT.Scenario{:gradient, :out}` whose
`res1` carries a ForwardDiff reference when `with_reference = true`. Use
`category` to select a scenario group (e.g. `:marginal` vs `:latent`); every
scenario here sits in the single `:marginal` group.
"""
function scenarios(; with_reference::Bool = false, category::Symbol = :marginal)
    out = DIT.Scenario{:gradient, :out}[]
    scen(f, θ, contexts, name) = push!(out,
        _scenario(f, θ, contexts, name; with_reference = with_reference))

    # --- The engine, one estimated parameter ------------------------------
    one_param = DistributionsInference.distribution_to_logdensity(
        GammaFit(2.0, 1.0), _GAMMA_DATA)
    scen(_engine_target, [2.0], (Constant(one_param),),
        "fit-protocol engine logdensity")

    # The `xlogy` edge case (DI#7, CD#99): `shape == 1.0` routes a nonzero
    # cotangent into `LogExpFunctions.xlogy`'s `iszero(x)` branch inside
    # `Distributions.gammalogpdf`, where Mooncake derives `0` instead of
    # `log(y)` (chalk-lab/Mooncake.jl#1241). `DistributionsInferenceMooncakeExt`
    # imports the `ChainRulesCore` rules as Mooncake primitives to fix this; if
    # those ever stop working, quarantine this scenario via
    # `broken_scenario_names`/`backend_broken_scenarios` below.
    scen(_engine_target, [1.0], (Constant(one_param),),
        "fit-protocol engine logdensity (shape at 1.0, xlogy edge case)")

    # --- The engine, several estimated parameters -------------------------
    two_gamma = DistributionsInference.distribution_to_logdensity(
        GammaShapeScaleFit(2.0, 1.5), _GAMMA_DATA)
    scen(_engine_target, [2.2, 1.3], (Constant(two_gamma),),
        "engine logdensity (Gamma shape and scale estimated)")

    normal_prob = DistributionsInference.distribution_to_logdensity(
        NormalFit(0.0, 1.0), _NORMAL_DATA)
    scen(_engine_target, [0.35, 1.2], (Constant(normal_prob),),
        "engine logdensity (Normal location and scale estimated)")

    unit_prob = DistributionsInference.distribution_to_logdensity(
        UnitIntervalFit(0.4, 5.0), _UNIT_DATA)
    scen(_engine_target, [0.45, 6.0], (Constant(unit_prob),),
        "engine logdensity (unit-interval mean and concentration estimated)")

    # --- The `Bijectors` extension's unconstrained transform --------------
    # The same `xlogy` edge case as above, reached through `to_constrained`
    # instead of directly: `z = 0.0` maps to `shape = exp(0) = 1.0`, so a
    # Mooncake-class `xlogy` regression inside the bijectors path is caught
    # here rather than only on the engine's own hot path (DI#33).
    scen(_unconstrained_target, [0.0], (Constant(one_param),),
        "unconstrained logdensity (shape at 1.0, xlogy edge case)")

    # An identity link (the unconstrained `Normal` prior on `mu`) beside a
    # log link (the positive-truncated prior on `sigma`).
    scen(_unconstrained_target, [0.35, 0.2], (Constant(normal_prob),),
        "unconstrained logdensity (identity and log links)")

    # A logit link (the `Beta` prior on the unit-interval `p`) beside a log
    # link (the `LogNormal` prior on `kappa`).
    scen(_unconstrained_target, [-0.2, 0.15], (Constant(unit_prob),),
        "unconstrained logdensity (logit and log links)")

    # --- The `ComposedDistributions` extension ----------------------------
    composed_prob = DistributionsInference.distribution_to_logdensity(
        _composed_tree(), _COMPOSED_DATA)
    scen(_engine_target, [2.2, 3.1], (Constant(composed_prob),),
        "engine logdensity (ComposedDistributions tree)")

    scen(_unconstrained_target, [0.7, 1.1], (Constant(composed_prob),),
        "unconstrained logdensity (ComposedDistributions tree)")

    # --- The `DynamicPPL` extension's turing model ------------------------
    turing_ldf = DynamicPPL.LogDensityFunction(
        DistributionsInference.distribution_to_turing(
        GammaShapeScaleFit(2.0, 1.5), _GAMMA_DATA))
    scen(_turing_target, [2.2, 1.3], (Constant(turing_ldf),),
        "distribution_to_turing model logdensity (Gamma shape and scale estimated)")

    return out
end

"""
    backends()

The AD backends to test, as `(; name, backend)` named tuples. One entry per
testitem in `test/ad/scenarios.jl`.
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
