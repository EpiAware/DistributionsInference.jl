# DistributionsInference × DynamicPPL: `distribution_to_turing(obj, data)` builds a
# DynamicPPL model over a fittable object's estimated parameters, a light
# wrapper on the `distribution_to_logdensity` codec. The `VarName`-keyed readback lives in
# a separate extension over both DynamicPPL and FlexiChains, so the last items
# here cover that boundary.

@testsnippet TuringFixture begin
    using DistributionsInference, Distributions

    # `NUTS` evaluates `reconstruct` at a `ForwardDiff.Dual`-valued flat
    # vector, so the estimated field's type must be generic; a concretely
    # `Float64` field (like `ToyFixture`'s `ToyGammaLeaf`) errors under `NUTS`.
    struct TuringGammaLeaf{S <: Real}
        shape::S
        scale::Float64
        shape_prior::Distribution
    end

    Distributions.logpdf(d::TuringGammaLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

    function DistributionsInference.parameter_rows(d::TuringGammaLeaf)
        return [
            (name = :shape, value = d.shape,
                prior = d.shape_prior, support = (0.0, Inf)),
            (name = :scale, value = d.scale, prior = nothing,
                support = (0.0, Inf))]
    end

    function DistributionsInference.reconstruct(d::TuringGammaLeaf, x::AbstractVector)
        return TuringGammaLeaf(x[1], d.scale, d.shape_prior)
    end

    # Two estimated parameters under a dotted row name: `distribution_to_turing` must split
    # it into DynamicPPL's nested `VarName` segments the same way the readback
    # rebuilds them. Independently typed for the same AD reason as above.
    struct TwoParamLeaf{S <: Real, C <: Real}
        shape::S
        scale::C
    end

    Distributions.logpdf(d::TwoParamLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

    function DistributionsInference.parameter_rows(d::TwoParamLeaf)
        return [
            (name = Symbol("leaf.shape"), value = d.shape,
                prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
            (name = Symbol("leaf.scale"), value = d.scale,
                prior = LogNormal(log(1.0), 0.2), support = (0.0, Inf))]
    end

    function DistributionsInference.reconstruct(d::TwoParamLeaf, x::AbstractVector)
        return TwoParamLeaf(x[1], x[2])
    end

    # A leaf whose sole estimated row has no per-row prior, scored instead
    # through `extra_logprior`: `distribution_to_turing` has no `~` site for it.
    struct NoPriorLeaf
        shape::Float64
        scale::Float64
    end

    Distributions.logpdf(d::NoPriorLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

    function DistributionsInference.parameter_rows(d::NoPriorLeaf)
        return [(name = :shape, value = d.shape, prior = nothing,
                support = (0.0, Inf)),
            (name = :scale, value = d.scale, prior = nothing,
                support = (0.0, Inf))]
    end
    DistributionsInference.estimated_rows(d::NoPriorLeaf) = [
        DistributionsInference.parameter_rows(d)[1]]
    DistributionsInference.flat_dimension(::NoPriorLeaf) = 1
    function DistributionsInference.reconstruct(d::NoPriorLeaf, x::AbstractVector)
        return NoPriorLeaf(x[1], d.scale)
    end
    function DistributionsInference.extra_logprior(::NoPriorLeaf, r, x, ::Any)
        return -0.5 * r.shape^2
    end
end

@testsnippet ExtraLogpriorFixture begin
    using DistributionsInference, Distributions

    # An object-dependent `extra_logprior` term threaded through `distribution_to_turing`:
    # `mu` is the one estimated row; `a`/`b` are fixed, but their
    # `extra_logprior` term depends on the RECONSTRUCTED `mu`.
    struct PooledPairLeaf
        a::Float64
        b::Float64
        mu::Float64
    end

    Distributions.logpdf(p::PooledPairLeaf, y::Real) = logpdf(Normal(p.mu, 1.0), y)

    function DistributionsInference.parameter_rows(p::PooledPairLeaf)
        return [
            (name = :mu, value = p.mu, prior = Normal(0.0, 1.0),
                support = (-Inf, Inf)),
            (name = :a, value = p.a, prior = nothing, support = (-Inf, Inf)),
            (name = :b, value = p.b, prior = nothing, support = (-Inf, Inf))]
    end

    function DistributionsInference.reconstruct(p::PooledPairLeaf, x::AbstractVector)
        return PooledPairLeaf(p.a, p.b, x[1])
    end

    function DistributionsInference.extra_logprior(p::PooledPairLeaf, r, x, ::Any)
        return logpdf(Normal(r.mu, 1.0), r.a) + logpdf(Normal(r.mu, 1.0), r.b)
    end
end

@testitem "distribution_to_turing extension loads under DynamicPPL alone" begin
    using DistributionsInference, DynamicPPL
    @test Base.get_extension(DistributionsInference,
        :DistributionsInferenceDynamicPPLExt) !== nothing
end

@testitem "distribution_to_turing: model log-density equals the engine's logdensity" setup=[ToyFixture] begin
    using DistributionsInference, Distributions, DynamicPPL

    scale = 1.5
    leaf = ToyGammaLeaf(2.0, scale, LogNormal(log(2.0), 0.2))
    data = [1.5, 2.0, 3.2, 2.8, 1.9]

    model = DistributionsInference.distribution_to_turing(leaf, data)
    prob = DistributionsInference.distribution_to_logdensity(leaf, data)
    x = [2.3]

    # Conditioning the `~` site at its readback name scores the same total
    # that `logdensity` sums.
    cm = DynamicPPL.condition(model, @varname(d.shape) => x[1])
    @test DynamicPPL.logjoint(cm, DynamicPPL.VarInfo(cm)) ≈
          DistributionsInference.logdensity(prob, x)
end

@testitem "distribution_to_turing: model log-density equals logdensity with a nonzero extra_logprior" setup=[ExtraLogpriorFixture] begin
    using DistributionsInference, Distributions, DynamicPPL

    leaf = PooledPairLeaf(0.2, -0.1, 0.0)
    data = [0.5, -0.3, 1.1]

    model = DistributionsInference.distribution_to_turing(leaf, data)
    prob = DistributionsInference.distribution_to_logdensity(leaf, data)
    x = [0.7]

    # A nonzero `extra_logprior` exercises the `@addlogprob!` term the
    # equality guarantee depends on, not just the per-row-prior + likelihood
    # terms the single-site test above covers.
    rebuilt = DistributionsInference.reconstruct(leaf, x)
    @test DistributionsInference.extra_logprior(leaf, rebuilt, x, nothing) != 0.0

    cm = DynamicPPL.condition(model, @varname(d.mu) => x[1])
    @test DynamicPPL.logjoint(cm, DynamicPPL.VarInfo(cm)) ≈
          DistributionsInference.logdensity(prob, x)
end

@testitem "distribution_to_turing: model log-density equals logdensity with 2 estimated parameters" setup=[TuringFixture] begin
    using DistributionsInference, Distributions, DynamicPPL, ForwardDiff
    using LogDensityProblems

    leaf = TwoParamLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2, 2.8, 1.9]

    model = DistributionsInference.distribution_to_turing(leaf, data)
    prob = DistributionsInference.distribution_to_logdensity(leaf, data)
    x = [2.3, 1.1]

    # Both `~` sites conditioned at once: exact equality for a multi-site
    # model, which the NUTS round-trip below does not check.
    cm = DynamicPPL.condition(
        model, @varname(d.leaf.shape) => x[1], @varname(d.leaf.scale) => x[2])
    @test DynamicPPL.logjoint(cm, DynamicPPL.VarInfo(cm)) ≈
          DistributionsInference.logdensity(prob, x)

    # The same equality through `LogDensityProblems`, the interface a
    # gradient-based sampler reaches, and then at the DERIVATIVE level. The AD
    # matrix in `test/ADFixtures` takes both targets' references from
    # ForwardDiff separately, so nothing there would notice the turing model
    # drifting from the codec (a dropped `~` site, say); this comparison is
    # what pins the two gradients together.
    ldf = DynamicPPL.LogDensityFunction(model)
    @test LogDensityProblems.logdensity(ldf, x) ≈
          DistributionsInference.logdensity(prob, x)
    @test ForwardDiff.gradient(θ -> LogDensityProblems.logdensity(ldf, θ), x) ≈
          ForwardDiff.gradient(
        θ -> DistributionsInference.logdensity(prob, θ), x)
end

@testitem "distribution_to_turing round-trip: NUTS chain reads back through point_estimate" setup=[TuringFixture] begin
    using DistributionsInference, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain

    scale = 1.5
    leaf = TuringGammaLeaf(2.0, scale, LogNormal(log(2.0), 0.2))
    data = [1.5, 2.0, 3.2, 2.8, 1.9]

    model = DistributionsInference.distribution_to_turing(leaf, data)

    Random.seed!(1)
    chain = sample(model, NUTS(), 200; chain_type = VNChain, progress = false)

    # The single estimated parameter is sampled at the readback's dotted name.
    vns = Set(string.(collect(FlexiChains.parameters(chain))))
    @test "d.shape" in vns

    fitted = DistributionsInference.point_estimate(leaf, chain)
    @test fitted.scale == scale
    @test fitted.shape > 0

    all_fitted = DistributionsInference.distribution_draws(leaf, chain)
    @test length(all_fitted) == 200
    @test mean(f -> f.shape, all_fitted) ≈ fitted.shape

    # `distribution_params` also dispatches on a VarName-keyed chain, through
    # the same `_to_symbol_chain` conversion `point_estimate` uses.
    nt = DistributionsInference.distribution_params(leaf, chain)
    @test keys(nt) == (:shape,)
    @test nt.shape == fitted.shape
end

@testitem "distribution_to_turing acceptance: NUTS recovers the true parameter" setup=[TuringFixture] begin
    using DistributionsInference, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain

    rng = Random.Xoshiro(1)
    true_shape = 3.0
    scale = 1.5
    data = rand(rng, Gamma(true_shape, scale), 500)

    leaf = TuringGammaLeaf(2.0, scale, LogNormal(log(2.0), 0.5))
    model = DistributionsInference.distribution_to_turing(leaf, data)

    Random.seed!(2)
    chain = sample(model, NUTS(), 1000; chain_type = VNChain, progress = false)
    fitted = DistributionsInference.point_estimate(leaf, chain)

    prior_mean = mean(LogNormal(log(2.0), 0.5))
    @test abs(fitted.shape - true_shape) < abs(prior_mean - true_shape)
    @test abs(fitted.shape - true_shape) < 0.5
    @test fitted.scale == scale
end

@testitem "distribution_to_turing round-trip: 2 estimated parameters with dotted names" setup=[TuringFixture] begin
    using DistributionsInference, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain

    leaf = TwoParamLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2, 2.8, 1.9]
    model = DistributionsInference.distribution_to_turing(leaf, data)

    Random.seed!(3)
    chain = sample(model, NUTS(), 200; chain_type = VNChain, progress = false)

    vns = Set(string.(collect(FlexiChains.parameters(chain))))
    @test "d.leaf.shape" in vns
    @test "d.leaf.scale" in vns

    fitted = DistributionsInference.point_estimate(leaf, chain)
    @test fitted.shape > 0
    @test fitted.scale > 0

    all_fitted = DistributionsInference.distribution_draws(leaf, chain)
    @test length(all_fitted) == 200

    # A chain read back at the wrong prefix errors rather than silently
    # matching nothing.
    @test_throws ArgumentError DistributionsInference.point_estimate(
        leaf, chain; prefix = :wrong)
end

@testitem "distribution_to_turing: a 0-estimated object samples and reads back unchanged" setup=[ToyFixture] begin
    using DistributionsInference, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain

    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2]
    model = DistributionsInference.distribution_to_turing(fixed_leaf, data)
    @test DistributionsInference.flat_dimension(fixed_leaf) == 0

    Random.seed!(4)
    chain = sample(model, Prior(), 50; chain_type = VNChain, progress = false)
    @test isempty(FlexiChains.parameters(chain))

    fitted = DistributionsInference.point_estimate(fixed_leaf, chain)
    @test fitted == fixed_leaf

    all_fitted = DistributionsInference.distribution_draws(fixed_leaf, chain)
    @test length(all_fitted) == 50
    @test all(==(fixed_leaf), all_fitted)
end

@testitem "distribution_to_turing: the concrete-field guard fires at the turing call site too" begin
    using DistributionsInference, Distributions, DynamicPPL, ForwardDiff
    using LogDensityProblems

    # DI#48's guard runs at BOTH call sites: `logdensity` (covered in
    # `test/protocol.jl`) and the turing model built here. Only the second
    # sees a `Vector{Real}` whose elements are tracer numbers while the
    # container's own eltype says nothing, which is exactly why the check is
    # per element rather than on the container. Without the guard a
    # gradient-based sampler on this model dies with an opaque `MethodError`
    # from `TuringConcreteLeaf`'s own inner constructor.
    struct TuringConcreteLeaf
        shape::Float64
        scale::Float64
    end

    function Distributions.logpdf(d::TuringConcreteLeaf, y::Real)
        return logpdf(Gamma(d.shape, d.scale), y)
    end

    function DistributionsInference.parameter_rows(d::TuringConcreteLeaf)
        return [
            (name = :shape, value = d.shape,
                prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
            (name = :scale, value = d.scale, prior = nothing,
                support = (0.0, Inf))]
    end

    function DistributionsInference.reconstruct(
            d::TuringConcreteLeaf, x::AbstractVector)
        return TuringConcreteLeaf(x[1], d.scale)
    end

    leaf = TuringConcreteLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2]
    model = DistributionsInference.distribution_to_turing(leaf, data)
    ldf = DynamicPPL.LogDensityFunction(model)

    # Ordinary (non-AD) evaluation is unaffected: the model still scores the
    # same total the codec does at the same point.
    x = [2.5]
    prob = DistributionsInference.distribution_to_logdensity(leaf, data)
    @test LogDensityProblems.logdensity(ldf, x) ≈
          DistributionsInference.logdensity(prob, x)

    # A gradient threads `Dual`s through the model's `Vector{Real}`, where
    # the guard raises its named `ArgumentError`.
    err = try
        ForwardDiff.gradient(
            θ -> LogDensityProblems.logdensity(ldf, θ), x)
        nothing
    catch caught
        caught
    end
    @test err isa ArgumentError
    @test occursin("shape", err.msg)
    @test occursin("TuringConcreteLeaf", err.msg)
    @test occursin("generically typed", err.msg)
end

@testitem "distribution_to_turing rejects an estimated row with no per-row prior" setup=[TuringFixture] begin
    using DistributionsInference

    leaf = NoPriorLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2]
    @test_throws ArgumentError DistributionsInference.distribution_to_turing(leaf, data)
end

@testitem "point_estimate: the VarName empty-chain shortcut guards on the chain, not obj" setup=[
    ToyFixture, TuringFixture] begin
    using DistributionsInference, Distributions, DynamicPPL
    using FlexiChains: FlexiChains, VarName, @varname

    # (a) `obj` estimates nothing but the chain carries a parameter: the
    # `_to_symbol_chain` shortcut must guard on the chain being empty, not on
    # `obj`'s estimated rows, or the mismatch is silently swallowed.
    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    mismatched_chain = FlexiChains.FlexiChain{VarName}(3, 1,
        Dict{FlexiChains.ParameterOrExtra{<:VarName}, Matrix}(
            FlexiChains.Parameter(@varname(d.shape)) => reshape(
            [1.0, 2.0, 3.0], 3, 1)))
    @test_throws ArgumentError DistributionsInference.point_estimate(
        fixed_leaf, mismatched_chain)

    # (b) `obj` estimates a parameter but the chain is genuinely empty: must
    # raise the ordinary "not found in chain" mismatch rather than
    # stack-overflowing inside `FlexiChains.map_parameters`.
    leaf = TuringGammaLeaf(2.0, 1.5, LogNormal(log(2.0), 0.2))
    empty_chain = FlexiChains.FlexiChain{VarName}(5, 1,
        Dict{FlexiChains.ParameterOrExtra{<:VarName}, Matrix}())
    @test_throws ArgumentError DistributionsInference.point_estimate(leaf, empty_chain)
end

@testitem "the DynamicPPL x FlexiChains readback extension loads" begin
    using DistributionsInference, DynamicPPL
    using FlexiChains: FlexiChains

    @test Base.get_extension(
        DistributionsInference,
        :DistributionsInferenceDynamicPPLFlexiChainsExt) !== nothing
end

@testitem "distribution_to_turing works with just `using DynamicPPL`" begin
    using DistributionsInference

    # `FlexiChains` is a hard dependency, so it is always in the session once
    # `DistributionsInference` is; a caller sampling with Turing needs to
    # `using DynamicPPL` (to trigger the readback's `VarName` dispatch) but
    # never has to `using FlexiChains` explicitly. Sibling items load
    # `DynamicPPL` here, so this needs a fresh process to check the caller's
    # `using` list alone is enough.
    script = """
    using DistributionsInference, Distributions, DynamicPPL

    Base.get_extension(
        DistributionsInference,
        :DistributionsInferenceDynamicPPLFlexiChainsExt) !== nothing ||
        error("the DynamicPPL x FlexiChains extension did not load")

    struct AloneLeaf{S <: Real}
        shape::S
        scale::Float64
    end

    Distributions.logpdf(d::AloneLeaf, y::Real) = logpdf(
        Gamma(d.shape, d.scale), y)

    function DistributionsInference.parameter_rows(d::AloneLeaf)
        return [(name = :shape, value = d.shape,
                prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
            (name = :scale, value = d.scale, prior = nothing,
                support = (0.0, Inf))]
    end

    function DistributionsInference.reconstruct(
            d::AloneLeaf, x::AbstractVector)
        return AloneLeaf(x[1], d.scale)
    end

    leaf = AloneLeaf(2.0, 1.5)
    model = DistributionsInference.distribution_to_turing(leaf, [1.5, 2.0, 3.2])
    vi = DynamicPPL.VarInfo(model)
    isfinite(DynamicPPL.logjoint(model, vi)) ||
        error("the model's log-joint is not finite")
    print("as-turing-alone-ok")
    """

    out = IOBuffer()
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project())
           --startup-file=no -e $script`
    ok = success(pipeline(cmd; stdout = out, stderr = out))
    output = String(take!(out))
    ok || println(output)
    @test ok
    @test occursin("as-turing-alone-ok", output)
end
