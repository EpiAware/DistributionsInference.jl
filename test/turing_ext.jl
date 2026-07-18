# DistributionsInference × DynamicPPL: `as_turing(obj, data)` builds a
# DynamicPPL model over a fittable object's estimated parameters, a light
# wrapper on the `as_logdensity` codec. The extension loads when DynamicPPL
# alone is present. These tests prove (a) the model's `~` site names match the
# `VarName`-keyed readback exactly, so a fitted chain reads back through
# `readback`/`readback_draws` unchanged; (b) the model's total log-density
# equals the codec's `logdensity` by construction, for a single site, several
# sites, AND a nonzero `extra_logprior` term; (c) a 0-estimated and a
# 2-parameter (dotted-name) object both round-trip; (d) an estimated row with
# no per-row prior (scored instead through `extra_logprior`) is rejected with
# a clear pointer to the codec path; and (e) the `VarName`-chain empty-chain
# shortcut (`_to_symbol_chain`) guards on the CHAIN's own parameter count, not
# `obj`'s, so both mismatch directions (an empty-estimate `obj` against a
# nonempty chain, and an estimating `obj` against a genuinely empty chain)
# still raise the mismatch rather than one of them silently passing or
# stack-overflowing.

@testsnippet TuringFixture begin
    using DistributionsInference, Distributions

    # A gradient-based sampler (`NUTS`) evaluates `reconstruct` at a
    # `ForwardDiff.Dual`-valued flat vector, so the ESTIMATED field's type
    # must be generic (a concretely `Float64` field, like `ToyFixture`'s
    # `ToyGammaLeaf`, errors under `NUTS`; see `as_turing`'s docstring). A
    # `Distributions.jl` leaf gets this for free from its own parametric
    # type; these toy leaves need the same.
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

    # Two estimated parameters under a DOTTED row name (mirrors a nested
    # parameter path, e.g. a leaf nested under an edge): proves `as_turing`
    # splits the dotted name into DynamicPPL's nested `VarName` segments the
    # same way the readback rebuilds them. Independently-typed fields for the
    # same AD reason as `TuringGammaLeaf` (both are estimated here, and each
    # may carry a different `Dual` tag/perturbation).
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

    # A leaf whose sole estimated row has no per-row prior (scored instead
    # through `extra_logprior`, e.g. an object-dependent/hierarchical term):
    # `as_turing` has no `~` site to sample it from and must reject it.
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
    function DistributionsInference.extra_logprior(::NoPriorLeaf, r, x)
        return -0.5 * r.shape^2
    end
end

@testsnippet ExtraLogpriorFixture begin
    using DistributionsInference, Distributions

    # An object-dependent `extra_logprior` term threaded through `as_turing`:
    # `mu` is the one ESTIMATED row (an ordinary `~` prior); `a`/`b` are FIXED
    # (`prior = nothing`, so `estimated_rows` excludes them from the flat
    # vector and they need no `~` site), but their `extra_logprior` term
    # still depends on the RECONSTRUCTED `mu`, so it is nonzero and
    # non-trivial — mirrors `parameter_rows`'s own `PooledPair` docstring
    # example (a location hyperparameter pooling two fixed members).
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

    function DistributionsInference.extra_logprior(p::PooledPairLeaf, r, x)
        return logpdf(Normal(r.mu, 1.0), r.a) + logpdf(Normal(r.mu, 1.0), r.b)
    end
end

@testitem "as_turing extension loads under DynamicPPL alone" begin
    using DistributionsInference, DynamicPPL
    @test Base.get_extension(DistributionsInference,
        :DistributionsInferenceDynamicPPLExt) !== nothing
end

@testitem "as_turing: model log-density equals the engine's logdensity" setup=[ToyFixture] begin
    using DistributionsInference, Distributions, DynamicPPL

    scale = 1.5
    leaf = ToyGammaLeaf(2.0, scale, LogNormal(log(2.0), 0.2))
    data = [1.5, 2.0, 3.2, 2.8, 1.9]

    model = DistributionsInference.as_turing(leaf, data)
    prob = DistributionsInference.as_logdensity(leaf, data)
    x = [2.3]

    # Conditioning the single `~` site at its readback name scores the same
    # total (prior + `@addlogprob!` likelihood) that `logdensity` sums.
    cm = DynamicPPL.condition(model, @varname(d.shape) => x[1])
    @test DynamicPPL.logjoint(cm, DynamicPPL.VarInfo(cm)) ≈
          DistributionsInference.logdensity(prob, x)
end

@testitem "as_turing: model log-density equals logdensity with a nonzero extra_logprior" setup=[ExtraLogpriorFixture] begin
    using DistributionsInference, Distributions, DynamicPPL

    leaf = PooledPairLeaf(0.2, -0.1, 0.0)
    data = [0.5, -0.3, 1.1]

    model = DistributionsInference.as_turing(leaf, data)
    prob = DistributionsInference.as_logdensity(leaf, data)
    x = [0.7]

    # `extra_logprior` here is nonzero (it scores the reconstructed object's
    # FIXED `a`/`b` fields against a Normal centred on the ESTIMATED `mu`),
    # so this exercises the `@addlogprob! extra_logprior(...)` term the
    # equality guarantee depends on, not just the per-row-prior + likelihood
    # terms the single-site test above already covers.
    rebuilt = DistributionsInference.reconstruct(leaf, x)
    @test DistributionsInference.extra_logprior(leaf, rebuilt, x) != 0.0

    cm = DynamicPPL.condition(model, @varname(d.mu) => x[1])
    @test DynamicPPL.logjoint(cm, DynamicPPL.VarInfo(cm)) ≈
          DistributionsInference.logdensity(prob, x)
end

@testitem "as_turing: model log-density equals logdensity with 2 estimated parameters" setup=[TuringFixture] begin
    using DistributionsInference, Distributions, DynamicPPL

    leaf = TwoParamLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2, 2.8, 1.9]

    model = DistributionsInference.as_turing(leaf, data)
    prob = DistributionsInference.as_logdensity(leaf, data)
    x = [2.3, 1.1]

    # Both `~` sites conditioned at once: the exact-equality guarantee for a
    # multi-site model, not just the single-parameter case above (the
    # NUTS-based round-trip test below only checks recovery, not exactness).
    cm = DynamicPPL.condition(
        model, @varname(d.leaf.shape) => x[1], @varname(d.leaf.scale) => x[2])
    @test DynamicPPL.logjoint(cm, DynamicPPL.VarInfo(cm)) ≈
          DistributionsInference.logdensity(prob, x)
end

@testitem "as_turing round-trip: NUTS chain reads back through readback" setup=[TuringFixture] begin
    using DistributionsInference, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain

    scale = 1.5
    leaf = TuringGammaLeaf(2.0, scale, LogNormal(log(2.0), 0.2))
    data = [1.5, 2.0, 3.2, 2.8, 1.9]

    model = DistributionsInference.as_turing(leaf, data)

    Random.seed!(1)
    chain = sample(model, NUTS(), 200; chain_type = VNChain, progress = false)

    # The single estimated parameter is sampled at the readback's dotted name.
    vns = Set(string.(collect(FlexiChains.parameters(chain))))
    @test "d.shape" in vns

    fitted = DistributionsInference.readback(leaf, chain)
    @test fitted.scale == scale  # the fixed parameter, untouched
    @test fitted.shape > 0

    all_fitted = DistributionsInference.readback_draws(leaf, chain)
    @test length(all_fitted) == 200
    @test mean(f -> f.shape, all_fitted) ≈ fitted.shape
end

@testitem "as_turing acceptance: NUTS recovers the true parameter" setup=[TuringFixture] begin
    using DistributionsInference, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain

    rng = Random.Xoshiro(1)
    true_shape = 3.0
    scale = 1.5
    data = rand(rng, Gamma(true_shape, scale), 500)

    leaf = TuringGammaLeaf(2.0, scale, LogNormal(log(2.0), 0.5))
    model = DistributionsInference.as_turing(leaf, data)

    Random.seed!(2)
    chain = sample(model, NUTS(), 1000; chain_type = VNChain, progress = false)
    fitted = DistributionsInference.readback(leaf, chain)

    prior_mean = mean(LogNormal(log(2.0), 0.5))
    @test abs(fitted.shape - true_shape) < abs(prior_mean - true_shape)
    @test abs(fitted.shape - true_shape) < 0.5
    @test fitted.scale == scale
end

@testitem "as_turing round-trip: 2 estimated parameters with dotted names" setup=[TuringFixture] begin
    using DistributionsInference, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain

    leaf = TwoParamLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2, 2.8, 1.9]
    model = DistributionsInference.as_turing(leaf, data)

    Random.seed!(3)
    chain = sample(model, NUTS(), 200; chain_type = VNChain, progress = false)

    vns = Set(string.(collect(FlexiChains.parameters(chain))))
    @test "d.leaf.shape" in vns
    @test "d.leaf.scale" in vns

    fitted = DistributionsInference.readback(leaf, chain)
    @test fitted.shape > 0
    @test fitted.scale > 0

    all_fitted = DistributionsInference.readback_draws(leaf, chain)
    @test length(all_fitted) == 200

    # A chain read back at the wrong prefix errors rather than silently
    # matching nothing.
    @test_throws ArgumentError DistributionsInference.readback(
        leaf, chain; prefix = :wrong)
end

@testitem "as_turing: a 0-estimated object samples and reads back unchanged" setup=[ToyFixture] begin
    using DistributionsInference, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain

    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2]
    model = DistributionsInference.as_turing(fixed_leaf, data)
    @test DistributionsInference.flat_dimension(fixed_leaf) == 0

    Random.seed!(4)
    chain = sample(model, Prior(), 50; chain_type = VNChain, progress = false)
    @test isempty(FlexiChains.parameters(chain))

    fitted = DistributionsInference.readback(fixed_leaf, chain)
    @test fitted == fixed_leaf

    all_fitted = DistributionsInference.readback_draws(fixed_leaf, chain)
    @test length(all_fitted) == 50
    @test all(==(fixed_leaf), all_fitted)
end

@testitem "as_turing rejects an estimated row with no per-row prior" setup=[TuringFixture] begin
    using DistributionsInference

    leaf = NoPriorLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2]
    @test_throws ArgumentError DistributionsInference.as_turing(leaf, data)
end

@testitem "readback: the VarName empty-chain shortcut guards on the chain, not obj" setup=[
    ToyFixture, TuringFixture] begin
    using DistributionsInference, Distributions, DynamicPPL
    using FlexiChains: FlexiChains, VarName, @varname

    # (a) `obj` estimates NOTHING but the chain carries a parameter: the
    # empty-chain shortcut in `_to_symbol_chain` must guard on the CHAIN
    # being empty, not on `obj`'s estimated rows — otherwise this mismatch
    # would be silently swallowed (reading the template back unchanged
    # instead of raising it).
    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    mismatched_chain = FlexiChains.FlexiChain{VarName}(3, 1,
        Dict{FlexiChains.ParameterOrExtra{<:VarName}, Matrix}(
            FlexiChains.Parameter(@varname(d.shape)) => reshape(
            [1.0, 2.0, 3.0], 3, 1)))
    @test_throws ArgumentError DistributionsInference.readback(
        fixed_leaf, mismatched_chain)

    # (b) `obj` estimates a parameter but the chain is genuinely empty: must
    # raise the ordinary "not found in chain" mismatch (from the core
    # readback) rather than stack-overflowing inside
    # `FlexiChains.map_parameters` — the bug the empty-chain shortcut exists
    # to avoid in the first place.
    leaf = TuringGammaLeaf(2.0, 1.5, LogNormal(log(2.0), 0.2))
    empty_chain = FlexiChains.FlexiChain{VarName}(5, 1,
        Dict{FlexiChains.ParameterOrExtra{<:VarName}, Matrix}())
    @test_throws ArgumentError DistributionsInference.readback(leaf, empty_chain)
end
