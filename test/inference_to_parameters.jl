# The raw-draws access verbs (#90): `inference_to_parameters` (exact draws,
# keyed by dotted name — a Tables.jl-compatible column table) and
# `inference_to_parameter_distribution` (the joint posterior fitted as a
# `MvNormal`, on the unconstrained scale, for Markov melding). Additive:
# every other posterior-output verb is unaffected.

@testitem "inference_to_parameters: names and lengths match estimated_rows and the selected draw count" setup=[
    TuringFixture] begin
    using FlexiChains: FlexiChains

    leaf = TwoParamLeaf(2.0, 1.0)
    n = 5
    shape_draws = [2.0, 2.1, 1.9, 2.2, 2.05]
    scale_draws = [1.0, 1.1, 0.9, 1.2, 1.05]
    chain = FlexiChains.FlexiChain{Symbol}(n,
        1,
        Dict(
            FlexiChains.Parameter(Symbol("leaf.shape")) => reshape(
                shape_draws, n, 1),
            FlexiChains.Parameter(Symbol("leaf.scale")) => reshape(
                scale_draws, n, 1)))

    nt = DistributionsInference.inference_to_parameters(leaf, chain)
    @test keys(nt) == (Symbol("leaf.shape"), Symbol("leaf.scale"))
    @test length(nt[Symbol("leaf.shape")]) == length(nt[Symbol("leaf.scale")]) ==
          n

    sub = DistributionsInference.inference_to_parameters(
        leaf, chain; draws = 2:4)
    @test length(sub[Symbol("leaf.shape")]) ==
          length(sub[Symbol("leaf.scale")]) == 3
end

@testitem "inference_to_parameters: exact values, not just shapes" setup=[
    GammaLeafFixture] begin
    using FlexiChains: FlexiChains

    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values_ = [1.1, 2.2, 3.3, 4.4]
    chain = FlexiChains.FlexiChain{Symbol}(
        4, 1, Dict(FlexiChains.Parameter(:shape) => reshape(values_, 4, 1)))

    nt = DistributionsInference.inference_to_parameters(leaf, chain)
    @test keys(nt) == (:shape,)
    @test nt.shape == values_

    sub = DistributionsInference.inference_to_parameters(
        leaf, chain; draws = [1, 3])
    @test sub.shape == values_[[1, 3]]
end

@testitem "inference_to_parameters: every draws form agrees with inference_to_distributions' selection" setup=[
    GammaLeafFixture] begin
    using FlexiChains: FlexiChains
    using Random

    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values_ = [10.0, 20.0, 30.0, 40.0, 50.0]
    chain = FlexiChains.FlexiChain{Symbol}(
        5, 1, Dict(FlexiChains.Parameter(:shape) => reshape(values_, 5, 1)))

    for draws in (nothing, 2:4, [1, 3, 5], 3)
        nt = DistributionsInference.inference_to_parameters(
            leaf, chain; draws = draws, rng = Random.Xoshiro(11))
        dists = DistributionsInference.inference_to_distributions(
            leaf, chain; draws = draws, rng = Random.Xoshiro(11))
        @test collect(nt.shape) == [d.α for d in dists]
    end
end

@testitem "inference_to_parameters: a multi-chain run pools chain-major, agreeing with the other verbs" setup=[
    GammaLeafFixture] begin
    using FlexiChains: FlexiChains

    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    chain1 = [1.0, 2.0, 3.0]
    chain2 = [10.0, 20.0, 30.0]
    mat = reshape(vcat(chain1, chain2), 3, 2)
    chain = FlexiChains.FlexiChain{Symbol}(
        3, 2, Dict(FlexiChains.Parameter(:shape) => mat))

    # `draws = 1:3` on a 2-chain run is chain 1's own draws — the same
    # chain-major pooling trap `inference_to_distributions` documents.
    @test DistributionsInference.inference_to_parameters(
        leaf, chain; draws = 1:3).shape == chain1
    @test DistributionsInference.inference_to_parameters(
        leaf, chain; draws = 4:6).shape == chain2
    @test DistributionsInference.inference_to_parameters(leaf, chain).shape ==
          vcat(chain1, chain2)
end

@testitem "inference_to_parameters: an object estimating nothing returns NamedTuple()" setup=[
    GammaLeafFixture] begin
    using FlexiChains: FlexiChains

    fixed_leaf = GammaLeaf(2.0, 1.0)  # no prior: nothing estimated
    @test DistributionsInference.flat_dimension(fixed_leaf) == 0

    chain = FlexiChains.FlexiChain{Symbol}(
        5, 1, Dict{FlexiChains.ParameterOrExtra{<:Symbol}, Matrix}())
    @test DistributionsInference.inference_to_parameters(
        fixed_leaf, chain) == NamedTuple()
end

@testitem "inference_to_parameters: raw draws with no chain, and nchains" setup=[
    GammaLeafFixture] begin
    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    raw = reshape([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 1, 6)

    nt = DistributionsInference.inference_to_parameters(leaf, raw; nchains = 2)
    @test nt.shape == [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]

    chain2_only = DistributionsInference.inference_to_parameters(
        leaf, raw; nchains = 2, draws = 4:6)
    @test chain2_only.shape == [10.0, 20.0, 30.0]
end

@testitem "inference_to_parameters: duplicate dotted names are refused" begin
    using DistributionsInference, Distributions
    using FlexiChains: FlexiChains

    # A bare row vector is its own `parameter_rows`/fittable object (see
    # `protocol.jl`), so two rows sharing a `name` is a direct way to trigger
    # the protocol-invariant violation this guards.
    rows = [
        (name = :shape, value = 1.0, prior = LogNormal(0.0, 0.2),
            support = (0.0, Inf)),
        (name = :shape, value = 2.0, prior = LogNormal(0.0, 0.2),
            support = (0.0, Inf))]
    chain = FlexiChains.FlexiChain{Symbol}(
        3, 1, Dict(FlexiChains.Parameter(:shape) => reshape(
            [1.0, 2.0, 3.0], 3, 1)))

    err = try
        DistributionsInference.inference_to_parameters(rows, chain)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("inference_to_parameters", err.msg)
    @test occursin("shape", err.msg)
end

@testitem "inference_to_parameters: the NamedTuple satisfies the Tables.jl column-table contract structurally" setup=[
    GammaLeafFixture] begin
    using FlexiChains: FlexiChains

    # Neither Tables.jl nor DataFrames is a test dependency of this package
    # (absent from `test/Project.toml`), so the Tables.jl claim in the
    # docstring is checked structurally rather than through the actual
    # package: a `NamedTuple` of equal-length `AbstractVector`s already
    # satisfies `Tables.istable`'s column-table contract by definition, with
    # no wrapper type and no dependency this package adds.
    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values_ = [1.0, 2.0, 3.0, 4.0]
    chain = FlexiChains.FlexiChain{Symbol}(
        4, 1, Dict(FlexiChains.Parameter(:shape) => reshape(values_, 4, 1)))

    nt = DistributionsInference.inference_to_parameters(leaf, chain)
    @test nt isa NamedTuple
    @test all(v -> v isa AbstractVector, values(nt))
    lens = length.(values(nt))
    @test isempty(lens) || all(==(first(lens)), lens)
end

@testitem "inference_to_parameter_distribution: fits on the unconstrained scale, not the constrained one" setup=[
    GammaLeafFixture] begin
    using Bijectors
    using FlexiChains: FlexiChains
    using Statistics: mean

    # A positive-support parameter, deliberately skewed (one large draw), so
    # the constrained- and unconstrained-scale means diverge sharply: this is
    # the test that catches the whole class of scale error.
    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values_ = [1.0, 2.0, 3.0, 4.0, 5.0, 20.0]
    chain = FlexiChains.FlexiChain{Symbol}(
        6, 1, Dict(FlexiChains.Parameter(:shape) => reshape(values_, 6, 1)))

    mvn = DistributionsInference.inference_to_parameter_distribution(
        leaf, chain)
    @test mvn isa Distributions.MvNormal

    @test mean(mvn)[1] ≈ mean(log.(values_))
    @test !(mean(mvn)[1] ≈ mean(values_))
end

@testitem "inference_to_parameter_distribution: captures correlation between parameters" setup=[
    TuringFixture] begin
    using Bijectors
    using FlexiChains: FlexiChains
    using Statistics: cov

    leaf = TwoParamLeaf(2.0, 1.0)
    # Deterministic and strongly, but NOT perfectly, correlated on the
    # unconstrained (log) scale: `to_unconstrained` for a LogNormal-prior row
    # is a bare `log`, with no location/scale shift, so the covariance of
    # these log values is exactly what the fitted `MvNormal` should
    # reproduce. The offsets matter — an exactly collinear second row gives a
    # rank-deficient covariance that no Gaussian can be fitted to, which is
    # its own refusal and is covered separately below.
    log_shape = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    log_scale = 2.0 .* log_shape .+ [0.1, -0.05, 0.08, -0.12, 0.03, -0.07]
    shape_draws = exp.(log_shape)
    scale_draws = exp.(log_scale)
    n = length(log_shape)
    chain = FlexiChains.FlexiChain{Symbol}(n,
        1,
        Dict(
            FlexiChains.Parameter(Symbol("leaf.shape")) => reshape(
                shape_draws, n, 1),
            FlexiChains.Parameter(Symbol("leaf.scale")) => reshape(
                scale_draws, n, 1)))

    mvn = DistributionsInference.inference_to_parameter_distribution(
        leaf, chain)
    Sigma = cov(mvn)
    expected = cov(hcat(log_shape, log_scale))
    @test Sigma ≈ expected
    # Materially non-zero off-diagonal: the correlation this function exists
    # to keep, not lost to a set of independent marginals.
    @test abs(Sigma[1, 2]) > 1.0
end

@testitem "inference_to_parameter_distribution: collinear parameters are refused" setup=[
    TuringFixture] begin
    using Bijectors
    using FlexiChains: FlexiChains

    leaf = TwoParamLeaf(2.0, 1.0)
    # Exactly collinear on the unconstrained scale: the second row is a
    # deterministic multiple of the first, so the covariance is rank 1 at any
    # draw count. Without a check this reaches `MvNormal` and dies inside
    # PDMats' Cholesky with a `PosDefException` naming neither this function
    # nor the cause.
    log_shape = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    log_scale = 2.0 .* log_shape
    n = length(log_shape)
    chain = FlexiChains.FlexiChain{Symbol}(n,
        1,
        Dict(
            FlexiChains.Parameter(Symbol("leaf.shape")) => reshape(
                exp.(log_shape), n, 1),
            FlexiChains.Parameter(Symbol("leaf.scale")) => reshape(
                exp.(log_scale), n, 1)))

    err = try
        DistributionsInference.inference_to_parameter_distribution(
            leaf, chain)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("not positive definite", err.msg)
    @test occursin("collinear", err.msg)
end

@testitem "inference_to_parameter_distribution: a multi-chain run pools chain-major" setup=[
    GammaLeafFixture] begin
    using Bijectors
    using FlexiChains: FlexiChains
    using Statistics: mean

    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    chain1 = [1.0, 2.0, 3.0, 4.0]
    chain2 = [10.0, 20.0, 30.0, 40.0]
    mat = reshape(vcat(chain1, chain2), 4, 2)
    chain = FlexiChains.FlexiChain{Symbol}(
        4, 2, Dict(FlexiChains.Parameter(:shape) => mat))

    mvn_all = DistributionsInference.inference_to_parameter_distribution(
        leaf, chain)
    @test mean(mvn_all)[1] ≈ mean(log.(vcat(chain1, chain2)))

    # The explicit pooled range is chain 1 only — the same chain-major
    # pooling trap every other verb in this family documents.
    mvn_chain1_only = DistributionsInference.inference_to_parameter_distribution(
        leaf, chain; draws = 1:4)
    @test mean(mvn_chain1_only)[1] ≈ mean(log.(chain1))
end

@testitem "inference_to_parameter_distribution: refuses a draw count at or below the parameter dimension" setup=[
    TuringFixture] begin
    using Bijectors
    using FlexiChains: FlexiChains

    leaf = TwoParamLeaf(2.0, 1.0)  # 2 estimated parameters
    shape_draws = [2.0, 2.1]
    scale_draws = [1.0, 1.1]
    chain = FlexiChains.FlexiChain{Symbol}(2,
        1,
        Dict(
            FlexiChains.Parameter(Symbol("leaf.shape")) => reshape(
                shape_draws, 2, 1),
            FlexiChains.Parameter(Symbol("leaf.scale")) => reshape(
                scale_draws, 2, 1)))

    # n == dim: still singular (a 2-draw sample covariance in 2 dimensions
    # has no residual degrees of freedom), so this must be refused too, not
    # just n < dim.
    err = try
        DistributionsInference.inference_to_parameter_distribution(
            leaf, chain)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("2 draw", err.msg)
    @test occursin("2 parameter", err.msg)
end

@testitem "inference_to_parameter_distribution: an empty draws selection is refused" setup=[
    GammaLeafFixture] begin
    using Bijectors
    using FlexiChains: FlexiChains

    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    chain = FlexiChains.FlexiChain{Symbol}(
        4, 1, Dict(
            FlexiChains.Parameter(:shape) => reshape(
            [1.0, 2.0, 3.0, 4.0], 4, 1)))

    err = try
        DistributionsInference.inference_to_parameter_distribution(
            leaf, chain; draws = 0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("selected draw", err.msg)
end

@testitem "inference_to_parameter_distribution: an object estimating nothing is refused" setup=[
    GammaLeafFixture] begin
    using Bijectors
    using FlexiChains: FlexiChains

    fixed_leaf = GammaLeaf(2.0, 1.0)  # no prior: nothing estimated
    chain = FlexiChains.FlexiChain{Symbol}(
        5, 1, Dict{FlexiChains.ParameterOrExtra{<:Symbol}, Matrix}())

    err = try
        DistributionsInference.inference_to_parameter_distribution(
            fixed_leaf, chain)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("estimates nothing", err.msg)
end

@testitem "inference_to_parameter_distribution: raw draws with no chain, and nchains" setup=[
    GammaLeafFixture] begin
    using Bijectors
    using Statistics: mean

    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    raw = reshape([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 1, 6)

    mvn = DistributionsInference.inference_to_parameter_distribution(
        leaf, raw; nchains = 2)
    @test mvn isa Distributions.MvNormal
    @test mean(mvn)[1] ≈
          mean(log.([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]))
end

# The no-Bijectors-loaded error path is covered alongside every other
# extension-backed stub in `test/DistributionsInference.jl`'s "an
# extension-backed stub names the package to load" item, which already runs
# a fresh process for exactly this.
