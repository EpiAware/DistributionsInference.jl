# The posterior-output API: `inference_to_distribution`/`inference_to_distributions`
# (and their `inference_to_dist`/`inference_to_dists` aliases), the equal-weight
# `MixtureModel`/vectorised/plug-in trio replacing `point_estimate`/
# `distribution_draws` for a caller who wants a `Distribution` out. Additive:
# `point_estimate`/`distribution_params`/`distribution_draws` are covered
# separately in `test/readback.jl` and are unaffected by this file.

@testsnippet GammaLeafFixture begin
    using DistributionsInference, Distributions

    # `reconstruct` returns a `Gamma` directly (unlike `ToyFixture`'s
    # `ToyGammaLeaf`, which reconstructs to its own wrapper type) so this
    # fixture can exercise the MixtureModel/plug-in paths that need a
    # `Distribution` out.
    struct GammaLeaf
        shape::Float64
        scale::Float64
        shape_prior::Union{Nothing, Distribution}
    end

    GammaLeaf(shape::Real, scale::Real) = GammaLeaf(shape, scale, nothing)

    function DistributionsInference.parameter_rows(d::GammaLeaf)
        return [
            (name = :shape, value = d.shape, prior = d.shape_prior,
                support = (0.0, Inf)),
            (name = :scale, value = d.scale, prior = nothing,
                support = (0.0, Inf))]
    end

    function DistributionsInference.reconstruct(d::GammaLeaf, x::AbstractVector)
        n = DistributionsInference.flat_dimension(d)
        n == 0 && return Gamma(d.shape, d.scale)
        return Gamma(x[1], d.scale)
    end
end

@testitem "inference_to_distributions: agrees with distribution_draws on the same input" setup=[
    GammaLeafFixture] begin
    using FlexiChains: FlexiChains

    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    chain = FlexiChains.FlexiChain{Symbol}(
        6, 1, Dict(FlexiChains.Parameter(:shape) => reshape(values, 6, 1)))

    new_dists = DistributionsInference.inference_to_distributions(leaf, chain)
    old_dists = DistributionsInference.distribution_draws(leaf, chain)

    @test length(new_dists) == length(old_dists) == 6
    @test all(d -> d isa Gamma, new_dists)
    @test [d.α for d in new_dists] == [d.α for d in old_dists] == values
end

@testitem "inference_to_distribution(obj, chain, summary): agrees with point_estimate" setup=[
    GammaLeafFixture] begin
    using FlexiChains: FlexiChains
    using Statistics: mean, median

    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [1.0, 2.0, 3.0, 4.0]
    chain = FlexiChains.FlexiChain{Symbol}(
        4, 1, Dict(FlexiChains.Parameter(:shape) => reshape(values, 4, 1)))

    plugin_mean = DistributionsInference.inference_to_distribution(
        leaf, chain, mean)
    old_mean = DistributionsInference.point_estimate(leaf, chain)
    @test plugin_mean isa Gamma
    @test plugin_mean.α ≈ old_mean.α

    plugin_median = DistributionsInference.inference_to_distribution(
        leaf, chain, median)
    old_median = DistributionsInference.point_estimate(
        leaf, chain; summary = median)
    @test plugin_median.α ≈ old_median.α

    # Not the same distribution as the Monte Carlo mixture: the plug-in
    # collapses the draws before reconstructing. With `scale = 1`,
    # `mean(Gamma(shape, 1)) == shape` is linear in `shape`, so the two
    # means coincide here even though the distributions differ — the `pdf`
    # at a point does not, which is the honest way to show they differ.
    mm = DistributionsInference.inference_to_distribution(leaf, chain)
    @test mean(mm) ≈ mean(plugin_mean)  # means coincide (linear in shape)
    # `Gamma(mean(shape draws), scale) != mean(Gamma.(shape draws, scale))`:
    # the two distributions are not the same, even though their means agree.
    @test pdf(mm, 1.0) != pdf(plugin_mean, 1.0)
end

@testitem "inference_to_distribution: the equal-weight MixtureModel over selected draws" setup=[
    GammaLeafFixture] begin
    using FlexiChains: FlexiChains
    using Statistics: mean

    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [1.0, 2.0, 3.0, 4.0]
    chain = FlexiChains.FlexiChain{Symbol}(
        4, 1, Dict(FlexiChains.Parameter(:shape) => reshape(values, 4, 1)))

    mm = DistributionsInference.inference_to_distribution(leaf, chain)
    @test mm isa MixtureModel
    @test Distributions.ncomponents(mm) == 4
    @test mean(mm) ≈ mean(mean(Gamma(v, 1.0)) for v in values)

    # `draws` restricts the selection before mixing.
    mm_subset = DistributionsInference.inference_to_distribution(
        leaf, chain; draws = 2:3)
    @test Distributions.ncomponents(mm_subset) == 2
    @test mean(mm_subset) ≈ mean([mean(Gamma(2.0, 1.0)), mean(Gamma(3.0, 1.0))])
end

@testitem "aliases: inference_to_dist/inference_to_dists are the same functions" begin
    using DistributionsInference

    @test DistributionsInference.inference_to_dist ===
          DistributionsInference.inference_to_distribution
    @test DistributionsInference.inference_to_dists ===
          DistributionsInference.inference_to_distributions
end

@testitem "the draws keyword: nothing, range, index vector, Integer" setup=[
    GammaLeafFixture] begin
    using FlexiChains: FlexiChains
    using Random

    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    chain = FlexiChains.FlexiChain{Symbol}(
        5, 1, Dict(FlexiChains.Parameter(:shape) => reshape(values, 5, 1)))

    # `nothing`: every draw.
    @test [d.α for d in DistributionsInference.inference_to_distributions(leaf, chain)] ==
          values

    # An `AbstractRange` of pooled indices.
    ranged = DistributionsInference.inference_to_distributions(
        leaf, chain; draws = 2:4)
    @test [d.α for d in ranged] == values[2:4]

    # An index vector.
    idx = DistributionsInference.inference_to_distributions(
        leaf, chain; draws = [1, 3, 5])
    @test [d.α for d in idx] == values[[1, 3, 5]]

    # An `Integer`: a random subsample of that size, reproducible via `rng`.
    sub1 = DistributionsInference.inference_to_distributions(
        leaf, chain; draws = 3, rng = Random.Xoshiro(7))
    sub2 = DistributionsInference.inference_to_distributions(
        leaf, chain; draws = 3, rng = Random.Xoshiro(7))
    @test length(sub1) == 3
    @test [d.α for d in sub1] == [d.α for d in sub2]
    @test all(d -> d.α in values, sub1)

    # Sampling more draws than exist errors.
    @test_throws ArgumentError DistributionsInference.inference_to_distributions(
        leaf, chain; draws = 6)

    # An unsupported `draws` type errors clearly rather than falling through
    # to a bare `MethodError`.
    @test_throws ArgumentError DistributionsInference.inference_to_distributions(
        leaf, chain; draws = "nope")
end

@testitem "the Integer draws form spans every chain, not just the first" setup=[
    GammaLeafFixture] begin
    using FlexiChains: FlexiChains
    using Random

    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    # 3 chains x 4 iterations, each chain's values in a disjoint range so
    # "did this span chains" is checkable by value.
    chain1 = [1.0, 2.0, 3.0, 4.0]
    chain2 = [100.0, 200.0, 300.0, 400.0]
    chain3 = [10000.0, 20000.0, 30000.0, 40000.0]
    mat = reshape(vcat(chain1, chain2, chain3), 4, 3)
    chain = FlexiChains.FlexiChain{Symbol}(
        4, 3, Dict(FlexiChains.Parameter(:shape) => mat))

    # A large-enough random sample must, with overwhelming probability,
    # include a draw from more than one chain. Use a fixed seed and a
    # generous sample size (9 of 12 pooled draws) to make this deterministic
    # in practice rather than flaky.
    sampled = DistributionsInference.inference_to_distributions(
        leaf, chain; draws = 9, rng = Random.Xoshiro(3))
    shapes = [d.α for d in sampled]
    spans_chain1 = any(v -> v in chain1, shapes)
    spans_chain2 = any(v -> v in chain2, shapes)
    spans_chain3 = any(v -> v in chain3, shapes)
    @test count([spans_chain1, spans_chain2, spans_chain3]) >= 2
end

@testitem "the chain-major pooling trap: draws=1:n on a multi-chain run is chain 1 only" setup=[
    GammaLeafFixture] begin
    using FlexiChains: FlexiChains

    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    chain1 = [1.0, 2.0, 3.0]
    chain2 = [10.0, 20.0, 30.0]
    mat = reshape(vcat(chain1, chain2), 3, 2)
    chain = FlexiChains.FlexiChain{Symbol}(
        3, 2, Dict(FlexiChains.Parameter(:shape) => mat))

    # This is the documented trap: an explicit range selector operates on
    # the pooled (chain-major) index range, so `1:3` on a 2-chain run is
    # chain 1's own 3 draws, not "the first 3 of a subsample spanning both".
    looks_like_a_subsample = DistributionsInference.inference_to_distributions(
        leaf, chain; draws = 1:3)
    @test [d.α for d in looks_like_a_subsample] == chain1

    # The `Integer` form is the honest "give me some draws" request: it
    # samples from the whole pooled range, not just chain 1.
    explicit_full_pool = DistributionsInference.inference_to_distributions(
        leaf, chain; draws = 4:6)
    @test [d.α for d in explicit_full_pool] == chain2
end

@testitem "inference_to_distribution: a non-Distribution reconstruct errors, naming the type" setup=[
    ToyFixture] begin
    using FlexiChains: FlexiChains

    # `ToyGammaLeaf` reconstructs to itself, not a `Distribution` — exactly
    # the toy fittable type this package's own docstrings use.
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [2.1, 2.4, 2.0, 2.6]
    chain = FlexiChains.FlexiChain{Symbol}(
        4, 1, Dict(FlexiChains.Parameter(:shape) => reshape(values, 4, 1)))

    # The vectorised plural form is unaffected: it does not require a
    # `Distribution` element type.
    dists = DistributionsInference.inference_to_distributions(leaf, chain)
    @test length(dists) == 4
    @test all(d -> d isa ToyGammaLeaf, dists)

    # Both `inference_to_distribution` forms require it and name the type
    # actually reached.
    err_mixture = try
        DistributionsInference.inference_to_distribution(leaf, chain)
        nothing
    catch e
        e
    end
    @test err_mixture isa ArgumentError
    @test occursin("ToyGammaLeaf", err_mixture.msg)
    @test occursin("Distribution", err_mixture.msg)

    err_plugin = try
        using Statistics: mean
        DistributionsInference.inference_to_distribution(leaf, chain, mean)
        nothing
    catch e
        e
    end
    @test err_plugin isa ArgumentError
    @test occursin("ToyGammaLeaf", err_plugin.msg)
end

@testitem "dim == 0 with ndraws > 0" setup=[GammaLeafFixture] begin
    using FlexiChains: FlexiChains
    using Statistics: mean

    fixed_leaf = GammaLeaf(2.0, 1.0)  # no prior: nothing estimated
    @test DistributionsInference.flat_dimension(fixed_leaf) == 0

    chain = FlexiChains.FlexiChain{Symbol}(
        5, 1, Dict{FlexiChains.ParameterOrExtra{<:Symbol}, Matrix}())

    dists = DistributionsInference.inference_to_distributions(
        fixed_leaf, chain)
    @test length(dists) == 5
    @test all(==(Gamma(2.0, 1.0)), dists)

    mm = DistributionsInference.inference_to_distribution(fixed_leaf, chain)
    @test mm isa MixtureModel
    @test mean(mm) ≈ mean(Gamma(2.0, 1.0))

    plugin = DistributionsInference.inference_to_distribution(
        fixed_leaf, chain, mean)
    @test plugin == Gamma(2.0, 1.0)
end

@testitem "raw draws with no chain: a dim x ndraws matrix and nchains" setup=[
    GammaLeafFixture] begin
    leaf = GammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))

    # 6 pooled draws, split into 2 chains of 3, chain-major.
    raw = reshape([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 1, 6)

    dists = DistributionsInference.inference_to_distributions(
        leaf, raw; nchains = 2)
    @test [d.α for d in dists] == [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]

    # `draws` still resolves over the pooled (chain-major) range.
    chain2_only = DistributionsInference.inference_to_distributions(
        leaf, raw; nchains = 2, draws = 4:6)
    @test [d.α for d in chain2_only] == [10.0, 20.0, 30.0]

    mm = DistributionsInference.inference_to_distribution(leaf, raw; nchains = 2)
    @test mm isa MixtureModel
    @test Distributions.ncomponents(mm) == 6

    plugin = DistributionsInference.inference_to_distribution(
        leaf, raw, sum; nchains = 2)
    @test plugin.α ≈ sum([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])

    # A vector-of-vectors raw form is accepted too (dim = 1 here).
    raw_vov = [[v] for v in [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]]
    dists_vov = DistributionsInference.inference_to_distributions(
        leaf, raw_vov; nchains = 2)
    @test [d.α for d in dists_vov] == [d.α for d in dists]

    # A draw count not divisible by nchains errors, naming both numbers.
    err = try
        DistributionsInference.inference_to_distributions(leaf, raw; nchains = 4)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("6", err.msg)
    @test occursin("4", err.msg)
end

@testitem "readback acceptance: an AdvancedMH sampler round-trips through inference_to_distribution" setup=[
    GammaLeafFixture] begin
    using AdvancedMH
    using FlexiChains: FlexiChains
    using LogDensityProblems
    using LinearAlgebra: I
    using Random
    using Statistics: mean

    rng = Random.Xoshiro(1)
    true_shape = 3.0
    scale = 1.5
    data = rand(rng, Gamma(true_shape, scale), 500)

    leaf = GammaLeaf(2.0, scale, LogNormal(log(2.0), 0.5))
    prob = DistributionsInference.distribution_to_logdensity(leaf, data)

    model = AdvancedMH.DensityModel() do x
        any(<=(0), x) ? -Inf : LogDensityProblems.logdensity(prob, x)
    end
    spl = RWMH(MvNormal(zeros(1), 0.05^2 * I))
    transitions = sample(
        rng, model, spl, 5000; param_names = ["shape"], progress = false)
    draws = [t.params for t in transitions][2001:end]

    mm = DistributionsInference.inference_to_distribution(leaf, draws)
    @test mm isa MixtureModel
    @test abs(mean(mm) - mean(Gamma(true_shape, scale))) < 1.0

    plugin = DistributionsInference.inference_to_distribution(leaf, draws, mean)
    @test abs(plugin.α - true_shape) < 0.5
end
