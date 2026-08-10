# DistributionsInference x AdvancedMH x Bijectors: `distribution_to_advancedmh`
# (#94) — the sampling verb alongside `distribution_to_turing`'s chain-
# returning form (tested in `turing_ext.jl`) — and the parity claim the
# package sells: two engines fitting the same problem give comparable
# answers.

@testitem "distribution_to_advancedmh round-trips and recovers the true parameter" setup=[
    TuringFixture] begin
    using DistributionsInference, Distributions, AdvancedMH, Bijectors, Random
    using LinearAlgebra: I

    rng = Random.Xoshiro(7)
    true_shape = 3.0
    scale = 1.5
    data = rand(rng, Gamma(true_shape, scale), 400)

    leaf = TuringGammaLeaf(2.0, scale, LogNormal(log(2.0), 0.5))
    dim = DistributionsInference.flat_dimension(leaf)
    sampler = RWMH(MvNormal(zeros(dim), 0.1^2 * I))
    chain = distribution_to_advancedmh(leaf, data, sampler, 4000;
        burnin = 1000, rng = Random.Xoshiro(4))

    dists = inference_to_distributions(leaf, chain)
    @test length(dists) == 3000
    @test all(d -> d isa TuringGammaLeaf, dists)

    fitted = point_estimate(leaf, chain)
    @test fitted.scale == scale

    # Closer to the true shape than the prior mean, with a loose tolerance:
    # this is a smoke test, not a calibration study (matches the acceptance
    # style already used for `distribution_to_turing`/hand-rolled Metropolis).
    prior_mean = mean(LogNormal(log(2.0), 0.5))
    @test abs(fitted.shape - true_shape) < abs(prior_mean - true_shape)
    @test abs(fitted.shape - true_shape) < 1.0
end

@testitem "distribution_to_advancedmh: nchains > 1 pools independent chains" setup=[
    TuringFixture] begin
    using DistributionsInference, Distributions, AdvancedMH, Bijectors, Random
    using LinearAlgebra: I
    using FlexiChains: FlexiChains

    leaf = TuringGammaLeaf(2.0, 1.5, LogNormal(log(2.0), 0.2))
    data = [1.5, 2.0, 3.2, 2.8, 1.9]
    dim = DistributionsInference.flat_dimension(leaf)
    sampler = RWMH(MvNormal(zeros(dim), 0.1^2 * I))

    chain = distribution_to_advancedmh(leaf, data, sampler, 500;
        burnin = 100, nchains = 2, rng = Random.Xoshiro(5))

    @test FlexiChains.nchains(chain) == 2
    @test FlexiChains.niters(chain) == 400
    @test length(inference_to_distributions(leaf, chain)) == 800
end

@testitem "distribution_to_advancedmh: a bad burnin is rejected by name" setup=[
    TuringFixture] begin
    using DistributionsInference, Distributions, AdvancedMH, Bijectors, Random
    using LinearAlgebra: I

    leaf = TuringGammaLeaf(2.0, 1.5, LogNormal(log(2.0), 0.2))
    data = [1.5, 2.0, 3.2]
    sampler = RWMH(MvNormal(zeros(1), 0.1^2 * I))

    thrown = try
        distribution_to_advancedmh(leaf, data, sampler, 100; burnin = 100)
        nothing
    catch caught
        caught
    end
    @test thrown isa ArgumentError
    @test occursin("burnin", thrown.msg)
end

@testitem "distribution_to_turing and distribution_to_advancedmh agree on the same problem" setup=[
    TuringFixture] begin
    using DistributionsInference, Distributions, DynamicPPL, Turing,
          AdvancedMH, Bijectors, Random
    using LinearAlgebra: I

    # The parity claim the package sells: two engines fitting the same
    # problem give comparable answers, not merely runnable ones.
    rng = Random.Xoshiro(3)
    true_shape = 2.5
    scale = 1.0
    data = rand(rng, Gamma(true_shape, scale), 400)
    leaf = TuringGammaLeaf(2.0, scale, LogNormal(log(2.0), 0.5))
    dim = DistributionsInference.flat_dimension(leaf)

    Random.seed!(21)
    turing_chain = distribution_to_turing(leaf, data, NUTS(), 1000;
        progress = false)
    sampler = RWMH(MvNormal(zeros(dim), 0.1^2 * I))
    advancedmh_chain = distribution_to_advancedmh(leaf, data, sampler, 6000;
        burnin = 1000, rng = Random.Xoshiro(6))

    turing_fit = point_estimate(leaf, turing_chain)
    advancedmh_fit = point_estimate(leaf, advancedmh_chain)

    @test turing_fit.scale == advancedmh_fit.scale == scale
    @test isapprox(turing_fit.shape, advancedmh_fit.shape; atol = 0.5)
    @test isapprox(turing_fit.shape, true_shape; atol = 0.5)
    @test isapprox(advancedmh_fit.shape, true_shape; atol = 0.5)
end
