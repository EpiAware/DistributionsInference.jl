# The inference-engine contract (#94): `distribution_to_chain` dispatching on
# an engine's own concrete type, the `template`/`observations`/`flat_priors`
# accessors it is built from, `TuringEngine` (the `DynamicPPL` consumer) and
# `MetropolisEngine` (the `AdvancedMH` + `Bijectors` consumer), and the parity
# claim the package sells: two engines on the same problem give comparable
# answers.

@testitem "FitLogDensity accessors: template/observations/flat_priors match direct field access" setup=[
    ToyFixture] begin
    using DistributionsInference, Distributions

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    data = [1.5, 2.0, 3.2]
    prob = DistributionsInference.distribution_to_logdensity(leaf, data)

    @test DistributionsInference.template(prob) === leaf
    @test DistributionsInference.template(prob) === prob.obj
    @test DistributionsInference.observations(prob) === data
    @test DistributionsInference.observations(prob) === prob.data
    @test DistributionsInference.flat_priors(prob) == [leaf.shape_prior]
    @test DistributionsInference.flat_priors(prob) === prob.flat_priors
end

@testitem "draws_to_chain is public and keys raw draws by dotted name" setup=[
    ToyFixture] begin
    using DistributionsInference, Distributions
    using FlexiChains: FlexiChains

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    draws = reshape([2.1, 2.4, 2.0, 2.6], 1, 4)
    chain = DistributionsInference.draws_to_chain(leaf, draws)

    @test chain isa FlexiChains.FlexiChain
    @test FlexiChains.has_parameter(chain, :shape)
    @test FlexiChains.niters(chain) == 4
    @test FlexiChains.nchains(chain) == 1
end

@testitem "distribution_to_chain: TuringEngine round-trips through inference_to_distributions/point_estimate" setup=[
    TuringFixture] begin
    using DistributionsInference, Distributions, DynamicPPL, Turing, Random

    scale = 1.5
    leaf = TuringGammaLeaf(2.0, scale, LogNormal(log(2.0), 0.2))
    data = [1.5, 2.0, 3.2, 2.8, 1.9]

    Random.seed!(11)
    chain = distribution_to_chain(
        leaf, data, TuringEngine(NUTS(); nsamples = 200, progress = false))

    dists = inference_to_distributions(leaf, chain)
    @test length(dists) == 200
    @test all(d -> d isa TuringGammaLeaf, dists)

    fitted = point_estimate(leaf, chain)
    @test fitted.scale == scale
    @test fitted.shape > 0
end

@testitem "distribution_to_chain: TuringEngine nchains > 1 pools independent chains" setup=[
    TuringFixture] begin
    using DistributionsInference, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains

    leaf = TuringGammaLeaf(2.0, 1.5, LogNormal(log(2.0), 0.2))
    data = [1.5, 2.0, 3.2, 2.8, 1.9]

    Random.seed!(12)
    chain = distribution_to_chain(leaf, data,
        TuringEngine(NUTS(); nsamples = 100, nchains = 3, progress = false))

    @test FlexiChains.nchains(chain) == 3
    @test FlexiChains.niters(chain) == 100
    @test length(inference_to_distributions(leaf, chain)) == 300
end

@testitem "distribution_to_chain: MetropolisEngine round-trips and recovers the true parameter" setup=[
    TuringFixture] begin
    using DistributionsInference, Distributions, AdvancedMH, Bijectors, Random

    rng = Random.Xoshiro(7)
    true_shape = 3.0
    scale = 1.5
    data = rand(rng, Gamma(true_shape, scale), 400)

    leaf = TuringGammaLeaf(2.0, scale, LogNormal(log(2.0), 0.5))
    chain = distribution_to_chain(leaf, data,
        MetropolisEngine(; nsamples = 4000, burnin = 1000);
        rng = Random.Xoshiro(4))

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

@testitem "distribution_to_chain: MetropolisEngine nchains > 1 pools independent chains" setup=[
    TuringFixture] begin
    using DistributionsInference, Distributions, AdvancedMH, Bijectors, Random
    using FlexiChains: FlexiChains

    leaf = TuringGammaLeaf(2.0, 1.5, LogNormal(log(2.0), 0.2))
    data = [1.5, 2.0, 3.2, 2.8, 1.9]

    chain = distribution_to_chain(leaf, data,
        MetropolisEngine(; nsamples = 500, burnin = 100, nchains = 2);
        rng = Random.Xoshiro(5))

    @test FlexiChains.nchains(chain) == 2
    @test FlexiChains.niters(chain) == 400
    @test length(inference_to_distributions(leaf, chain)) == 800
end

@testitem "distribution_to_chain: MetropolisEngine on an object estimating nothing samples and reads back unchanged" setup=[
    ToyFixture] begin
    using DistributionsInference, Distributions, AdvancedMH, Bijectors

    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2]
    @test DistributionsInference.flat_dimension(fixed_leaf) == 0

    chain = distribution_to_chain(
        fixed_leaf, data, MetropolisEngine(; nsamples = 50))
    all_fitted = inference_to_distributions(fixed_leaf, chain)
    @test length(all_fitted) == 50
    @test all(==(fixed_leaf), all_fitted)
end

@testitem "distribution_to_chain: TuringEngine and MetropolisEngine agree on the same problem" setup=[
    TuringFixture] begin
    using DistributionsInference, Distributions, DynamicPPL, Turing,
          AdvancedMH, Bijectors, Random

    # The parity claim the package sells: two engines fitting the same
    # problem give comparable answers, not merely runnable ones.
    rng = Random.Xoshiro(3)
    true_shape = 2.5
    scale = 1.0
    data = rand(rng, Gamma(true_shape, scale), 400)
    leaf = TuringGammaLeaf(2.0, scale, LogNormal(log(2.0), 0.5))

    Random.seed!(21)
    turing_chain = distribution_to_chain(
        leaf, data, TuringEngine(NUTS(); nsamples = 1000, progress = false))
    metropolis_chain = distribution_to_chain(leaf, data,
        MetropolisEngine(; nsamples = 6000, burnin = 1000);
        rng = Random.Xoshiro(6))

    turing_fit = point_estimate(leaf, turing_chain)
    metropolis_fit = point_estimate(leaf, metropolis_chain)

    @test turing_fit.scale == metropolis_fit.scale == scale
    @test isapprox(turing_fit.shape, metropolis_fit.shape; atol = 0.5)
    @test isapprox(turing_fit.shape, true_shape; atol = 0.5)
    @test isapprox(metropolis_fit.shape, true_shape; atol = 0.5)
end
