# The PPL-neutral log-density engine: `as_logdensity`/`logdensity` over a
# fit-protocol object, the direct `LogDensityProblems` implementation (hard
# dep, no glue extension), and the acceptance criterion for #2 — a hand-rolled
# toy protocol object fits end-to-end via a generic LDP sampler.

@testitem "engine: as_logdensity/logdensity evaluate the estimated posterior" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    data = [1.5, 2.0, 3.2, 2.8, 1.9, 4.1, 2.5, 3.0]
    prob = DistributionsInference.as_logdensity(leaf, data)

    @test DistributionsInference.flat_dimension(leaf) == 1
    x = [2.5]
    expected = logpdf(LogNormal(log(2.0), 0.2), 2.5) +
               sum(y -> logpdf(Gamma(2.5, leaf.scale), y), data)
    @test DistributionsInference.logdensity(prob, x) ≈ expected
    @test isfinite(DistributionsInference.logdensity(prob, x))

    # An object with no estimated rows: logdensity is just the data
    # likelihood, and the flat vector is empty.
    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    fixed_prob = DistributionsInference.as_logdensity(fixed_leaf, data)
    @test DistributionsInference.flat_dimension(fixed_leaf) == 0
    @test DistributionsInference.logdensity(fixed_prob, Float64[]) ≈
          sum(y -> logpdf(Gamma(2.0, 1.0), y), data)

    @test_throws DimensionMismatch DistributionsInference.logdensity(prob, Float64[])
    @test_throws DimensionMismatch DistributionsInference.logdensity(prob, [1.0, 2.0])
end

@testitem "engine: a custom loglik reducer is honoured" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    data = [1.5, 2.0, 3.2]
    double_count = (obj, d) -> 2 * sum(y -> logpdf(obj, y), d)
    prob = DistributionsInference.as_logdensity(leaf, data; loglik = double_count)

    x = [2.5]
    expected = logpdf(LogNormal(log(2.0), 0.2), 2.5) +
               2 * sum(y -> logpdf(Gamma(2.5, leaf.scale), y), data)
    @test DistributionsInference.logdensity(prob, x) ≈ expected
end

@testitem "engine: LogDensityProblems interface conformance" setup=[ToyFixture] begin
    using LogDensityProblems

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    data = [1.5, 2.0, 3.2, 2.8, 1.9, 4.1, 2.5, 3.0]
    prob = DistributionsInference.as_logdensity(leaf, data)

    @test LogDensityProblems.dimension(prob) ==
          DistributionsInference.flat_dimension(leaf) == 1
    @test LogDensityProblems.capabilities(typeof(prob)) ==
          LogDensityProblems.LogDensityOrder{0}()

    x = [2.5]
    @test LogDensityProblems.logdensity(prob, x) ==
          DistributionsInference.logdensity(prob, x)

    # A fully fixed object: dimension zero, evaluated at the empty vector.
    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    fixed_prob = DistributionsInference.as_logdensity(fixed_leaf, data)
    @test LogDensityProblems.dimension(fixed_prob) == 0
    @test isfinite(LogDensityProblems.logdensity(fixed_prob, Float64[]))
end

@testitem "engine acceptance: toy protocol object fits end-to-end via an LDP sampler" setup=[ToyFixture] begin
    using LogDensityProblems
    using Random

    # Data drawn from Gamma(shape = true_shape, scale); the scale is fixed at
    # its true value in the template so only the shape is estimated.
    rng = Random.Xoshiro(1)
    true_shape = 3.0
    scale = 1.5
    data = rand(rng, Gamma(true_shape, scale), 500)

    leaf = ToyGammaLeaf(2.0, scale, LogNormal(log(2.0), 0.5))
    prob = DistributionsInference.as_logdensity(leaf, data)
    @test LogDensityProblems.dimension(prob) == 1

    # A minimal random-walk Metropolis sampler driven purely through the
    # `LogDensityProblems` interface: no sampler package is added to this
    # repo, so `LogDensityProblems.logdensity` is the only contact point,
    # proving the toy protocol object is sampleable by a generic LDP
    # consumer.
    function metropolis(prob, x0; n = 4000, step = 0.15, rng = rng)
        x = copy(x0)
        lp = LogDensityProblems.logdensity(prob, x)
        draws = Vector{Vector{Float64}}(undef, n)
        for i in 1:n
            prop = x .+ step .* randn(rng, length(x))
            if any(<=(0), prop)
                draws[i] = copy(x)
                continue
            end
            lp_prop = LogDensityProblems.logdensity(prob, prop)
            if log(rand(rng)) < lp_prop - lp
                x, lp = prop, lp_prop
            end
            draws[i] = copy(x)
        end
        return draws
    end

    draws = metropolis(prob, [2.0])
    burn = draws[1001:end]
    post_mean = sum(first, burn) / length(burn)

    # The posterior mean lands closer to the true shape than the prior mean
    # (a real fit, not merely a runnable loop), and within a tight tolerance
    # given 500 observations.
    prior_mean = mean(LogNormal(log(2.0), 0.5))
    @test abs(post_mean - true_shape) < abs(prior_mean - true_shape)
    @test abs(post_mean - true_shape) < 0.5

    # The template reconstructs at the posterior mean into a genuine
    # concrete object, the fixed scale untouched.
    fitted = DistributionsInference.reconstruct(leaf, [post_mean])
    @test fitted.scale == scale
    @test fitted.shape ≈ post_mean
end
