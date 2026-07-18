# The dotted-name FlexiChains readback: `to_flexichain` (matrix and
# vector-of-vectors input, the 0-estimated edge case, malformed input),
# `readback`/`readback_draws` selection semantics (summary/draw/draws,
# matching ComposedDistributions' `chain_to_params`/`param_draws`
# docstrings), and the DI#3 acceptance criterion — a real LogDensityProblems
# sampler (AdvancedMH) round-trips through `to_flexichain`/`readback`.

@testitem "to_flexichain: matrix and vector-of-vectors input agree" setup=[ToyFixture] begin
    using FlexiChains: FlexiChains

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [2.1, 2.4, 2.0, 2.6]

    chain_mat = DistributionsInference.to_flexichain(leaf, reshape(values, 1, :))
    chain_vec = DistributionsInference.to_flexichain(leaf, [[v] for v in values])

    @test FlexiChains.niters(chain_mat) == FlexiChains.niters(chain_vec) == 4
    @test Set(FlexiChains.parameters(chain_mat)) == Set([:shape])
    @test vec(chain_mat[:shape]) == vec(chain_vec[:shape]) == values
end

@testitem "to_flexichain: keys are the estimated rows' dotted names" begin
    using DistributionsInference, Distributions
    using FlexiChains: FlexiChains

    rows = [
        (name = Symbol("leaf.shape"), value = 2.0, prior = LogNormal(0.0, 0.2),
            support = (0.0, Inf)),
        (name = Symbol("leaf.rate"), value = 1.0, prior = Gamma(2.0, 1.0),
            support = (0.0, Inf)),
        (name = :scale, value = 1.0, prior = nothing, support = (0.0, Inf))]

    draws = [1.0 2.0 3.0; 0.5 0.4 0.3]  # 2 estimated params x 3 draws
    chain = DistributionsInference.to_flexichain(rows, draws)

    @test Set(FlexiChains.parameters(chain)) ==
          Set([Symbol("leaf.shape"), Symbol("leaf.rate")])
    @test vec(chain[Symbol("leaf.shape")]) == [1.0, 2.0, 3.0]
    @test vec(chain[Symbol("leaf.rate")]) == [0.5, 0.4, 0.3]
end

@testitem "to_flexichain: the 0-estimated edge case" setup=[ToyFixture] begin
    using FlexiChains: FlexiChains

    fixed_leaf = ToyGammaLeaf(2.0, 1.0)  # no prior: nothing estimated
    @test DistributionsInference.flat_dimension(fixed_leaf) == 0

    chain_mat = DistributionsInference.to_flexichain(fixed_leaf, zeros(0, 5))
    chain_vec = DistributionsInference.to_flexichain(
        fixed_leaf, [Float64[] for _ in 1:5])

    @test FlexiChains.niters(chain_mat) == FlexiChains.niters(chain_vec) == 5
    @test isempty(FlexiChains.parameters(chain_mat))
    @test isempty(FlexiChains.parameters(chain_vec))

    # A chain with no parameters still readback the (unchanged) fixed object,
    # once per iteration.
    fitted = DistributionsInference.readback(fixed_leaf, chain_mat)
    @test fitted == fixed_leaf
    all_fitted = DistributionsInference.readback_draws(fixed_leaf, chain_mat)
    @test length(all_fitted) == 5
    @test all(==(fixed_leaf), all_fitted)
end

@testitem "to_flexichain: malformed draws raise" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))

    @test_throws DimensionMismatch DistributionsInference.to_flexichain(
        leaf, [1.0 2.0; 3.0 4.0])  # 2 rows but only 1 estimated parameter
    @test_throws DimensionMismatch DistributionsInference.to_flexichain(
        leaf, [[1.0, 2.0], [3.0, 4.0]])  # draws of length 2, dim is 1
    @test_throws ArgumentError DistributionsInference.to_flexichain(
        leaf, "not a matrix or vector-of-vectors")
end

@testitem "readback: summary/draw/draws selection semantics" setup=[ToyFixture] begin
    using Statistics: median

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [1.0, 2.0, 3.0, 4.0]
    chain = DistributionsInference.to_flexichain(leaf, reshape(values, 1, :))

    # Default summary is `mean` over every draw.
    @test DistributionsInference.readback(leaf, chain).shape ≈ 2.5

    # A custom summary reducer.
    @test DistributionsInference.readback(leaf, chain; summary = median).shape ≈ 2.5
    @test DistributionsInference.readback(leaf, chain; summary = maximum).shape ≈ 4.0

    # A single draw overrides `summary`.
    @test DistributionsInference.readback(leaf, chain; draw = 2).shape ≈ 2.0

    # `draws` restricts to a subset of iterations before reducing: an index
    # range/vector, or a predicate over the iteration index.
    @test DistributionsInference.readback(leaf, chain; draws = 2:3).shape ≈ 2.5
    @test DistributionsInference.readback(leaf, chain; draws = [1, 4]).shape ≈ 2.5
    @test DistributionsInference.readback(leaf, chain; draws = i -> i > 2).shape ≈ 3.5

    # The fixed parameter is always held at its template value.
    @test DistributionsInference.readback(leaf, chain).scale == leaf.scale
end

@testitem "readback_draws: keeps every draw, restricted by `draws`" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [1.0, 2.0, 3.0, 4.0]
    chain = DistributionsInference.to_flexichain(leaf, reshape(values, 1, :))

    all_fitted = DistributionsInference.readback_draws(leaf, chain)
    @test length(all_fitted) == 4
    @test [f.shape for f in all_fitted] == values

    subset = DistributionsInference.readback_draws(leaf, chain; draws = 2:3)
    @test [f.shape for f in subset] == [2.0, 3.0]

    predicate = DistributionsInference.readback_draws(leaf, chain; draws = i -> i > 2)
    @test [f.shape for f in predicate] == [3.0, 4.0]
end

@testitem "readback: a chain missing an estimated parameter errors" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    other_leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))

    rows = [(name = :not_shape, value = 2.0, prior = LogNormal(0.0, 0.2),
        support = (0.0, Inf))]
    mismatched_chain = DistributionsInference.to_flexichain(rows, reshape([2.0], 1, :))

    @test_throws ArgumentError DistributionsInference.readback(leaf, mismatched_chain)
end

@testitem "readback acceptance: an AdvancedMH sampler round-trips" setup=[ToyFixture] begin
    using AdvancedMH
    using FlexiChains: FlexiChains
    using LogDensityProblems
    using LinearAlgebra: I
    using Random

    # Data drawn from Gamma(shape = true_shape, scale); the scale is fixed at
    # its true value in the template so only the shape is estimated, exactly
    # as in the engine's own acceptance test (test/engine.jl) — here the
    # sampler is a real `LogDensityProblems` consumer (AdvancedMH's
    # random-walk Metropolis), not a hand-rolled loop, and its draws are read
    # back through `to_flexichain`/`readback` rather than indexed by hand.
    rng = Random.Xoshiro(1)
    true_shape = 3.0
    scale = 1.5
    data = rand(rng, Gamma(true_shape, scale), 500)

    leaf = ToyGammaLeaf(2.0, scale, LogNormal(log(2.0), 0.5))
    prob = DistributionsInference.as_logdensity(leaf, data)
    @test LogDensityProblems.dimension(prob) == 1

    # Guard the support the same way the engine's own acceptance test does
    # (test/engine.jl): the shape prior is defined only for positive values,
    # and AdvancedMH's random-walk proposal does not respect that on its own.
    model = AdvancedMH.DensityModel() do x
        any(<=(0), x) ? -Inf : LogDensityProblems.logdensity(prob, x)
    end
    spl = RWMH(MvNormal(zeros(1), 0.05^2 * I))
    transitions = sample(
        rng, model, spl, 5000; param_names = ["shape"], progress = false)
    # AdvancedMH hands draws back as a vector of `Transition`s; `.params` is
    # the raw niter-vector-of-dim-vectors shape `to_flexichain` accepts.
    draws = [t.params for t in transitions][2001:end]

    chain = DistributionsInference.to_flexichain(leaf, draws)
    @test FlexiChains.niters(chain) == length(draws)

    fitted = DistributionsInference.readback(leaf, chain)
    prior_mean = mean(LogNormal(log(2.0), 0.5))
    @test abs(fitted.shape - true_shape) < abs(prior_mean - true_shape)
    @test abs(fitted.shape - true_shape) < 0.5
    @test fitted.scale == scale  # the fixed parameter is untouched

    # `readback_draws`'s own posterior mean agrees with `readback`'s point
    # summary.
    all_fitted = DistributionsInference.readback_draws(leaf, chain)
    @test length(all_fitted) == length(draws)
    @test mean(f -> f.shape, all_fitted) ≈ fitted.shape
end
