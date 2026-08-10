# The dotted-name FlexiChains readback: `draws_to_chain`,
# `point_estimate`/`distribution_draws` selection semantics (deliberately
# matching ComposedDistributions' `chain_to_params`/`param_draws`), and the DI#3
# acceptance criterion (a real AdvancedMH round-trip). `FlexiChains` is a hard
# dependency (implemented directly in `src/readback.jl`), so every item here
# still loads `FlexiChains` itself for its own `FlexiChain`/`Parameter`
# constructors, rather than relying on a sibling item having done so.

@testitem "the readback names the chain type for a non-chain argument" begin
    using DistributionsInference, Distributions

    # A non-chain second argument is the caller's mistake, so the message
    # must name the type. `distribution_params`/`point_estimate`/
    # `distribution_draws` accept a `FlexiChain` only, so a raw matrix is
    # already the wrong type for them.
    rows = [(name = :shape, value = 2.0, prior = LogNormal(0.0, 0.2),
        support = (0.0, Inf))]
    for f in (DistributionsInference.distribution_params,
        DistributionsInference.point_estimate,
        DistributionsInference.distribution_draws)
        thrown = try
            f(rows, [1.0 2.0])
            nothing
        catch e
            e
        end
        @test thrown isa ArgumentError
        @test occursin("has no method for a chain of type", thrown.msg)
    end

    # `inference_to_distribution(s)` also accept raw draws (a matrix or a
    # vector-of-vectors), so a matrix is not the wrong type for them; use a
    # genuinely unsupported second argument instead.
    for f in (DistributionsInference.inference_to_distributions,
        DistributionsInference.inference_to_distribution)
        thrown = try
            f(rows, nothing)
            nothing
        catch e
            e
        end
        @test thrown isa ArgumentError
        @test occursin("has no method for a chain of type", thrown.msg)
    end
end

@testitem "draws_to_chain: matrix and vector-of-vectors input agree" setup=[ToyFixture] begin
    using FlexiChains: FlexiChains

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [2.1, 2.4, 2.0, 2.6]

    chain_mat = DistributionsInference.draws_to_chain(leaf, reshape(values, 1, :))
    chain_vec = DistributionsInference.draws_to_chain(leaf, [[v] for v in values])

    @test FlexiChains.niters(chain_mat) == FlexiChains.niters(chain_vec) == 4
    @test Set(FlexiChains.parameters(chain_mat)) == Set([:shape])
    @test vec(chain_mat[:shape]) == vec(chain_vec[:shape]) == values
end

@testitem "draws_to_chain: keys are the estimated rows' dotted names" begin
    using DistributionsInference, Distributions
    using FlexiChains: FlexiChains

    rows = [
        (name = Symbol("leaf.shape"), value = 2.0, prior = LogNormal(0.0, 0.2),
            support = (0.0, Inf)),
        (name = Symbol("leaf.rate"), value = 1.0, prior = Gamma(2.0, 1.0),
            support = (0.0, Inf)),
        (name = :scale, value = 1.0, prior = nothing, support = (0.0, Inf))]

    draws = [1.0 2.0 3.0; 0.5 0.4 0.3]  # 2 estimated params x 3 draws
    chain = DistributionsInference.draws_to_chain(rows, draws)

    @test Set(FlexiChains.parameters(chain)) ==
          Set([Symbol("leaf.shape"), Symbol("leaf.rate")])
    @test vec(chain[Symbol("leaf.shape")]) == [1.0, 2.0, 3.0]
    @test vec(chain[Symbol("leaf.rate")]) == [0.5, 0.4, 0.3]
end

@testitem "draws_to_chain: the 0-estimated edge case" setup=[ToyFixture] begin
    using FlexiChains: FlexiChains

    fixed_leaf = ToyGammaLeaf(2.0, 1.0)  # no prior: nothing estimated
    @test DistributionsInference.flat_dimension(fixed_leaf) == 0

    chain_mat = DistributionsInference.draws_to_chain(fixed_leaf, zeros(0, 5))
    chain_vec = DistributionsInference.draws_to_chain(
        fixed_leaf, [Float64[] for _ in 1:5])

    @test FlexiChains.niters(chain_mat) == FlexiChains.niters(chain_vec) == 5
    @test isempty(FlexiChains.parameters(chain_mat))
    @test isempty(FlexiChains.parameters(chain_vec))

    fitted = DistributionsInference.point_estimate(fixed_leaf, chain_mat)
    @test fitted == fixed_leaf
    all_fitted = DistributionsInference.distribution_draws(
        fixed_leaf, chain_mat)
    @test length(all_fitted) == 5
    @test all(==(fixed_leaf), all_fitted)
end

@testitem "draws_to_chain: malformed draws raise" setup=[ToyFixture] begin
    using FlexiChains: FlexiChains

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))

    @test_throws DimensionMismatch DistributionsInference.draws_to_chain(
        leaf, [1.0 2.0; 3.0 4.0])  # 2 rows but only 1 estimated parameter
    @test_throws DimensionMismatch DistributionsInference.draws_to_chain(
        leaf, [[1.0, 2.0], [3.0, 4.0]])  # draws of length 2, dim is 1
    @test_throws ArgumentError DistributionsInference.draws_to_chain(
        leaf, "not a matrix or vector-of-vectors")
end

@testitem "distribution_params: keyed by dotted name, params-first" setup=[ToyFixture] begin
    using FlexiChains: FlexiChains

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [1.0, 2.0, 3.0, 4.0]
    chain = DistributionsInference.draws_to_chain(leaf, reshape(values, 1, :))

    nt = DistributionsInference.distribution_params(leaf, chain)
    @test nt isa NamedTuple
    @test keys(nt) == (:shape,)   # only the estimated row, not `scale`
    @test nt.shape ≈ 2.5          # default summary is `mean`

    # Same selection semantics as `point_estimate`, the primitive underneath.
    @test DistributionsInference.distribution_params(
        leaf, chain; draw = 2).shape ≈ 2.0
    @test DistributionsInference.distribution_params(
        leaf, chain; draws = 2:3).shape ≈ 2.5

    # `point_estimate` collapses this to a flat vector in `estimated_rows` order.
    @test DistributionsInference.point_estimate(leaf, chain).shape == nt.shape

    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    fixed_chain = DistributionsInference.draws_to_chain(fixed_leaf, zeros(0, 3))
    @test DistributionsInference.distribution_params(fixed_leaf, fixed_chain) ==
          NamedTuple()
end

@testitem "distribution_params: a dotted (nested) row name round-trips" begin
    using DistributionsInference, Distributions
    using FlexiChains: FlexiChains

    rows = [
        (name = Symbol("leaf.shape"), value = 2.0, prior = LogNormal(0.0, 0.2),
            support = (0.0, Inf)),
        (name = :scale, value = 1.0, prior = nothing, support = (0.0, Inf))]
    draws = [1.0 2.0 3.0]
    chain = DistributionsInference.draws_to_chain(rows, draws)

    nt = DistributionsInference.distribution_params(rows, chain)
    @test keys(nt) == (Symbol("leaf.shape"),)
    @test nt[Symbol("leaf.shape")] ≈ 2.0
end

@testitem "distribution_params: a duplicate estimated name errors clearly" begin
    using DistributionsInference, Distributions
    using FlexiChains: FlexiChains

    # Two rows sharing a dotted name is a `parameter_rows` protocol bug with
    # no sensible dedupe; `NamedTuple{names}(...)` would otherwise fail with a
    # bare "duplicate field name" naming neither the object nor the name.
    rows = [
        (name = :shape, value = 2.0, prior = LogNormal(0.0, 0.2),
            support = (0.0, Inf)),
        (name = :shape, value = 1.0, prior = LogNormal(0.0, 0.2),
            support = (0.0, Inf))]
    chain = DistributionsInference.draws_to_chain(rows, [1.0 2.0 3.0; 4.0 5.0 6.0])

    err = try
        DistributionsInference.distribution_params(rows, chain)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("duplicate", err.msg)
    @test occursin("shape", err.msg)
end

@testitem "point_estimate: summary/draw/draws selection semantics" setup=[ToyFixture] begin
    using FlexiChains: FlexiChains
    using Statistics: median

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [1.0, 2.0, 3.0, 4.0]
    chain = DistributionsInference.draws_to_chain(leaf, reshape(values, 1, :))

    # Default summary is `mean` over every draw.
    @test DistributionsInference.point_estimate(leaf, chain).shape ≈ 2.5

    @test DistributionsInference.point_estimate(leaf, chain; summary = median).shape ≈ 2.5
    @test DistributionsInference.point_estimate(leaf, chain; summary = maximum).shape ≈ 4.0

    # A single draw overrides `summary`.
    @test DistributionsInference.point_estimate(leaf, chain; draw = 2).shape ≈ 2.0

    # `draws` restricts to a subset of iterations before reducing.
    @test DistributionsInference.point_estimate(leaf, chain; draws = 2:3).shape ≈ 2.5
    @test DistributionsInference.point_estimate(leaf, chain; draws = [1, 4]).shape ≈ 2.5
    @test DistributionsInference.point_estimate(leaf, chain; draws = i -> i > 2).shape ≈ 3.5

    @test DistributionsInference.point_estimate(leaf, chain).scale == leaf.scale
end

@testitem "distribution_draws: keeps every draw, restricted by `draws`" setup=[ToyFixture] begin
    using FlexiChains: FlexiChains

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [1.0, 2.0, 3.0, 4.0]
    chain = DistributionsInference.draws_to_chain(leaf, reshape(values, 1, :))

    all_fitted = DistributionsInference.distribution_draws(leaf, chain)
    @test length(all_fitted) == 4
    @test [f.shape for f in all_fitted] == values

    subset = DistributionsInference.distribution_draws(leaf, chain; draws = 2:3)
    @test [f.shape for f in subset] == [2.0, 3.0]

    predicate = DistributionsInference.distribution_draws(
        leaf, chain; draws = i -> i > 2)
    @test [f.shape for f in predicate] == [3.0, 4.0]
end

@testitem "distribution_draws: pools draws across every chain" setup=[ToyFixture] begin
    using FlexiChains: FlexiChains

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    # 2 chains x 3 iterations, distinct value ranges per chain so pooling
    # (rather than truncation to chain 1) can be checked by value, not just
    # count. `_chain_column` returns the full niter x nchains array and
    # `vec` flattens it column-major, so chain 1 occupies entries 1:3 and
    # chain 2 occupies entries 4:6 of the pooled range.
    chain1 = [1.0, 2.0, 3.0]
    chain2 = [10.0, 20.0, 30.0]
    mat = reshape(vcat(chain1, chain2), 3, 2)  # niter x nchains
    data = Dict{FlexiChains.ParameterOrExtra{<:Symbol}, Matrix}(
        FlexiChains.Parameter(:shape) => mat)
    chain = FlexiChains.FlexiChain{Symbol}(3, 2, data)
    @test FlexiChains.nchains(chain) == 2

    all_fitted = DistributionsInference.distribution_draws(leaf, chain)
    @test length(all_fitted) == 6
    @test [f.shape for f in all_fitted] == vcat(chain1, chain2)

    # A predicate selector operates over the pooled 1:6 range, not 1:3.
    predicate = DistributionsInference.distribution_draws(
        leaf, chain; draws = i -> i > 3)
    @test [f.shape for f in predicate] == chain2

    explicit = DistributionsInference.distribution_draws(
        leaf, chain; draws = 4:6)
    @test [f.shape for f in explicit] == chain2
end

@testitem "distribution_params: a predicate selector pools across chains" setup=[ToyFixture] begin
    using FlexiChains: FlexiChains

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    chain1 = [1.0, 2.0, 3.0]
    chain2 = [10.0, 20.0, 30.0]
    mat = reshape(vcat(chain1, chain2), 3, 2)
    data = Dict{FlexiChains.ParameterOrExtra{<:Symbol}, Matrix}(
        FlexiChains.Parameter(:shape) => mat)
    chain = FlexiChains.FlexiChain{Symbol}(3, 2, data)

    # Before the fix, a predicate's index range was capped at
    # `1:niters(chain)` (3), so `i -> i > 3` selected nothing out of a
    # 6-draw pooled range.
    nt = DistributionsInference.distribution_params(
        leaf, chain; draws = i -> i > 3)
    @test nt.shape ≈ mean(chain2)

    nt_all = DistributionsInference.distribution_params(leaf, chain)
    @test nt_all.shape ≈ mean(vcat(chain1, chain2))
end

@testitem "point_estimate: a chain missing an estimated parameter errors" setup=[ToyFixture] begin
    using FlexiChains: FlexiChains

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    other_leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))

    rows = [(name = :not_shape, value = 2.0, prior = LogNormal(0.0, 0.2),
        support = (0.0, Inf))]
    mismatched_chain = DistributionsInference.draws_to_chain(rows, reshape([2.0], 1, :))

    @test_throws ArgumentError DistributionsInference.point_estimate(leaf, mismatched_chain)
end

@testitem "readback acceptance: an AdvancedMH sampler round-trips" setup=[ToyFixture] begin
    using AdvancedMH
    using FlexiChains: FlexiChains
    using LogDensityProblems
    using LinearAlgebra: I
    using Random

    # As in the engine's own acceptance test, but with a real
    # `LogDensityProblems` consumer rather than a hand-rolled loop, and draws
    # read back through `draws_to_chain`/`point_estimate`.
    rng = Random.Xoshiro(1)
    true_shape = 3.0
    scale = 1.5
    data = rand(rng, Gamma(true_shape, scale), 500)

    leaf = ToyGammaLeaf(2.0, scale, LogNormal(log(2.0), 0.5))
    prob = DistributionsInference.distribution_to_logdensity(leaf, data)
    @test LogDensityProblems.dimension(prob) == 1

    # The shape prior is defined only for positive values and AdvancedMH's
    # random-walk proposal does not respect that on its own.
    model = AdvancedMH.DensityModel() do x
        any(<=(0), x) ? -Inf : LogDensityProblems.logdensity(prob, x)
    end
    spl = RWMH(MvNormal(zeros(1), 0.05^2 * I))
    transitions = sample(
        rng, model, spl, 5000; param_names = ["shape"], progress = false)
    # `.params` is the niter-vector-of-dim-vectors `draws_to_chain` accepts.
    draws = [t.params for t in transitions][2001:end]

    chain = DistributionsInference.draws_to_chain(leaf, draws)
    @test FlexiChains.niters(chain) == length(draws)

    fitted = DistributionsInference.point_estimate(leaf, chain)
    prior_mean = mean(LogNormal(log(2.0), 0.5))
    @test abs(fitted.shape - true_shape) < abs(prior_mean - true_shape)
    @test abs(fitted.shape - true_shape) < 0.5
    @test fitted.scale == scale

    all_fitted = DistributionsInference.distribution_draws(leaf, chain)
    @test length(all_fitted) == length(draws)
    @test mean(f -> f.shape, all_fitted) ≈ fitted.shape
end
