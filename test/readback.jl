# `draws_to_chain`: keys raw sampler draws (a `dim x niter` matrix, or a
# vector of `dim`-length vectors) by the estimated rows' dotted names into a
# `FlexiChain` (deliberately matching ComposedDistributions' own
# `chain_to_params`/`param_draws` selection semantics). `FlexiChains` is a
# hard dependency (implemented directly in `src/readback.jl`), so every item
# here still loads `FlexiChains` itself for its own `FlexiChain`/`Parameter`
# constructors, rather than relying on a sibling item having done so.
# Chain-to-object readback (`inference_to_distribution`/
# `inference_to_distributions`) is covered in `test/inference_to_distribution.jl`.

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
    using Statistics: mean

    fixed_leaf = ToyGammaLeaf(2.0, 1.0)  # no prior: nothing estimated
    @test DistributionsInference.flat_dimension(fixed_leaf) == 0

    chain_mat = DistributionsInference.draws_to_chain(fixed_leaf, zeros(0, 5))
    chain_vec = DistributionsInference.draws_to_chain(
        fixed_leaf, [Float64[] for _ in 1:5])

    @test FlexiChains.niters(chain_mat) == FlexiChains.niters(chain_vec) == 5
    @test isempty(FlexiChains.parameters(chain_mat))
    @test isempty(FlexiChains.parameters(chain_vec))

    fitted = DistributionsInference.inference_to_distribution(
        fixed_leaf, chain_mat, mean)
    @test fitted == fixed_leaf
    all_fitted = DistributionsInference.inference_to_distributions(
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

@testitem "draws_to_chain: nchains > 1 pools chain-major, not iteration-major" setup=[
    ToyFixture] begin
    using FlexiChains: FlexiChains

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    # Chain 1 all 1.0s, chain 2 all 100.0s: a flat draws vector whose
    # per-chain blocks are disjoint and obviously identifiable, so a layout
    # bug (pooling iteration-major instead of chain-major) fails on the
    # actual numbers rather than merely on a count. `distribution_to_advancedmh`,
    # `distribution_to_turing`'s chain form, and `inference_to_distribution`'s
    # `draws` selection all share this ordering assumption, so this pins the
    # one place that assumption is actually built.
    chain1 = fill(1.0, 3)
    chain2 = fill(100.0, 3)
    flat = vcat(chain1, chain2)
    chain = DistributionsInference.draws_to_chain(
        leaf, reshape(flat, 1, :); nchains = 2)

    @test FlexiChains.niters(chain) == 3
    @test FlexiChains.nchains(chain) == 2

    pooled = vec(chain[:shape])
    @test pooled[1:3] == chain1
    @test pooled[4:6] == chain2

    all_fitted = DistributionsInference.inference_to_distributions(leaf, chain)
    @test [f.shape for f in all_fitted[1:3]] == chain1
    @test [f.shape for f in all_fitted[4:6]] == chain2
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
