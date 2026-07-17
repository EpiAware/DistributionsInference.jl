# The fit protocol: `parameter_rows`/`estimated_rows`/`flat_dimension` over a
# bare row vector (the identity fallback) and over a toy protocol
# implementation, plus `reconstruct`'s round-trip contract (#2).

@testsnippet ToyFixture begin
    using DistributionsInference, Distributions

    # A minimal fit-protocol object: a Gamma leaf with its shape ESTIMATED (an
    # attached prior) and its scale fixed. Implementing the protocol needs
    # only two methods on the object's own type, no DistributionsInference
    # dependency at the point of definition beyond the method extension
    # itself (ComposedDistributions#185's "implementable without loading us").
    struct ToyGammaLeaf
        shape::Float64
        scale::Float64
        shape_prior::Union{Nothing, Distribution}
    end

    ToyGammaLeaf(shape::Real, scale::Real) = ToyGammaLeaf(shape, scale, nothing)

    Distributions.logpdf(d::ToyGammaLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

    function DistributionsInference.parameter_rows(d::ToyGammaLeaf)
        return [
            (name = :shape, value = d.shape, prior = d.shape_prior,
                support = (0.0, Inf)),
            (name = :scale, value = d.scale, prior = nothing,
                support = (0.0, Inf))
        ]
    end

    function DistributionsInference.reconstruct(d::ToyGammaLeaf, x::AbstractVector)
        n = DistributionsInference.flat_dimension(d)
        length(x) == n || throw(DimensionMismatch(
            "ToyGammaLeaf has $n estimated parameter(s), got $(length(x))"))
        # `shape` is the only parameter that can carry a prior; a fixed
        # leaf (n == 0) has nothing to read from `x` and round-trips as-is.
        n == 0 && return d
        return ToyGammaLeaf(x[1], d.scale, d.shape_prior)
    end
end

@testitem "parameter_rows: bare row vector is its own identity" setup=[ToyFixture] begin
    rows = [
        (name = :shape, value = 2.0, prior = LogNormal(0.0, 0.2),
            support = (0.0, Inf)),
        (name = :scale, value = 1.0, prior = nothing, support = (0.0, Inf))]
    @test DistributionsInference.parameter_rows(rows) === rows
end

@testitem "estimated_rows: filters to the prior-carrying rows" setup=[ToyFixture] begin
    rows = [
        (name = :shape, value = 2.0, prior = LogNormal(0.0, 0.2),
            support = (0.0, Inf)),
        (name = :scale, value = 1.0, prior = nothing, support = (0.0, Inf))]
    est = DistributionsInference.estimated_rows(rows)
    @test length(est) == 1
    @test only(est).name == :shape

    # A fully fixed row set estimates nothing.
    fixed_rows = [(name = :scale, value = 1.0, prior = nothing,
        support = (0.0, Inf))]
    @test isempty(DistributionsInference.estimated_rows(fixed_rows))
end

@testitem "flat_dimension: counts the estimated rows" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    @test DistributionsInference.flat_dimension(leaf) == 1

    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    @test DistributionsInference.flat_dimension(fixed_leaf) == 0
end

@testitem "reconstruct: rebuilds the estimated parameter, holds the fixed one" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    rebuilt = DistributionsInference.reconstruct(leaf, [3.5])
    @test rebuilt.shape == 3.5
    @test rebuilt.scale == leaf.scale
    @test rebuilt.shape_prior === leaf.shape_prior

    # A tree with no estimated rows round-trips at the empty vector.
    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    @test DistributionsInference.reconstruct(fixed_leaf, Float64[]) == fixed_leaf

    @test_throws DimensionMismatch DistributionsInference.reconstruct(leaf, Float64[])
    @test_throws DimensionMismatch DistributionsInference.reconstruct(leaf, [1.0, 2.0])
end
