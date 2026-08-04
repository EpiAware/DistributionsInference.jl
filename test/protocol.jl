# The fit protocol over a bare row vector (the identity fallback) and over a
# toy protocol implementation, plus `reconstruct`'s round-trip contract (#2).

@testsnippet ToyFixture begin
    using DistributionsInference, Distributions

    # A minimal fit-protocol object: a Gamma leaf with its shape ESTIMATED (an
    # attached prior) and its scale fixed. Implementable without loading us
    # (CD#185).
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
        n == 0 && return d
        return ToyGammaLeaf(x[1], d.scale, d.shape_prior)
    end
end

@testitem "parameter_rows/reconstruct: a type omitting the protocol is named" begin
    using DistributionsInference

    # The two generic fallbacks are the protocol's only error surface for a
    # type that never implemented it, so the message has to name the offending
    # type AND the method to add. An `Int` stands in for any such type: it
    # matches neither the bare-row-vector identity nor a user method.
    rows_err = try
        DistributionsInference.parameter_rows(1)
        nothing
    catch caught
        caught
    end
    @test rows_err isa ArgumentError
    @test occursin("no `parameter_rows` method for Int", rows_err.msg)
    @test occursin("parameter_rows(obj)", rows_err.msg)

    rec_err = try
        DistributionsInference.reconstruct(1, [1.0])
        nothing
    catch caught
        caught
    end
    @test rec_err isa ArgumentError
    @test occursin("no `reconstruct` method for Int", rec_err.msg)
    @test occursin("reconstruct(obj, x::AbstractVector)", rec_err.msg)

    # Everything built on `parameter_rows` inherits the same clear error
    # rather than a `MethodError` from somewhere further in.
    @test_throws ArgumentError DistributionsInference.estimated_rows(1)
    @test_throws ArgumentError DistributionsInference.flat_dimension(1)
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

    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    @test DistributionsInference.reconstruct(fixed_leaf, Float64[]) == fixed_leaf

    @test_throws DimensionMismatch DistributionsInference.reconstruct(leaf, Float64[])
    @test_throws DimensionMismatch DistributionsInference.reconstruct(leaf, [1.0, 2.0])
end

@testitem "reconstruct: a bare row vector is a minimal fittable object" setup=[ToyFixture] begin
    rows = [
        (name = :shape, value = 2.0, prior = LogNormal(0.0, 0.2),
            support = (0.0, Inf)),
        (name = :scale, value = 1.0, prior = nothing, support = (0.0, Inf))]

    rebuilt = DistributionsInference.reconstruct(rows, [3.5])
    @test rebuilt[1].value == 3.5
    @test rebuilt[1].name == :shape
    @test rebuilt[1].prior == rows[1].prior
    @test rebuilt[2] == rows[2]

    @test_throws DimensionMismatch DistributionsInference.reconstruct(rows, Float64[])
    @test_throws DimensionMismatch DistributionsInference.reconstruct(rows, [1.0, 2.0])

    fixed_rows = [(name = :scale, value = 1.0, prior = nothing,
        support = (0.0, Inf))]
    @test DistributionsInference.reconstruct(fixed_rows, Float64[]) == fixed_rows
end

@testitem "extra_logprior: neutral by default, wired for an overriding type" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    rebuilt = DistributionsInference.reconstruct(leaf, [3.5])
    @test DistributionsInference.extra_prior_state(leaf) === nothing
    @test DistributionsInference.extra_logprior(leaf, rebuilt, [3.5], nothing) ==
          0.0

    # A type overriding it: an object-dependent penalty scored against the
    # reconstructed object rather than a per-row prior.
    struct PenalisedLeaf
        shape::Float64
        scale::Float64
    end

    function DistributionsInference.parameter_rows(d::PenalisedLeaf)
        return [(name = :shape, value = d.shape, prior = nothing,
                support = (0.0, Inf)),
            (name = :scale, value = d.scale, prior = nothing,
                support = (0.0, Inf))]
    end

    # Estimate `shape` only, overriding the generic prior-based default.
    DistributionsInference.estimated_rows(d::PenalisedLeaf) = [
        DistributionsInference.parameter_rows(d)[1]]
    DistributionsInference.flat_dimension(::PenalisedLeaf) = 1
    function DistributionsInference.reconstruct(d::PenalisedLeaf, x::AbstractVector)
        return PenalisedLeaf(x[1], d.scale)
    end
    function DistributionsInference.extra_logprior(
            ::PenalisedLeaf, rebuilt::PenalisedLeaf, x::AbstractVector, ::Any)
        return -0.5 * rebuilt.shape^2
    end
    Distributions.logpdf(d::PenalisedLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

    template = PenalisedLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2]
    prob = DistributionsInference.distribution_to_logdensity(template, data)
    x = [2.5]
    expected = -0.5 * 2.5^2 + sum(y -> logpdf(Gamma(2.5, 1.0), y), data)
    @test DistributionsInference.logdensity(prob, x) ≈ expected
end

@testitem "reconstruct: a concrete estimated field is guarded against a tracer number (DI#48)" begin
    using DistributionsInference, Distributions, ForwardDiff

    # The bug this guards: an ESTIMATED field typed to a CONCRETE Float64
    # rejects a `ForwardDiff.Dual` with an opaque `MethodError` inside
    # `reconstruct`.
    struct ConcreteFitLeaf
        shape::Float64
        scale::Float64
    end

    Distributions.logpdf(d::ConcreteFitLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

    function DistributionsInference.parameter_rows(d::ConcreteFitLeaf)
        return [
            (name = :shape, value = d.shape,
                prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
            (name = :scale, value = d.scale, prior = nothing,
                support = (0.0, Inf))]
    end

    function DistributionsInference.reconstruct(d::ConcreteFitLeaf, x::AbstractVector)
        return ConcreteFitLeaf(x[1], d.scale)
    end

    leaf = ConcreteFitLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2]
    prob = DistributionsInference.distribution_to_logdensity(leaf, data)

    @test DistributionsInference.logdensity(prob, [2.5]) ≈
          logpdf(LogNormal(log(2.0), 0.2), 2.5) +
          sum(y -> logpdf(Gamma(2.5, 1.0), y), data)

    # A `Dual`-valued flat vector: the guard raises a named `ArgumentError`
    # instead of the opaque `MethodError`.
    dual_x = [ForwardDiff.Dual(2.5, 1.0)]
    err = try
        DistributionsInference.logdensity(prob, dual_x)
        nothing
    catch caught
        caught
    end
    @test err isa ArgumentError
    @test occursin("shape", err.msg)
    @test occursin("ConcreteFitLeaf", err.msg)
    @test occursin("generically typed", err.msg)

    # A GENERICALLY typed field: the same `Dual`-valued vector flows through
    # untouched, no guard fires.
    struct GenericFitLeaf{S <: Real}
        shape::S
        scale::Float64
    end

    Distributions.logpdf(d::GenericFitLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

    function DistributionsInference.parameter_rows(d::GenericFitLeaf)
        return [
            (name = :shape, value = d.shape,
                prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
            (name = :scale, value = d.scale, prior = nothing,
                support = (0.0, Inf))]
    end

    function DistributionsInference.reconstruct(d::GenericFitLeaf, x::AbstractVector)
        return GenericFitLeaf(x[1], d.scale)
    end

    generic_leaf = GenericFitLeaf(2.0, 1.0)
    generic_prob = DistributionsInference.distribution_to_logdensity(generic_leaf, data)
    @test DistributionsInference.logdensity(generic_prob, dual_x) isa
          ForwardDiff.Dual
end
