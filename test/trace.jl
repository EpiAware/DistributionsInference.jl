# The core `PosteriorTrace` (#90): construction validation (divisibility,
# the empty-trace refusal, the `dim == 0` edge case), `AbstractVector`
# conformance (`getindex`, slicing to `nchains = 1`, `collect`), and the
# transforms built on it (`parameter_draws`, `trace_to_distribution`,
# `point_estimate`, and the `mean(::PosteriorTrace)` refusal). No
# `FlexiChains` dependency; `ToyFixture` (from `test/protocol.jl`) supplies
# `ToyGammaLeaf`.

@testsnippet TraceDistFixture begin
    using DistributionsInference, Distributions

    # Reconstructs to a genuine `Distributions.Distribution`, unlike
    # `ToyFixture`'s `ToyGammaLeaf` (which carries a `logpdf` method rather
    # than subtyping `Distribution`): `trace_to_distribution` needs a real
    # `Distribution` to mix over, and this is the fixture that gives it one.
    struct GammaShapeTemplate
        shape::Float64
        shape_prior::Distribution
    end

    function DistributionsInference.parameter_rows(d::GammaShapeTemplate)
        return [(name = :shape, value = d.shape, prior = d.shape_prior,
            support = (0.0, Inf))]
    end

    function DistributionsInference.reconstruct(
            d::GammaShapeTemplate, x::AbstractVector)
        return Gamma(x[1], 1.0)
    end
end

@testitem "draws_to_trace: matrix and vector-of-vectors input agree" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [2.1, 2.4, 2.0, 2.6]

    trace_mat = DistributionsInference.draws_to_trace(
        leaf, reshape(values, 1, :))
    trace_vec = DistributionsInference.draws_to_trace(
        leaf, [[v] for v in values])

    @test length(trace_mat) == length(trace_vec) == 4
    @test trace_mat.names == trace_vec.names == [:shape]
    @test [t.shape for t in trace_mat] == [t.shape for t in trace_vec] == values
    @test trace_mat.nchains == trace_vec.nchains == 1
    @test trace_mat.stats == trace_vec.stats == NamedTuple()
end

@testitem "draws_to_trace: malformed draws raise, same as the chain readback" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))

    @test_throws DimensionMismatch DistributionsInference.draws_to_trace(
        leaf, [1.0 2.0; 3.0 4.0])  # 2 rows but only 1 estimated parameter
    @test_throws DimensionMismatch DistributionsInference.draws_to_trace(
        leaf, [[1.0, 2.0], [3.0, 4.0]])  # draws of length 2, dim is 1
    @test_throws ArgumentError DistributionsInference.draws_to_trace(
        leaf, "not a matrix or vector-of-vectors")
end

@testitem "PosteriorTrace: refuses construction of an empty trace" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))

    err = try
        DistributionsInference.draws_to_trace(leaf, reshape(Float64[], 1, 0))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("0 draws", err.msg)

    # The vector-of-vectors shape hits the same guard.
    err2 = try
        DistributionsInference.draws_to_trace(leaf, Vector{Float64}[])
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("0 draws", err2.msg)
end

@testitem "PosteriorTrace: nchains must divide ndraws, naming both" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    draws = reshape(collect(1.0:5.0), 1, 5)

    err = try
        DistributionsInference.draws_to_trace(leaf, draws; nchains = 2)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("5", err.msg)
    @test occursin("2", err.msg)
end

@testitem "PosteriorTrace: nchains pools draws without reordering them" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    draws = reshape(collect(1.0:6.0), 1, 6)

    trace = DistributionsInference.draws_to_trace(leaf, draws; nchains = 2)
    @test trace.nchains == 2
    @test length(trace) == 6
    @test [t.shape for t in trace] == collect(1.0:6.0)
end

@testitem "PosteriorTrace: dim == 0 is legitimate and keeps working" setup=[ToyFixture] begin
    fixed_leaf = ToyGammaLeaf(2.0, 1.0)  # no prior: nothing estimated
    @test DistributionsInference.flat_dimension(fixed_leaf) == 0

    trace_mat = DistributionsInference.draws_to_trace(
        fixed_leaf, zeros(0, 5))
    trace_vec = DistributionsInference.draws_to_trace(
        fixed_leaf, [Float64[] for _ in 1:5])

    @test length(trace_mat) == length(trace_vec) == 5
    @test isempty(trace_mat.names) && isempty(trace_vec.names)
    @test all(==(fixed_leaf), trace_mat)
    @test all(==(fixed_leaf), trace_vec)

    # The transforms built on the trace must survive it too.
    @test DistributionsInference.point_estimate(trace_mat) == fixed_leaf
    @test DistributionsInference.parameter_draws(trace_mat) == NamedTuple()
end

@testitem "PosteriorTrace: getindex(::Int) reconstructs on demand from a view" begin
    using DistributionsInference, Distributions

    struct ViewCheckLeaf
        shape::Float64
    end

    function DistributionsInference.parameter_rows(d::ViewCheckLeaf)
        return [(name = :shape, value = d.shape,
            prior = LogNormal(0.0, 0.2), support = (0.0, Inf))]
    end

    seen = Ref{Any}(nothing)
    function DistributionsInference.reconstruct(
            d::ViewCheckLeaf, x::AbstractVector)
        seen[] = x
        return ViewCheckLeaf(x[1])
    end

    leaf = ViewCheckLeaf(2.0)
    trace = DistributionsInference.draws_to_trace(
        leaf, reshape([1.0, 2.0, 3.0], 1, :))

    @test trace[2].shape == 2.0
    # `reconstruct` was handed a view into `trace.draws`, not a copy.
    @test seen[] isa SubArray
    @test parent(seen[]) === trace.draws
end

@testitem "PosteriorTrace: slicing returns nchains = 1" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    draws = reshape(collect(1.0:6.0), 1, 6)
    trace = DistributionsInference.draws_to_trace(leaf, draws; nchains = 2)

    sliced = trace[2:4]
    @test sliced isa DistributionsInference.PosteriorTrace
    @test sliced.nchains == 1
    @test [t.shape for t in sliced] == [2.0, 3.0, 4.0]

    # A single-element vector index still slices (not the scalar method).
    single = trace[[1]]
    @test single isa DistributionsInference.PosteriorTrace
    @test single.nchains == 1
    @test length(single) == 1
end

# `T` is a phantom type parameter (it names no field of the struct), so it
# cannot be inferred from arguments — see the #90 sign-off's "adversarial
# review" correction. These are exactly the headline examples that correction
# found broken in the issue's own naive sketch: a `PosteriorTrace` built with
# `PosteriorTrace{T,O,D,S}(...)` directly (bypassing the outer constructor
# that computes `T`) fails to resolve `T` from its arguments, and the issue's
# own slicing call reproduced that failure. `draws_to_trace` and slicing
# `getindex` both route through the one outer constructor instead, so all
# three of these must actually run, not just type-check by inspection.
@testitem "PosteriorTrace: UnitRange, arbitrary index vector and Bool mask all slice" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    draws = reshape(collect(1.0:6.0), 1, 6)
    trace = DistributionsInference.draws_to_trace(leaf, draws)

    by_range = trace[1:3]
    @test by_range isa DistributionsInference.PosteriorTrace
    @test [t.shape for t in by_range] == [1.0, 2.0, 3.0]

    by_index = trace[[1, 3, 5]]
    @test by_index isa DistributionsInference.PosteriorTrace
    @test [t.shape for t in by_index] == [1.0, 3.0, 5.0]

    mask = [true, false, true, false, true, false]
    by_mask = trace[mask]
    @test by_mask isa DistributionsInference.PosteriorTrace
    @test [t.shape for t in by_mask] == [1.0, 3.0, 5.0]
end

@testitem "PosteriorTrace: is a well-behaved AbstractVector" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [2.1, 2.4, 2.0, 2.6]
    trace = DistributionsInference.draws_to_trace(leaf, reshape(values, 1, :))

    @test trace isa AbstractVector
    @test size(trace) == (4,)
    @test eltype(trace) == typeof(leaf)
    collected = collect(trace)
    @test collected isa Vector{typeof(leaf)}
    @test [c.shape for c in collected] == values
end

@testitem "PosteriorTrace: stats is a type parameter, not a bare NamedTuple field" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [2.1, 2.4, 2.0, 2.6]
    trace = DistributionsInference.draws_to_trace(leaf, reshape(values, 1, :))

    # An undecorated `stats::NamedTuple` field is abstract, so its declared
    # type would not be concrete even though every value stored in it is; a
    # fourth type parameter `S` (with `stats::S`) is what makes it concrete.
    S = fieldtype(typeof(trace), :stats)
    @test isconcretetype(S)
    @test S !== NamedTuple  # not the bare abstract type
    @test trace.stats isa S
end

@testitem "filter(trace): preserves trace-ness, unlike a plain Vector" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    draws = reshape(collect(1.0:6.0), 1, 6)
    trace = DistributionsInference.draws_to_trace(leaf, draws)

    kept = filter(t -> t.shape > 3.0, trace)
    @test kept isa DistributionsInference.PosteriorTrace
    @test [t.shape for t in kept] == [4.0, 5.0, 6.0]
    @test kept.nchains == 1

    # A trace `filter` keeps is still usable by the transforms built on
    # `PosteriorTrace`, which is the whole point of not degrading to a
    # `Vector` (e.g. dropping warmup or divergent draws before summarising).
    @test DistributionsInference.point_estimate(kept).shape ≈ 5.0
end

@testitem "parameter_draws: per-parameter unreduced draws, keyed by name" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [1.0, 2.0, 3.0, 4.0]
    trace = DistributionsInference.draws_to_trace(leaf, reshape(values, 1, :))

    nt = DistributionsInference.parameter_draws(trace)
    @test nt isa NamedTuple
    @test keys(nt) == (:shape,)
    @test collect(nt.shape) == values
end

@testitem "parameter_draws: a duplicate estimated name errors clearly" begin
    using DistributionsInference, Distributions

    rows = [
        (name = :shape, value = 2.0, prior = LogNormal(0.0, 0.2),
            support = (0.0, Inf)),
        (name = :shape, value = 1.0, prior = LogNormal(0.0, 0.2),
            support = (0.0, Inf))]
    trace = DistributionsInference.draws_to_trace(
        rows, [1.0 2.0 3.0; 4.0 5.0 6.0])

    err = try
        DistributionsInference.parameter_draws(trace)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("duplicate", err.msg)
    @test occursin("shape", err.msg)
end

@testitem "point_estimate(trace): summary selection, default mean" setup=[ToyFixture] begin
    using Statistics: median

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [1.0, 2.0, 3.0, 4.0]
    trace = DistributionsInference.draws_to_trace(leaf, reshape(values, 1, :))

    @test DistributionsInference.point_estimate(trace).shape ≈ 2.5
    @test DistributionsInference.point_estimate(trace; summary = median).shape ≈
          2.5
    @test DistributionsInference.point_estimate(trace; summary = maximum).shape ≈
          4.0
    @test DistributionsInference.point_estimate(trace).scale == leaf.scale
end

@testitem "point_estimate: the chain and trace methods dispatch without ambiguity" setup=[ToyFixture] begin
    using FlexiChains: FlexiChains

    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [1.0, 2.0, 3.0, 4.0]
    chain = DistributionsInference._to_flexichain(leaf, reshape(values, 1, :))
    trace = DistributionsInference.draws_to_trace(leaf, reshape(values, 1, :))

    from_chain = DistributionsInference.point_estimate(leaf, chain)
    from_trace = DistributionsInference.point_estimate(trace)
    @test from_chain.shape ≈ from_trace.shape ≈ 2.5
end

@testitem "trace_to_distribution: refuses a non-Distribution eltype, naming it" setup=[ToyFixture] begin
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(log(2.0), 0.2))
    values = [1.0, 2.0, 3.0, 4.0]
    trace = DistributionsInference.draws_to_trace(leaf, reshape(values, 1, :))
    @test eltype(trace) == ToyGammaLeaf  # not a `Distribution`

    err = try
        DistributionsInference.trace_to_distribution(trace)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("ToyGammaLeaf", err.msg)
    @test occursin("Distribution", err.msg)
end

@testitem "trace_to_distribution: an equally-weighted mixture over the draws" setup=[TraceDistFixture] begin
    template = GammaShapeTemplate(2.0, LogNormal(log(2.0), 0.2))
    shapes = [1.0, 2.0, 3.0, 4.0]
    trace = DistributionsInference.draws_to_trace(
        template, reshape(shapes, 1, :))
    @test eltype(trace) <: Distribution

    mixture = DistributionsInference.trace_to_distribution(trace)
    @test mixture isa MixtureModel
    @test ncomponents(mixture) == 4
    @test probs(mixture) ≈ fill(0.25, 4)  # equally weighted

    # Gamma(shape, 1.0) has mean `shape`, so the mixture mean is the mean of
    # the shapes.
    @test mean(mixture) ≈ mean(shapes)
    @test pdf(mixture, 1.0) > 0
    @test rand(mixture) isa Real
end

@testitem "mean(::PosteriorTrace) refuses, naming both alternatives" setup=[TraceDistFixture] begin
    template = GammaShapeTemplate(2.0, LogNormal(log(2.0), 0.2))
    shapes = [1.0, 2.0, 3.0, 4.0]
    trace = DistributionsInference.draws_to_trace(
        template, reshape(shapes, 1, :))

    err = try
        mean(trace)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("trace_to_distribution", err.msg)
    @test occursin("point_estimate", err.msg)

    # Broadcasting is unaffected: `mean.(trace)` maps `mean` over each
    # reconstructed element (each `Gamma(shape, 1.0)`), never calling
    # `mean(trace)` itself.
    @test mean.(trace) ≈ shapes
end

@testitem "every ambiguous reduction refuses consistently, not just mean" setup=[TraceDistFixture] begin
    using Statistics: median, std, var, quantile

    # `sum`, `median`, `std`, `var` and non-broadcast `quantile` are exactly
    # as ambiguous as `mean` (trace_to_distribution vs point_estimate), and
    # left undefended would fail two frames deeper inside Statistics/Base
    # (a `+`/`isless` MethodError naming the reconstructed element type,
    # `Gamma{Float64}`, rather than the trace) instead of failing here with a
    # message that names both alternatives.
    template = GammaShapeTemplate(2.0, LogNormal(log(2.0), 0.2))
    shapes = [1.0, 2.0, 3.0, 4.0]
    trace = DistributionsInference.draws_to_trace(
        template, reshape(shapes, 1, :))

    calls = [
        (:sum, () -> sum(trace)),
        (:median, () -> median(trace)),
        (:std, () -> std(trace)),
        (:var, () -> var(trace)),
        (:quantile, () -> quantile(trace, 0.9))]
    for (name, call) in calls
        err = try
            call()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("trace_to_distribution", err.msg)
        @test occursin("point_estimate", err.msg)
        @test occursin(String(name), err.msg)
    end

    # Broadcasting every one of them is unaffected, same as `mean.(trace)`.
    @test quantile.(trace, 0.9) ≈ quantile.(Gamma.(shapes, 1.0), 0.9)
end
