# Default-prior assembly over the fit protocol: `default_prior`'s per-row
# heuristic (mirroring ComposedDistributions', by name-classification not
# support) and `distribution_priors`'s override/attached-prior/default
# precedence (CD#195/DI#20).

@testitem "default_prior: classifies by the row's own name, not just support" begin
    using DistributionsInference, Distributions

    # A positive-by-name parameter gets a positive-truncated default even
    # though its `support` here is reported unbounded (mirrors CD: a
    # location-family delay's `sigma` still lives on the positive line).
    shape_row = (name = :shape, value = 2.0, prior = nothing,
        support = (-Inf, Inf))
    p = DistributionsInference.default_prior(shape_row)
    @test p isa Truncated
    @test p.lower == 0.0

    # A location-by-name parameter gets an unconstrained default even under
    # a positive-support row.
    mu_row = (name = :mu, value = 1.0, prior = nothing, support = (0.0, Inf))
    @test DistributionsInference.default_prior(mu_row) isa Normal

    # A [0, 1]-support row (a simplex/probability parameter) -> Uniform(0, 1).
    prob_row = (name = :branch_probs, value = 0.5, prior = nothing,
        support = (0.0, 1.0))
    @test DistributionsInference.default_prior(prob_row) == Uniform(0, 1)

    # An unmapped name falls back to the support: non-negative -> truncated.
    unmapped_pos = (name = :thin, value = 2.0, prior = nothing,
        support = (0.0, Inf))
    @test DistributionsInference.default_prior(unmapped_pos) isa Truncated
    unmapped_free = (name = :offset, value = 2.0, prior = nothing,
        support = (-Inf, Inf))
    @test DistributionsInference.default_prior(unmapped_free) isa Normal

    # A dotted name is classified by its own (last-segment) name, not the
    # dotted path.
    dotted_row = (name = Symbol("onset.shape"), value = 2.0, prior = nothing,
        support = (-Inf, Inf))
    @test DistributionsInference.default_prior(dotted_row) isa Truncated
end

@testitem "distribution_priors: override, attached, then default precedence" begin
    using DistributionsInference, Distributions

    rows = [
        (name = :shape, value = 2.0, prior = nothing, support = (0.0, Inf)),
        (name = :scale, value = 1.0, prior = LogNormal(0.0, 0.1),
            support = (0.0, Inf)),
        (name = :mu, value = 0.5, prior = nothing, support = (-Inf, Inf))]

    # No overrides: the attached prior wins for `scale`, the default fires
    # for `shape`/`mu`.
    priored = DistributionsInference.distribution_priors(rows)
    @test priored[1].prior isa Truncated   # shape: default, positive-by-name
    @test priored[2].prior == LogNormal(0.0, 0.1)   # scale: attached, kept
    @test priored[3].prior isa Normal   # mu: default, location-by-name

    # A `priors` override for `shape` wins over the default.
    custom = LogNormal(log(2.0), 0.05)
    overridden = DistributionsInference.distribution_priors(
        rows; priors = Dict(:shape => custom))
    @test overridden[1].prior == custom
    # An override also wins over an already-attached prior.
    overridden2 = DistributionsInference.distribution_priors(
        rows; priors = Dict(:scale => custom))
    @test overridden2[2].prior == custom

    # The rest of each row is untouched (only `prior` changes).
    @test priored[1].name == rows[1].name
    @test priored[1].value == rows[1].value
    @test priored[1].support == rows[1].support

    # The result is directly usable as a fittable object (the bare-row-vector
    # identity): reconstruct at the new priors' estimated dimension.
    @test DistributionsInference.flat_dimension(priored) == 3   # all now priored
end

@testitem "distribution_priors: a custom default function is honoured" begin
    using DistributionsInference, Distributions

    rows = [(name = :shape, value = 2.0, prior = nothing,
        support = (0.0, Inf))]
    always_flat(row) = Uniform(0, 10)
    priored = DistributionsInference.distribution_priors(rows; default = always_flat)
    @test priored[1].prior == Uniform(0, 10)
end
