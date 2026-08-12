# DistributionsInference × ModifiedDistributions (DI#17): `parameter_rows`/
# `reconstruct` for a STANDALONE modifier distribution. Load-bearing checks:
# a modifier's fixed structure (Affine's scale/shift, Weighted's weight,
# Modified's effect/link) is peeled through and NOT reported as a row —
# mirroring ComposedDistributions' own leaf-protocol precedent
# (`extra_leaf_params` also hides them) — while a `thin` factor IS reported as
# an extra row; nested modifier layers are peeled and their extras collected
# in encounter order; every row is fixed (`flat_dimension == 0`); `reconstruct`
# is the identity at the empty vector and rejects a non-empty one; and the
# blessed generic `with_priors` + custom-`loglik` pattern (the exact
# mechanism DI's own core test suite demonstrates for a bare `Gamma`) fits a
# standalone modifier's native parameter end-to-end.

@testsnippet ModifiedFixture begin
    using DistributionsInference, Distributions, ModifiedDistributions
    using ModifiedDistributions: affine, weight, thin, modify
end

@testitem "ModifiedDistributions extension loads" setup=[ModifiedFixture] begin
    @test Base.get_extension(DistributionsInference,
        :DistributionsInferenceModifiedDistributionsExt) !== nothing
end

@testitem "parameter_rows: Affine hides scale/shift, reports the inner leaf" setup=[ModifiedFixture] begin
    d = affine(Gamma(2.0, 1.0); scale = 2.0, shift = 1.0)
    rows = DistributionsInference.parameter_rows(d)
    @test length(rows) == 2
    @test rows[1].name == :shape && rows[1].value == 2.0
    @test rows[2].name == :scale && rows[2].value == 1.0
    @test all(row -> row.prior === nothing, rows)
    sup = (minimum(Gamma(2.0, 1.0)), maximum(Gamma(2.0, 1.0)))
    @test all(row -> row.support == sup, rows)
    @test DistributionsInference.flat_dimension(d) == 0
end

@testitem "parameter_rows: Weighted hides the weight" setup=[ModifiedFixture] begin
    d = weight(Normal(2.0, 1.0), 5.0)
    rows = DistributionsInference.parameter_rows(d)
    @test length(rows) == 2
    @test [row.name for row in rows] == [:mu, :sigma]
    @test [row.value for row in rows] == [2.0, 1.0]
end

@testitem "parameter_rows: thin reports the extra :thin row" setup=[ModifiedFixture] begin
    d = thin(Gamma(2.0, 1.0), 0.3)
    rows = DistributionsInference.parameter_rows(d)
    @test length(rows) == 3
    @test [row.name for row in rows] == [:shape, :scale, :thin]
    @test rows[3].value == 0.3
    @test rows[3].support == (0.0, 1.0)
    # Reported (inventoried), but not automatically ESTIMATED — every row is
    # fixed by default (see this file's header note); flat_dimension stays 0.
    @test DistributionsInference.flat_dimension(d) == 0
    @test length(DistributionsInference.estimated_rows(d)) == 0
end

@testitem "parameter_rows: Modified hides the hazard effect/link" setup=[ModifiedFixture] begin
    d = modify(LogNormal(1.5, 0.5), 0.2; link = :identity)
    rows = DistributionsInference.parameter_rows(d)
    @test length(rows) == 2
    @test [row.name for row in rows] == [:mu, :sigma]
end

@testitem "parameter_rows: nested modifiers peel through and collect extras" setup=[ModifiedFixture] begin
    # thin(affine(Gamma(...))): the affine's scale/shift stay hidden, the
    # thin factor is the one reported extra, and the native rows come from
    # the innermost Gamma.
    d = thin(affine(Gamma(2.0, 1.0); scale = 2.0, shift = 1.0), 0.4)
    rows = DistributionsInference.parameter_rows(d)
    @test length(rows) == 3
    @test [row.name for row in rows] == [:shape, :scale, :thin]
    @test [row.value for row in rows] == [2.0, 1.0, 0.4]
end

@testitem "parameter_rows: double-thin diverges from the tree-context report (#30)" setup=[ModifiedFixture] begin
    # thin(thin(Gamma(...))): the standalone path collects BOTH thin factors
    # but collides on the name :thin (four rows, two named :thin), while
    # ComposedDistributions' own tree-context `params_table` (reached via
    # ModifiedDistributionsComposedDistributionsExt's `extra_leaf_params`)
    # returns early at the first ThinOp and silently drops the inner factor
    # (three rows). Neither behaviour is right, and a leaf should report the
    # same parameter set standalone as inside a tree — captured broken here
    # rather than fixed, since the fix (e.g. indexing repeated extra names
    # thin_1/thin_2) must land on both packages in lockstep (#30).
    using ComposedDistributions: compose, params_table

    d = thin(thin(Gamma(2.0, 1.0), 0.3), 0.5)

    standalone_rows = DistributionsInference.parameter_rows(d)
    @test length(standalone_rows) == 4
    @test [row.name for row in standalone_rows] ==
          [:shape, :scale, :thin, :thin]
    @test [row.value for row in standalone_rows] == [2.0, 1.0, 0.5, 0.3]

    tree = compose((leaf = d,))
    tbl = params_table(tree)
    @test length(tbl.edge) == 3
    @test collect(tbl.param) == [:shape, :scale, :thin]
    @test collect(tbl.value) == [2.0, 1.0, 0.5]

    # The property that should hold and currently does not: a leaf reports
    # the same parameter count standalone as inside a tree.
    @test_broken length(standalone_rows) == length(tbl.edge)
end

@testitem "reconstruct: identity at the empty vector, rejects non-empty" setup=[ModifiedFixture] begin
    d = affine(Gamma(2.0, 1.0); scale = 2.0, shift = 1.0)
    @test DistributionsInference.reconstruct(d, Float64[]) === d
    @test_throws DimensionMismatch DistributionsInference.reconstruct(d, [1.0])
end

@testitem "end-to-end: with_priors + a custom loglik fits a standalone modifier" setup=[ModifiedFixture] begin
    using Random

    # An Affine over a Gamma: the affine's own scale/shift stay fixed
    # structure (hidden from parameter_rows entirely, closed over in the
    # loglik below), while BOTH the Gamma's native parameters are estimated —
    # with_priors' documented "estimate everything" default, via the
    # generic bare-row-vector path (parameter_rows(d) alone makes this
    # reachable; no bespoke "make this Affine estimated" mechanism is needed,
    # following the pattern shipped in DistributionsInference#23.
    affine_scale, affine_shift = 1.5, 0.5
    true_shape, true_scale = 3.0, 1.2
    rng = Random.Xoshiro(1)
    data = rand(
        rng, affine(Gamma(true_shape, true_scale);
            scale = affine_scale, shift = affine_shift), 800)

    template = affine(Gamma(2.0, 1.0); scale = affine_scale, shift = affine_shift)
    rows = DistributionsInference.with_priors(
        template; priors = Dict(:shape => LogNormal(log(2.0), 0.5)))
    @test rows[1].name == :shape && rows[1].prior isa LogNormal
    # :scale (the Gamma's own, not the Affine's) gets a support-derived
    # default prior — with_priors fills EVERY row, not just the
    # overridden one.
    @test rows[2].name == :scale && rows[2].prior !== nothing
    @test DistributionsInference.flat_dimension(rows) == 2

    # A loglik that rebuilds the concrete Affine(Gamma(...)) from the rows'
    # values, closing over the fixed affine scale/shift (not part of the row
    # set at all).
    loglik(built, ds) = sum(
        y -> logpdf(
            affine(Gamma(built[1].value, built[2].value);
                scale = affine_scale, shift = affine_shift),
            y),
        ds)
    prob = DistributionsInference.distribution_to_logdensity(rows, data; loglik = loglik)

    function metropolis(prob, x0; n = 5000, step = 0.1, rng = rng)
        x = copy(x0)
        lp = DistributionsInference.logdensity(prob, x)
        draws = Vector{Vector{Float64}}(undef, n)
        for i in 1:n
            prop = x .+ step .* randn(rng, length(x))
            if any(<=(0), prop)
                draws[i] = copy(x)
                continue
            end
            lp_prop = DistributionsInference.logdensity(prob, prop)
            if log(rand(rng)) < lp_prop - lp
                x, lp = prop, lp_prop
            end
            draws[i] = copy(x)
        end
        return draws
    end

    draws = metropolis(prob, [2.0, 1.0])
    burn = draws[1501:end]
    post_mean = sum(burn) / length(burn)
    @test abs(post_mean[1] - true_shape) < 0.6
    @test abs(post_mean[2] - true_scale) < 0.6
end
