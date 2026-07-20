# DistributionsInference × ComposedDistributions (DI#5): `parameter_rows`,
# `estimated_rows`, `flat_dimension`, `reconstruct` and `extra_logprior` for a
# composed distribution over CD's public codec. The load-bearing checks: the
# dotted-name row mapping (including the `CentredPoolPrior` -> `nothing`
# translation), that `estimated_rows`/`flat_dimension` agree with CD's own
# generated `flat_dimension` across plain/pooled (centred and non-centred)/
# `Resolve`-Dirichlet/shared-tag trees, that `reconstruct` matches CD's own
# `reconstruct` exactly, that `logdensity` through this extension agrees with
# CD's own (pre-extension) `as_logdensity`/`logdensity` for the same tree and
# data, that `extra_logprior` scores a centred-pool term against an
# independent hand computation, and that the generic `to_flexichain`/`readback`
# machinery — no ComposedDistributions-specific code needed — round-trips a
# pooled, shared-tag tree.

@testsnippet ComposedFixture begin
    using DistributionsInference, Distributions, ComposedDistributions
    using ComposedDistributions: compose, uncertain, pool, resolve, shared,
                                 update, params_table

    # A plain two-leaf tree, one leaf uncertain (shape), one fully fixed.
    plain_tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
        admit_death = LogNormal(0.5, 0.4)))

    # Non-centred pooling: three districts sharing a default LogNormal
    # population (2 hyperparameters + 3 latents = 5 estimated, per Pool.jl's
    # own docstring example).
    noncentred_tree = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        south = uncertain(Gamma(2.0, 1.0); shape = pool(:district))))

    # Centred pooling: a fixed (non-uncertain) Beta population, so there are
    # no hyperparameter rows, only the two members' own centred latents.
    centred_pop = Beta(2.0, 3.0)
    centred_tree = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:region, centred_pop)),
        south = uncertain(Gamma(2.0, 1.0); shape = pool(:region, centred_pop))))

    # A Resolve node with an uncertain (Dirichlet) simplex: the codec
    # estimates the K-1 stick-breaking coordinates, not the probabilities
    # directly.
    fixed_resolve = resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => Gamma(2.0, 1.5))
    resolve_tree = update(fixed_resolve, (branch_probs = Dirichlet(ones(2)),))

    # A shared tag tied across two branches: one uncertain parameter,
    # inventoried and estimated ONCE under the tag edge.
    shared_leaf = shared(:rate, uncertain(Gamma(2.0, 1.0);
        shape = LogNormal(log(2.0), 0.2)))
    shared_tree = compose((a = shared_leaf, b = shared_leaf))
end

@testitem "ComposedDistributions extension loads" setup=[ComposedFixture] begin
    @test Base.get_extension(DistributionsInference,
        :DistributionsInferenceComposedDistributionsExt) !== nothing
end

@testitem "parameter_rows: dotted edge.param names, prior passed through" setup=[ComposedFixture] begin
    rows = DistributionsInference.parameter_rows(plain_tree)
    tbl = params_table(plain_tree)
    @test length(rows) == length(tbl.edge)
    for i in eachindex(rows)
        @test rows[i].name == Symbol(tbl.edge[i], ".", tbl.param[i])
        @test rows[i].value == tbl.value[i]
        @test rows[i].support == tbl.support[i]
        @test rows[i].prior === tbl.prior[i]
    end
    # The one uncertain row (onset_admit.shape) is estimated; the fixed leaf's
    # rows are not.
    est = DistributionsInference.estimated_rows(plain_tree)
    @test length(est) == 1
    @test only(est).name == Symbol("onset_admit.shape")
end

@testitem "flat_dimension/estimated_rows agree with CD across tree shapes" setup=[ComposedFixture] begin
    for tree in (plain_tree, noncentred_tree, centred_tree, resolve_tree, shared_tree)
        n = ComposedDistributions.flat_dimension(tree)
        @test DistributionsInference.flat_dimension(tree) == n
        @test length(DistributionsInference.estimated_rows(tree)) == n
    end
    # Concrete counts from Pool.jl's own docstring example (non-centred: 2
    # hyperparameters + 3 latents) and the centred case (2 members, no
    # hyperparameters since the population is fixed).
    @test DistributionsInference.flat_dimension(noncentred_tree) == 5
    @test DistributionsInference.flat_dimension(centred_tree) == 2
    # A K=2 Dirichlet Resolve estimates one stick coordinate.
    @test DistributionsInference.flat_dimension(resolve_tree) == 1
    # A shared tag is inventoried once, not once per occurrence.
    @test DistributionsInference.flat_dimension(shared_tree) == 1
end

@testitem "estimated_rows: a centred-pooled row's DI prior is nothing" setup=[ComposedFixture] begin
    est = DistributionsInference.estimated_rows(centred_tree)
    @test length(est) == 2
    @test all(row -> row.prior === nothing, est)
    @test Set(row.name for row in est) ==
          Set(Symbol.(["north.shape", "south.shape"]))
end

@testitem "reconstruct matches ComposedDistributions.reconstruct exactly" setup=[ComposedFixture] begin
    for (tree, x) in ((plain_tree, [3.0]), (noncentred_tree, fill(0.1, 5)),
        (centred_tree, [0.3, 0.6]), (resolve_tree, [0.2]), (shared_tree, [2.5]))
        @test DistributionsInference.reconstruct(tree, x) ==
              ComposedDistributions.reconstruct(tree, x)
    end
end

@testitem "logdensity agrees with CD's own as_logdensity/logdensity" setup=[ComposedFixture] begin
    data = [[0.5, 2.0], [1.0, 3.0]]
    prob = DistributionsInference.as_logdensity(plain_tree, data)
    cd_prob = ComposedDistributions.as_logdensity(plain_tree, data)

    for x in ([1.5], [2.0], [3.2])
        @test DistributionsInference.logdensity(prob, x) ≈
              ComposedDistributions.logdensity(cd_prob, x)
    end
end

@testitem "logdensity agrees with CD for a non-centred pooled tree" setup=[ComposedFixture] begin
    data = [[0.5, 2.0, 1.2], [1.0, 3.0, 0.8], [0.9, 1.8, 1.1]]
    prob = DistributionsInference.as_logdensity(noncentred_tree, data)
    cd_prob = ComposedDistributions.as_logdensity(noncentred_tree, data)

    x = [0.1, 0.2, -0.3, 0.4, 0.5]
    @test DistributionsInference.logdensity(prob, x) ≈
          ComposedDistributions.logdensity(cd_prob, x)
end

@testitem "extra_logprior: centred-pool term matches a hand computation" setup=[ComposedFixture] begin
    x = [0.3, 0.6]
    reconstructed = DistributionsInference.reconstruct(centred_tree, x)
    expected = logpdf(centred_pop, x[1]) + logpdf(centred_pop, x[2])
    @test DistributionsInference.extra_logprior(centred_tree, reconstructed, x) ≈ expected

    # A tree with no centred pooling contributes nothing extra.
    @test DistributionsInference.extra_logprior(
        plain_tree, DistributionsInference.reconstruct(plain_tree, [2.0]), [2.0]) == 0.0
end

@testitem "logdensity agrees with CD for a centred pooled tree (incl. extra_logprior)" setup=[ComposedFixture] begin
    data = [[0.4, 0.7], [0.6, 0.5]]
    prob = DistributionsInference.as_logdensity(centred_tree, data)
    cd_prob = ComposedDistributions.as_logdensity(centred_tree, data)

    x = [0.3, 0.6]
    @test DistributionsInference.logdensity(prob, x) ≈
          ComposedDistributions.logdensity(cd_prob, x)
end

@testitem "readback: the generic FlexiChains machinery round-trips a pooled, shared tree" setup=[ComposedFixture] begin
    using FlexiChains

    tree = noncentred_tree
    n = DistributionsInference.flat_dimension(tree)
    # Distinct per-dimension offsets, so a column-ordering bug (a row read
    # back under the wrong dotted name) would be caught rather than masked by
    # every dimension sharing one value.
    draws = [[0.05 * i + 0.01 * j for j in 1:n] for i in 1:20]
    chain = DistributionsInference.to_flexichain(tree, draws)

    fitted = DistributionsInference.readback(tree, chain)
    expected_x = [mean(d[i] for d in draws) for i in 1:n]
    @test fitted == ComposedDistributions.reconstruct(tree, expected_x)

    all_fitted = DistributionsInference.readback_draws(tree, chain)
    @test length(all_fitted) == length(draws)
    @test all_fitted[end] == ComposedDistributions.reconstruct(tree, draws[end])
end

@testitem "as_turing rejects a centred-pooled tree, mirroring CD's own guard" setup=[ComposedFixture] begin
    using DynamicPPL

    data = [[0.4, 0.7], [0.6, 0.5]]
    @test_throws ArgumentError DistributionsInference.as_turing(centred_tree, data)
end

# The three real-chain round trips below (shared tag, Dirichlet stick
# coordinate, non-centred pool) were ComposedDistributions' own
# `test/composers/turing_ext.jl` before that package dropped its `as_turing`/
# `chain_to_params` surface in favour of this package's generic one (CD#221,
# CD#233): they exercise the trickiest ordering/dedup cases (a tied leaf
# sampled once but read back onto every occurrence; a K-1 stick-breaking
# simplex; a pooled member's `.z` latent) through an ACTUAL `sample(...,
# NUTS(), ...)` chain, not a hand-built one, so a table/codec ordering
# regression that only shows up under real sampling is still caught. Ported
# here rather than dropped, so the coverage does not thin when CD removes its
# copy.

@testitem "as_turing round-trip: shared-tag readback lands on the right leaf" setup=[
    ComposedFixture] begin
    using DynamicPPL, Turing, Random

    data = [[0.5, 0.6], [1.0, 0.9], [0.8, 0.7]]
    model = DistributionsInference.as_turing(shared_tree, data)

    Random.seed!(23)
    chain = sample(model, NUTS(), 200; progress = false)

    # Exactly ONE site for the tie, at the tag's dotted name — not one per
    # occurrence (`d.a.shape`/`d.b.shape` never appear).
    fitted = DistributionsInference.readback(shared_tree, chain)
    @test ComposedDistributions.event(fitted, :a) ==
          ComposedDistributions.event(fitted, :b)
    @test !ComposedDistributions.has_uncertain(fitted)
end

@testitem "as_turing round-trip: Dirichlet branch_probs stick coordinate" setup=[
    ComposedFixture] begin
    using DynamicPPL, Turing, Random, Distributions

    data = [0.8, 1.5, 2.2, 0.6]
    model = DistributionsInference.as_turing(resolve_tree, data)

    Random.seed!(7)
    chain = sample(model, NUTS(), 200; progress = false)

    fitted = DistributionsInference.readback(resolve_tree, chain)
    @test !ComposedDistributions.has_uncertain(fitted)
    p = collect(values(Distributions.probs(fitted)))
    @test sum(p) ≈ 1.0
end

@testitem "as_turing round-trip: non-centred pooled tree" setup=[ComposedFixture] begin
    using DynamicPPL, Turing, Random

    data = [[0.5, 2.0, 1.2], [1.0, 3.0, 0.8], [0.9, 1.8, 1.1]]
    model = DistributionsInference.as_turing(noncentred_tree, data)

    Random.seed!(13)
    chain = sample(model, NUTS(), 200; progress = false)

    fitted = DistributionsInference.readback(noncentred_tree, chain)
    @test !ComposedDistributions.has_uncertain(fitted)
    @test ComposedDistributions.event(fitted, :north) isa Distributions.Gamma
    @test ComposedDistributions.event(fitted, :east) isa Distributions.Gamma
    @test ComposedDistributions.event(fitted, :south) isa Distributions.Gamma
end
