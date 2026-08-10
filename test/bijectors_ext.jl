# DistributionsInference × Bijectors extension: `to_constrained` maps an
# unconstrained flat vector to the constrained estimated parameters plus a
# log-Jacobian, built per row from `FitLogDensity`'s `flat_priors`.

@testitem "Bijectors extension loads" begin
    using Bijectors

    @test Base.get_extension(DistributionsInference,
        :DistributionsInferenceBijectorsExt) !== nothing
end

@testitem "to_constrained: closed-form identity for a LogNormal-Gamma row" setup=[ToyFixture] begin
    using Bijectors

    # A LogNormal is exp of a Normal, so `logpdf(LogNormal(mu, sigma), exp(z))
    # + z` collapses to `logpdf(Normal(mu, sigma), z)`: a closed-form oracle
    # independent of the transform machinery.
    mu, sigma = 0.4, 0.3
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(mu, sigma))
    data = Float64[]
    zero_lik(d, ds) = 0.0
    prob = DistributionsInference.distribution_to_logdensity(leaf, data; loglik = zero_lik)

    for z0 in (-0.6, 0.0, 1.1)
        x, logjac = DistributionsInference.to_constrained(prob, [z0])
        @test x[1] ≈ exp(z0)
        @test DistributionsInference.logdensity(prob, x) + logjac ≈
              logpdf(Normal(mu, sigma), z0)
    end
end

@testitem "to_constrained: per-row transform across a multi-parameter object" setup=[TuringFixture] begin
    using Bijectors: Bijectors, bijector, inverse, with_logabsdet_jacobian

    leaf = TwoParamLeaf(2.0, 1.0)
    data = Float64[]
    zero_lik(d, ds) = 0.0
    prob = DistributionsInference.distribution_to_logdensity(leaf, data; loglik = zero_lik)
    n = DistributionsInference.flat_dimension(leaf)
    @test n == 2 == length(prob.flat_priors)

    z = [-0.4, 0.7]
    x, logjac = DistributionsInference.to_constrained(prob, z)
    @test length(x) == n

    per_row = map(eachindex(z)) do i
        binv = inverse(bijector(prob.flat_priors[i]))
        with_logabsdet_jacobian(binv, z[i])
    end
    @test x ≈ [xi for (xi, _) in per_row]
    @test logjac ≈ sum(last, per_row)

    for i in eachindex(x)
        @test insupport(prob.flat_priors[i], x[i])
    end

    @test_throws DimensionMismatch DistributionsInference.to_constrained(
        prob, z[1:(end - 1)])
end

@testitem "to_constrained: mixed prior families give mixed links" begin
    using DistributionsInference, Distributions
    using Bijectors: Bijectors, bijector, inverse, with_logabsdet_jacobian

    # A row set mixing prior families makes `FitLogDensity.flat_priors`
    # abstractly typed, so each row's bijector is a different concrete type:
    # an identity link for the unconstrained `Normal` on `mu`, a log link for
    # the positive-truncated `Normal` on `sigma`, and a logit link for the
    # `[0, 1]`-supported `Beta` on `p`. The uniform-family object above
    # cannot catch a regression that only shows up once the transforms differ
    # per row (DI#33).
    struct MixedLinkLeaf{M <: Real, S <: Real, P <: Real}
        mu::M
        sigma::S
        p::P
    end

    function DistributionsInference.parameter_rows(d::MixedLinkLeaf)
        return [
            (name = :mu, value = d.mu, prior = Normal(0.0, 2.0),
                support = (-Inf, Inf)),
            (name = :sigma, value = d.sigma,
                prior = truncated(Normal(1.0, 1.0); lower = 0.0),
                support = (0.0, Inf)),
            (name = :p, value = d.p, prior = Beta(2.0, 2.0),
                support = (0.0, 1.0))]
    end

    function DistributionsInference.reconstruct(
            d::MixedLinkLeaf, x::AbstractVector)
        return MixedLinkLeaf(x[1], x[2], x[3])
    end

    leaf = MixedLinkLeaf(0.0, 1.0, 0.4)
    zero_lik(d, ds) = 0.0
    prob = DistributionsInference.distribution_to_logdensity(
        leaf, Float64[]; loglik = zero_lik)
    @test DistributionsInference.flat_dimension(leaf) == 3

    # Chosen so the three constrained values land in disjoint places: `mu` is
    # negative, `sigma` is above 1, and `p` is inside `(0, 1)`. A permuted
    # `to_constrained` result then violates a support check, which a `z`
    # whose constrained image happens to be positive and below 1 in every row
    # could not detect.
    z = [-1.5, 2.0, 0.6]
    x, logjac = DistributionsInference.to_constrained(prob, z)

    # Each link checked against its own closed form, not against Bijectors'
    # generic machinery: identity, exp, logistic.
    @test x[1] ≈ z[1]
    @test x[2] ≈ exp(z[2])
    @test x[3] ≈ 1 / (1 + exp(-z[3]))
    @test logjac ≈ 0.0 + z[2] + (-z[3] - 2 * log1p(exp(-z[3])))

    # And the change-of-variables identity a sampler relies on still holds
    # row by row across the three different links.
    target = sum(eachindex(z)) do i
        logpdf(Bijectors.transformed(prob.flat_priors[i]), z[i])
    end
    @test DistributionsInference.logdensity(prob, x) + logjac ≈ target

    # Each field takes its own row of the constrained vector, checked against
    # the already-verified `x` rather than against a support every row could
    # satisfy, so a row-order slip in `to_constrained` fails here.
    rebuilt = DistributionsInference.reconstruct(leaf, x)
    @test rebuilt.mu ≈ x[1]
    @test rebuilt.sigma ≈ x[2]
    @test rebuilt.p ≈ x[3]
    @test rebuilt.mu < 0
    @test rebuilt.sigma > 1
    @test 0 < rebuilt.p < 1
end

@testitem "to_constrained: a 0-estimated object round-trips at the empty vector" setup=[ToyFixture] begin
    using Bijectors

    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    data = Float64[]
    zero_lik(d, ds) = 0.0
    prob = DistributionsInference.distribution_to_logdensity(fixed_leaf, data; loglik = zero_lik)
    @test DistributionsInference.flat_dimension(fixed_leaf) == 0

    x, logjac = DistributionsInference.to_constrained(prob, Float64[])
    @test isempty(x)
    @test logjac == 0.0
end

@testitem "to_constrained: logdensity(prob, x) + logjac is the unconstrained target" setup=[TuringFixture] begin
    using Bijectors: Bijectors, transformed

    leaf = TwoParamLeaf(2.0, 1.0)
    data = Float64[]
    zero_lik(d, ds) = 0.0
    prob = DistributionsInference.distribution_to_logdensity(leaf, data; loglik = zero_lik)
    n = DistributionsInference.flat_dimension(leaf)
    z = [-0.3, 0.5]
    x, logjac = DistributionsInference.to_constrained(prob, z)

    # Rebuilt through Bijectors' own `transformed`, a different code path from
    # `to_constrained`'s `with_logabsdet_jacobian` call. The zero likelihood
    # isolates the prior-transform identity from the data term.
    target = sum(eachindex(z)) do i
        logpdf(transformed(prob.flat_priors[i]), z[i])
    end
    @test DistributionsInference.logdensity(prob, x) + logjac ≈ target
end

@testitem "to_constrained: the constrained output feeds reconstruct correctly" setup=[ToyFixture] begin
    using Bijectors

    leaf = ToyGammaLeaf(2.0, 1.5, LogNormal(log(2.0), 0.2))
    data = Float64[]
    zero_lik(d, ds) = 0.0
    prob = DistributionsInference.distribution_to_logdensity(leaf, data; loglik = zero_lik)

    z0 = 0.35
    x, _ = DistributionsInference.to_constrained(prob, [z0])
    rebuilt = DistributionsInference.reconstruct(leaf, x)
    @test rebuilt.shape ≈ exp(z0)
    @test rebuilt.shape > 0
    @test rebuilt.scale == leaf.scale
end

@testitem "to_constrained rejects an estimated row with no per-row prior" setup=[TuringFixture] begin
    using Bijectors

    leaf = NoPriorLeaf(2.0, 1.0)
    data = Float64[]
    zero_lik(d, ds) = 0.0
    prob = DistributionsInference.distribution_to_logdensity(leaf, data; loglik = zero_lik)
    @test DistributionsInference.flat_dimension(leaf) == 1

    @test_throws ArgumentError DistributionsInference.to_constrained(prob, [0.1])
end

@testitem "gradient: ForwardDiff through to_constrained ∘ logdensity" setup=[TuringFixture] begin
    using Bijectors
    using ForwardDiff

    # The finite-difference oracle here is ForwardDiff-only by design; the
    # same composition is run through all six backends against a ForwardDiff
    # reference by the AD matrix (`test/ADFixtures`, DI#33), which is what
    # would catch a backend-specific regression such as Mooncake's `xlogy`.

    leaf = TwoParamLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2, 2.8]
    prob = DistributionsInference.distribution_to_logdensity(leaf, data)

    n = DistributionsInference.flat_dimension(leaf)
    z0 = fill(0.1, n)
    function target(z)
        x, logjac = DistributionsInference.to_constrained(prob, z)
        return DistributionsInference.logdensity(prob, x) + logjac
    end

    g = ForwardDiff.gradient(target, z0)
    @test length(g) == n
    @test all(isfinite, g)

    h = 1e-6
    for i in eachindex(z0)
        e = [j == i ? h : 0.0 for j in eachindex(z0)]
        fd = (target(z0 .+ e) - target(z0 .- e)) / (2h)
        @test g[i] ≈ fd atol = 1e-4
    end
end

@testitem "logdensity_to_objective: the negative of to_constrained ∘ logdensity" setup=[TuringFixture] begin
    using Bijectors

    leaf = TwoParamLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2, 2.8]
    prob = DistributionsInference.distribution_to_logdensity(leaf, data)
    f = DistributionsInference.logdensity_to_objective(prob)

    n = DistributionsInference.flat_dimension(leaf)
    z = fill(0.2, n)
    x, logjac = DistributionsInference.to_constrained(prob, z)
    @test f(z) ≈ -(DistributionsInference.logdensity(prob, x) + logjac)

    # At the origin both log-linked rows map to 1.0 with a zero log-Jacobian,
    # so the objective has a closed form independent of the transform
    # machinery: the two priors scored at 1.0 plus a Gamma(1, 1) likelihood.
    @test f(zeros(n)) ≈ -(logpdf(LogNormal(log(2.0), 0.2), 1.0) +
            logpdf(LogNormal(log(1.0), 0.2), 1.0) +
            sum(y -> logpdf(Gamma(1.0, 1.0), y), data))

    @test_throws DimensionMismatch f(z[1:(end - 1)])
end

@testitem "logdensity_to_objective: minimising it finds the MAP point (Optim.jl)" setup=[ToyFixture] begin
    using Bijectors, Optim

    mu, sigma = log(2.0), 0.2
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(mu, sigma))
    data = [1.5, 2.0, 3.2, 2.8, 1.9, 4.1, 2.5, 3.0]
    prob = DistributionsInference.distribution_to_logdensity(leaf, data)
    f = DistributionsInference.logdensity_to_objective(prob)

    res = optimize(f, [0.0], LBFGS())
    z_hat = Optim.minimizer(res)
    x_hat, _ = DistributionsInference.to_constrained(prob, z_hat)
    fitted = DistributionsInference.reconstruct(prob.obj, x_hat)

    # The MAP optimum is a stationary point of the unconstrained target,
    # checked independently of Optim's own convergence bookkeeping.
    h = 1e-6
    fd = (f(z_hat .+ h) - f(z_hat .- h)) / (2h)
    @test fd ≈ 0.0 atol = 1e-3
    @test fitted.shape > 0
    @test fitted.scale == leaf.scale
end

@testitem "logdensity_to_objective: a diffuse prior tracks the MLE" setup=[ToyFixture] begin
    using Bijectors, Optim, Distributions

    # `logdensity` always scores an estimated row's own prior, so an MLE point
    # needs the prior's curvature to be negligible next to the likelihood.
    diffuse = LogNormal(0.0, 100.0)
    leaf = ToyGammaLeaf(2.0, 1.0, diffuse)
    data = [1.5, 2.0, 3.2, 2.8, 1.9, 4.1, 2.5, 3.0]
    prob = DistributionsInference.distribution_to_logdensity(leaf, data)
    f = DistributionsInference.logdensity_to_objective(prob)

    res = optimize(f, [0.0], LBFGS())
    z_hat = Optim.minimizer(res)
    x_hat, _ = DistributionsInference.to_constrained(prob, z_hat)

    # Independent of the optimiser: at the MLE the data log-likelihood's own
    # derivative in `shape` is zero.
    loglik(shape) = sum(y -> logpdf(Gamma(shape, leaf.scale), y), data)
    h = 1e-6
    fd = (loglik(x_hat[1] + h) - loglik(x_hat[1] - h)) / (2h)
    @test fd ≈ 0.0 atol = 1e-2
end

@testitem "to_unconstrained: inverts to_constrained across mixed links" begin
    using DistributionsInference, Distributions
    using Bijectors

    # The same mixed-link object the `to_constrained` item above uses: an
    # identity link, a log link and a logit link in one flat vector, so a
    # per-row slip in either direction shows up here.
    struct RoundTripLeaf{M <: Real, S <: Real, P <: Real}
        mu::M
        sigma::S
        p::P
    end

    function DistributionsInference.parameter_rows(d::RoundTripLeaf)
        return [
            (name = :mu, value = d.mu, prior = Normal(0.0, 2.0),
                support = (-Inf, Inf)),
            (name = :sigma, value = d.sigma,
                prior = truncated(Normal(1.0, 1.0); lower = 0.0),
                support = (0.0, Inf)),
            (name = :p, value = d.p, prior = Beta(2.0, 2.0),
                support = (0.0, 1.0))]
    end

    function DistributionsInference.reconstruct(
            d::RoundTripLeaf, x::AbstractVector)
        return RoundTripLeaf(x[1], x[2], x[3])
    end

    leaf = RoundTripLeaf(-1.5, 2.0, 0.4)
    zero_lik(d, ds) = 0.0
    prob = DistributionsInference.distribution_to_logdensity(
        leaf, Float64[]; loglik = zero_lik)

    x = [-1.5, 2.0, 0.4]
    z = DistributionsInference.to_unconstrained(prob, x)

    # Each link against its own closed form: identity, log, logit.
    @test z[1] ≈ x[1]
    @test z[2] ≈ log(x[2])
    @test z[3] ≈ log(x[3] / (1 - x[3]))

    back, _ = DistributionsInference.to_constrained(prob, z)
    @test back ≈ x

    @test_throws DimensionMismatch DistributionsInference.to_unconstrained(
        prob, x[1:(end - 1)])
end

@testitem "to_unconstrained rejects an estimated row with no per-row prior" setup=[TuringFixture] begin
    using Bijectors

    leaf = NoPriorLeaf(2.0, 1.0)
    zero_lik(d, ds) = 0.0
    prob = DistributionsInference.distribution_to_logdensity(
        leaf, Float64[]; loglik = zero_lik)

    @test_throws ArgumentError DistributionsInference.to_unconstrained(
        prob, [0.1])
end

@testitem "objective_to_distribution: to_constrained then reconstruct" setup=[ToyFixture] begin
    using Bijectors

    leaf = ToyGammaLeaf(2.0, 1.5, LogNormal(log(2.0), 0.2))
    data = [1.5, 2.0, 3.2]
    prob = DistributionsInference.distribution_to_logdensity(leaf, data)

    z = [0.35]
    fitted = objective_to_distribution(prob, z)
    x, _ = DistributionsInference.to_constrained(prob, z)

    @test fitted == DistributionsInference.reconstruct(prob.obj, x)
    @test fitted.shape ≈ exp(z[1])
    @test fitted.scale == leaf.scale
end

@testitem "objective_to_distribution: a 0-estimated object rebuilds at the empty vector" setup=[ToyFixture] begin
    using Bijectors

    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    zero_lik(d, ds) = 0.0
    prob = DistributionsInference.distribution_to_logdensity(
        fixed_leaf, Float64[]; loglik = zero_lik)

    @test objective_to_distribution(prob, Float64[]) == fixed_leaf
end

@testitem "distribution_to_objective: exactly the two-step composition" setup=[TuringFixture] begin
    using Bijectors

    leaf = TwoParamLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2, 2.8]
    f = distribution_to_objective(leaf, data)

    expected = logdensity_to_objective(
        DistributionsInference.distribution_to_logdensity(leaf, data))

    n = DistributionsInference.flat_dimension(leaf)
    for z in ([0.0, 0.0], [0.2, -0.4], [-1.1, 0.9])
        @test f(z) == expected(z)
    end
    @test_throws DimensionMismatch f(fill(0.0, n - 1))
end

@testitem "distribution_to_objective: threads a custom loglik through" setup=[TuringFixture] begin
    using Bijectors

    leaf = TwoParamLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2, 2.8]
    doubled(obj, records) = 2 * sum(y -> logpdf(obj, y), records)

    f = distribution_to_objective(leaf, data; loglik = doubled)
    expected = logdensity_to_objective(
        DistributionsInference.distribution_to_logdensity(
            leaf, data; loglik = doubled))
    plain = logdensity_to_objective(
        DistributionsInference.distribution_to_logdensity(leaf, data))

    n = DistributionsInference.flat_dimension(leaf)
    z = fill(0.2, n)
    @test f(z) == expected(z)
    @test f(z) != plain(z)
end

@testitem "distribution_to_objective: composes with minimise and objective_to_distribution to reproduce optimise_distribution" setup=[ToyFixture] begin
    using Bijectors, Optim

    mu, sigma = log(2.0), 0.2
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(mu, sigma))
    data = [1.5, 2.0, 3.2, 2.8, 1.9, 4.1, 2.5, 3.0]

    prob = DistributionsInference.distribution_to_logdensity(leaf, data)
    z0 = DistributionsInference.to_unconstrained(prob, [leaf.shape])
    z_hat = DistributionsInference.minimise(
        distribution_to_objective(leaf, data), z0, LBFGS())
    by_hand = objective_to_distribution(prob, z_hat)

    fitted = optimise_distribution(leaf, data, LBFGS())
    @test by_hand == fitted
end

@testitem "distribution_to_objective needs Bijectors when it is not loaded" begin
    using DistributionsInference

    # Sibling items in this process `using Bijectors`, so the
    # extension-not-loaded path is only reachable in a fresh process.
    script = """
    using DistributionsInference, Distributions

    Base.get_extension(
        DistributionsInference,
        :DistributionsInferenceBijectorsExt) === nothing ||
        error("the Bijectors extension loaded without Bijectors")

    struct AloneLeaf
        shape::Float64
        scale::Float64
    end

    Distributions.logpdf(d::AloneLeaf, y::Real) = logpdf(
        Gamma(d.shape, d.scale), y)

    function DistributionsInference.parameter_rows(d::AloneLeaf)
        return [(name = :shape, value = d.shape,
                prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
            (name = :scale, value = d.scale, prior = nothing,
                support = (0.0, Inf))]
    end

    function DistributionsInference.reconstruct(
            d::AloneLeaf, x::AbstractVector)
        return AloneLeaf(x[1], d.scale)
    end

    leaf = AloneLeaf(2.0, 1.5)
    thrown = try
        DistributionsInference.distribution_to_objective(leaf, [1.5, 2.0])
        nothing
    catch e
        e
    end
    thrown isa ArgumentError ||
        error("expected an ArgumentError, got \$(repr(thrown))")
    occursin("Bijectors", thrown.msg) ||
        error("the message does not name Bijectors: \$(thrown.msg)")
    occursin("DistributionsInferenceBijectorsExt", thrown.msg) ||
        error("the message does not name the extension: \$(thrown.msg)")
    print("no-bijectors-ok")
    """

    out = IOBuffer()
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project())
           --startup-file=no -e $script`
    ok = success(pipeline(cmd; stdout = out, stderr = out))
    output = String(take!(out))
    ok || println(output)
    @test ok
    @test occursin("no-bijectors-ok", output)
end
