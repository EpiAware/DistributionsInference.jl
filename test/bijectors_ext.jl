# DistributionsInference × Bijectors extension: `to_constrained` maps an
# unconstrained flat vector to the constrained ESTIMATED parameters plus a
# log-Jacobian, built per row from `FitLogDensity`'s `flat_priors`. The
# load-bearing checks are the per-row transform against Bijectors itself
# across a single-row and a multi-row object, the 0-estimated and
# dimension-mismatch edge cases, the rejection of a row with no per-row prior
# (an `extra_logprior`-scored, object-dependent row — the one case this
# generic extension cannot build a bijector for), the change-of-variables
# identity a sampler relies on (`logdensity(prob, x) + logjac` is the
# unconstrained-space log-target at `z`), that the constrained output feeds
# `reconstruct` correctly, and a gradient flowing through the composition.

@testitem "Bijectors extension loads" begin
    using Bijectors

    @test Base.get_extension(DistributionsInference,
        :DistributionsInferenceBijectorsExt) !== nothing
end

@testitem "to_constrained: closed-form identity for a LogNormal-Gamma row" setup=[ToyFixture] begin
    using Bijectors

    # A single uncertain Gamma shape with a LogNormal(mu, sigma) prior: the
    # bijector maps the positive shape through log, so `x = exp(z)` and the
    # log-Jacobian is `z`. Because a LogNormal is exp of a Normal by
    # definition, `logpdf(LogNormal(mu, sigma), exp(z)) + z` collapses to
    # `logpdf(Normal(mu, sigma), z)` exactly — a closed-form oracle
    # independent of the transform machinery itself.
    mu, sigma = 0.4, 0.3
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(mu, sigma))
    data = Float64[]
    zero_lik(d, ds) = 0.0
    prob = DistributionsInference.as_logdensity(leaf, data; loglik = zero_lik)

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
    prob = DistributionsInference.as_logdensity(leaf, data; loglik = zero_lik)
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

    # Every constrained value lands in its prior's support (both rows here
    # are positive-support LogNormal priors).
    for i in eachindex(x)
        @test insupport(prob.flat_priors[i], x[i])
    end

    # A length mismatch is rejected eagerly, like the rest of the codec.
    @test_throws DimensionMismatch DistributionsInference.to_constrained(
        prob, z[1:(end - 1)])
end

@testitem "to_constrained: a 0-estimated object round-trips at the empty vector" setup=[ToyFixture] begin
    using Bijectors

    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    data = Float64[]
    zero_lik(d, ds) = 0.0
    prob = DistributionsInference.as_logdensity(fixed_leaf, data; loglik = zero_lik)
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
    prob = DistributionsInference.as_logdensity(leaf, data; loglik = zero_lik)
    n = DistributionsInference.flat_dimension(leaf)
    z = [-0.3, 0.5]
    x, logjac = DistributionsInference.to_constrained(prob, z)

    # The unconstrained-space log-target, rebuilt independently row by row
    # through Bijectors' own `transformed` distribution (a different code
    # path from `to_constrained`'s `with_logabsdet_jacobian` call). A
    # zero likelihood isolates the prior-transform identity this test
    # targets from the (already separately tested) data term.
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
    prob = DistributionsInference.as_logdensity(leaf, data; loglik = zero_lik)

    z0 = 0.35
    x, _ = DistributionsInference.to_constrained(prob, [z0])
    rebuilt = DistributionsInference.reconstruct(leaf, x)
    @test rebuilt.shape ≈ exp(z0)
    @test rebuilt.shape > 0  # lands in the Gamma shape's positive support
    @test rebuilt.scale == leaf.scale  # the fixed parameter, untouched
end

@testitem "to_constrained rejects an estimated row with no per-row prior" setup=[TuringFixture] begin
    using Bijectors

    leaf = NoPriorLeaf(2.0, 1.0)
    data = Float64[]
    zero_lik(d, ds) = 0.0
    prob = DistributionsInference.as_logdensity(leaf, data; loglik = zero_lik)
    @test DistributionsInference.flat_dimension(leaf) == 1

    @test_throws ArgumentError DistributionsInference.to_constrained(prob, [0.1])
end

@testitem "gradient: ForwardDiff through to_constrained ∘ logdensity" setup=[TuringFixture] begin
    using Bijectors
    using ForwardDiff

    leaf = TwoParamLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2, 2.8]
    prob = DistributionsInference.as_logdensity(leaf, data)

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

@testitem "as_optimisation_objective: the negative of to_constrained ∘ logdensity" setup=[TuringFixture] begin
    using Bijectors

    leaf = TwoParamLeaf(2.0, 1.0)
    data = [1.5, 2.0, 3.2, 2.8]
    prob = DistributionsInference.as_logdensity(leaf, data)
    f = DistributionsInference.as_optimisation_objective(prob)

    n = DistributionsInference.flat_dimension(leaf)
    z = fill(0.2, n)
    x, logjac = DistributionsInference.to_constrained(prob, z)
    @test f(z) ≈ -(DistributionsInference.logdensity(prob, x) + logjac)
    @test isfinite(f(zeros(n)))

    # A length mismatch is rejected eagerly, like the rest of the codec (via
    # `to_constrained`).
    @test_throws DimensionMismatch f(z[1:(end - 1)])
end

@testitem "as_optimisation_objective: minimising it finds the MAP point (Optim.jl)" setup=[ToyFixture] begin
    using Bijectors, Optim

    mu, sigma = log(2.0), 0.2
    leaf = ToyGammaLeaf(2.0, 1.0, LogNormal(mu, sigma))
    data = [1.5, 2.0, 3.2, 2.8, 1.9, 4.1, 2.5, 3.0]
    prob = DistributionsInference.as_logdensity(leaf, data)
    f = DistributionsInference.as_optimisation_objective(prob)

    res = optimize(f, [0.0], LBFGS())
    z_hat = Optim.minimizer(res)
    x_hat, _ = DistributionsInference.to_constrained(prob, z_hat)
    fitted = DistributionsInference.reconstruct(prob.obj, x_hat)

    # The MAP optimum is a stationary point of the unconstrained target: a
    # symmetric finite-difference gradient check, independent of Optim's own
    # convergence bookkeeping.
    h = 1e-6
    fd = (f(z_hat .+ h) - f(z_hat .- h)) / (2h)
    @test fd ≈ 0.0 atol = 1e-3
    @test fitted.shape > 0
    @test fitted.scale == leaf.scale
end

@testitem "as_optimisation_objective: a diffuse prior tracks the MLE" setup=[ToyFixture] begin
    using Bijectors, Optim, Distributions

    # `logdensity` always scores an ESTIMATED row's own prior, so a genuine
    # maximum-likelihood point needs the prior's curvature to be negligible
    # next to the likelihood: a very diffuse prior on `shape` gives that,
    # and the optimum tracks the closed-form Gamma-shape MLE.
    diffuse = LogNormal(0.0, 100.0)
    leaf = ToyGammaLeaf(2.0, 1.0, diffuse)
    data = [1.5, 2.0, 3.2, 2.8, 1.9, 4.1, 2.5, 3.0]
    prob = DistributionsInference.as_logdensity(leaf, data)
    f = DistributionsInference.as_optimisation_objective(prob)

    res = optimize(f, [0.0], LBFGS())
    z_hat = Optim.minimizer(res)
    x_hat, _ = DistributionsInference.to_constrained(prob, z_hat)

    # A closed-form check independent of the optimiser: at the MLE, the
    # data log-likelihood's own derivative in `shape` is zero (the diffuse
    # prior contributes negligibly here).
    loglik(shape) = sum(y -> logpdf(Gamma(shape, leaf.scale), y), data)
    h = 1e-6
    fd = (loglik(x_hat[1] + h) - loglik(x_hat[1] - h)) / (2h)
    @test fd ≈ 0.0 atol = 1e-2
end
