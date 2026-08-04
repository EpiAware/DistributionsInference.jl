# [Fitting an object](@id fitting)

A distribution becomes fittable by naming its own scalar parameters and how to rebuild itself from a flat vector.
No probabilistic programming language is required to reach that point.
The log-density this produces is PPL-neutral, and Turing (or any other PPL) is an optional layer on top, not a requirement.

This page carries the `ToyDelay` distribution from the README's own quickstart further, sampling it with a `LogDensityProblems`-compatible sampler and with Turing, then reading a fitted chain back onto the distribution either way, before showing the same calls working unchanged against a `ComposedDistributions` tree.

```@example fitting
using DistributionsInference, Distributions, Random
using FlexiChains: FlexiChains

struct ToyDelay{T <: Real}
    shape::T
    scale::T
end

Distributions.logpdf(d::ToyDelay, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::ToyDelay)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::ToyDelay, x::AbstractVector)
    return ToyDelay(x[1], oftype(x[1], d.scale))
end

delay = ToyDelay(2.0, 1.0)
data = [1.5, 2.0, 3.2, 1.8, 2.6]
```

`parameter_rows` names `shape` as estimated (it carries a prior) and `scale` as fixed (its `prior` is `nothing`); `reconstruct` is the one place a type says how to rebuild itself from the flat vector the engine works over.
`ToyDelay`'s fields are typed `T <: Real` rather than a concrete `Float64`, and `reconstruct` rebuilds through `oftype(x[1], d.scale)` rather than `d.scale` directly: a gradient-based sampler differentiates through `reconstruct`, so `x[1]` can carry a dual number rather than a plain `Float64`, and a concretely-typed field would reject it.

## The log density

[`as_logdensity`](@ref) packages the object and the data into a log-density over just its estimated parameters.
[`flat_dimension`](@ref) counts them: here that is `shape` alone, since `scale` carries no prior and stays fixed at its template value.

```@example fitting
prob = DistributionsInference.as_logdensity(delay, data)
DistributionsInference.flat_dimension(delay)
```

[`logdensity`](@ref) scores a flat parameter vector, adding the prior's log-density to the data likelihood of the object rebuilt at that value.

```@example fitting
DistributionsInference.logdensity(prob, [2.0])
```

## Custom likelihood reducers

`loglik` is a pluggable reducer `(obj, data) -> Real`, not fixed to a sum of `logpdf`: any scoring rule reusing the same parameter inventory, [`reconstruct`](@ref), priors and readback works unchanged, so a caller need not build a bespoke fitting path for it.

A SURVIVAL-scoring reducer is one worked case: records known only to exceed a bound (a right-censored observation) score through `logccdf` instead of `logpdf`.

```@example fitting
survival_loglik(obj, records) = sum(y -> logccdf(Gamma(obj.shape, obj.scale), y), records)
bounds = [1.0, 1.5, 2.0]
survival_prob = DistributionsInference.as_logdensity(delay, bounds; loglik = survival_loglik)
DistributionsInference.logdensity(survival_prob, [2.0])
```

Because `survival_loglik` is written generically (`logccdf` differentiates the same way `logpdf` does), point estimation with an external optimiser, penalised point estimation (the same objective, any prior), and gradient-based posterior sampling all work from this one reducer, exactly as they do for the default data likelihood elsewhere on this page.

An ESTIMATED (Monte Carlo) likelihood — one whose reducer draws random replicates internally, e.g. to marginalise a latent variable — is noisier and, unless written through a reparameterisation that keeps it differentiable, not usable by a gradient-based sampler at all: fall back to a gradient-free sampler (`AdvancedMH`, or any other `LogDensityProblems` consumer that does not need a gradient) for a reducer like this. More replicates reduce the noise in each evaluation at the cost of more compute per draw; too few replicates can bias a gradient-free sampler's acceptance rate as easily as they would bias a gradient-based one, so replicate count is a tuning knob to check by inflating it until further increases stop changing the posterior summary.

## Sampling without a probabilistic programming language

`prob` is a `LogDensityProblems` problem, so any consumer of that interface can sample it directly, with no PPL in the loop.
`AdvancedMH`'s random-walk Metropolis is the smallest such consumer to reach for; a gradient backend added through `LogDensityProblemsAD` opens up AdvancedHMC the same way.
`DensityModel` takes the log-density as a plain function, and the wrapper below returns `-Inf` off `shape`'s positive support, which a random-walk proposal does not respect on its own.

```@example fitting
using AdvancedMH
using LinearAlgebra: I

model = AdvancedMH.DensityModel() do x
    any(<=(0), x) ? -Inf : DistributionsInference.logdensity(prob, x)
end
sampler = RWMH(MvNormal(zeros(1), 0.05^2 * I))
transitions = sample(Xoshiro(1), model, sampler, 2000;
    param_names = ["shape"], progress = false)
draws = [t.params for t in transitions][1001:end]
length(draws)
```

[`to_flexichain`](@ref) keys the raw draws by the estimated rows' dotted names, so [`readback`](@ref) reduces them straight back onto the object.

```@example fitting
chain = DistributionsInference.to_flexichain(delay, draws)
fitted = DistributionsInference.readback(delay, chain)
fitted.shape
```

[`distribution_params`](@ref) is the params-first primitive underneath: the same reduction, keyed by dotted name, before the object is rebuilt.

```@example fitting
DistributionsInference.distribution_params(delay, chain)
```

[`readback_draws`](@ref) keeps every draw instead of reducing them, for a per-draw posterior-predictive summary.

```@example fitting
all_fitted = DistributionsInference.readback_draws(delay, chain)
length(all_fitted)
```

## Maximum likelihood and maximum a posteriori

`prob`'s objective is already a plain (unnormalised) log-density, so a standard external optimiser can find a point estimate directly: minimising the negative log-likelihood is maximum likelihood, and minimising the negative log-posterior is maximum a posteriori.
DistributionsInference ships no estimator method for this; [`as_optimisation_objective`](@ref) is only the thin wiring of the unconstrained transform, the objective and [`reconstruct`](@ref) together, once `Bijectors` is loaded — the optimiser itself (`Optim.jl` here) stays external.

```@example fitting
using Bijectors, Optim

f = DistributionsInference.as_optimisation_objective(prob)
res = optimize(f, zeros(DistributionsInference.flat_dimension(delay)), LBFGS())
z_hat = Optim.minimizer(res)
```

`z_hat` is on the unconstrained scale `f` optimises over; push it back through [`to_constrained`](@ref) and [`reconstruct`](@ref) — the same readback path every other sampler in this guide reconstructs through — to get the fitted object at the maximum-a-posteriori point.

```@example fitting
x_hat, _ = DistributionsInference.to_constrained(prob, z_hat)
map_fit = DistributionsInference.reconstruct(prob.obj, x_hat)
map_fit.shape
```

`logdensity` always adds an ESTIMATED row's own prior term (that is what makes the row estimated in the first place; see [`parameter_rows`](@ref)), so a maximum-likelihood point needs an object whose prior is negligible next to the data likelihood, rather than just swapping `loglik`.
A separate type carrying a very diffuse prior on `shape` gives exactly that: its curvature near the likelihood's own optimum is small enough that the MAP point below is the maximum-likelihood estimate up to numerical precision, with the wiring otherwise unchanged.

```@example fitting
struct ToyDelayML{T <: Real}
    shape::T
    scale::T
end

Distributions.logpdf(d::ToyDelayML, y::Real) =
    logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::ToyDelayML)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(0.0, 100.0), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing, support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::ToyDelayML, x::AbstractVector)
    return ToyDelayML(x[1], oftype(x[1], d.scale))
end

ml_delay = ToyDelayML(2.0, 1.0)
ml_prob = DistributionsInference.as_logdensity(ml_delay, data)
ml_f = DistributionsInference.as_optimisation_objective(ml_prob)
ml_n = DistributionsInference.flat_dimension(ml_delay)
ml_res = optimize(ml_f, zeros(ml_n), LBFGS())
ml_x, _ = DistributionsInference.to_constrained(
    ml_prob, Optim.minimizer(ml_res))
DistributionsInference.reconstruct(ml_delay, ml_x).shape
```

## Sampling with Turing

[`as_turing`](@ref) wraps the same log-density as a `DynamicPPL` model, so the object is sampleable with Turing directly.
Each estimated row becomes a named site drawn from its own prior, and the data likelihood is added from the object rebuilt at the draw.

```@example fitting
using DynamicPPL, Turing
using FlexiChains: VNChain

model = DistributionsInference.as_turing(delay, data)
Random.seed!(1)
turing_chain = sample(model, NUTS(), 200; chain_type = VNChain, progress = false)
```

[`readback`](@ref) and [`readback_draws`](@ref) read a `VNChain` back onto the distribution exactly as they read the AdvancedMH chain above; the dotted-name convention is the same either way, so a project can switch samplers without touching its readback code.

```@example fitting
DistributionsInference.readback(delay, turing_chain).shape
```

## Fitting a composed distribution

Nothing above is specific to a hand-written type like `ToyDelay`: the same verbs work directly on a `ComposedDistributions` tree once `ComposedDistributions` is loaded, through a weak extension that maps a tree's own `params_table` onto this package's row schema.

```@example fitting
using ComposedDistributions
using ComposedDistributions: compose, uncertain, event

tree = compose((
    onset_admit = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
    admit_death = LogNormal(0.5, 0.4)))
tree_data = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]

tree_prob = DistributionsInference.as_logdensity(tree, tree_data)
DistributionsInference.flat_dimension(tree)
```

The one estimated parameter is `onset_admit`'s shape (the [`uncertain`](@ref) leaf); `admit_death` stays fixed, exactly like `ToyDelay`'s `scale` above.
Sampling and reading the fit back are the same calls against `tree` instead of `delay`, `to_flexichain`/`readback` or `as_turing` alike; the AdvancedMH sampler from earlier on this page needs no change beyond pointing it at `tree_prob`.

```@example fitting
tree_mh_model = AdvancedMH.DensityModel() do x
    any(<=(0), x) ? -Inf : DistributionsInference.logdensity(tree_prob, x)
end
tree_transitions = sample(Xoshiro(1), tree_mh_model, sampler, 2000;
    param_names = ["onset_admit.shape"], progress = false)
tree_draws = [t.params for t in tree_transitions][1001:end]
tree_chain = DistributionsInference.to_flexichain(tree, tree_draws)
fitted_tree = DistributionsInference.readback(tree, tree_chain)
event(fitted_tree, :onset_admit)
```

`as_turing` works on a tree exactly as it does on `delay`, one named site per estimated row.

```@example fitting
tree_model = DistributionsInference.as_turing(tree, tree_data)
Random.seed!(1)
tree_turing_chain = sample(tree_model, NUTS(), 200; chain_type = VNChain, progress = false)
event(DistributionsInference.readback(tree, tree_turing_chain), :onset_admit)
```

A tree with a *centred* `pool` (ComposedDistributions' partial-pooling spec) is the one case `as_turing` refuses: a centred pool's member row has no fixed `~` prior of its own (it is scored against the reconstructed population instead, through [`extra_logprior`](@ref)), and DynamicPPL has no sampling path for that yet, so `as_turing` raises a clear error naming the affected rows rather than silently mis-scoring them; fit that tree through `as_logdensity` and a `LogDensityProblems`-compatible sampler instead, exactly as in the section above.

A `ComposedDistributions` tree pairs `readback`'s output directly with its own `update(tree, params)` (documented on ComposedDistributions' verb map, linked below) — chain readback itself is this package's job, not ComposedDistributions'; this page shows `readback` because it is the one spelling that works identically whether the object came from this package, from ComposedDistributions, or from a hand-written type like `ToyDelay`.

## See also

- [Public API](@ref public-api) for the full protocol surface (`parameter_rows`, `reconstruct`, `distribution_priors`, `distribution_params`, and the rest).
- ComposedDistributions' [verb map](https://composeddistributions.epiaware.org/dev/getting-started/concepts) for the tree-shaped verbs (`compose`, `uncertain`, `pool`, `update`) this page's composed-distribution example builds on.
