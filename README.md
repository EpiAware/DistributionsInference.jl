# DistributionsInference <img src="docs/src/assets/logo.svg" width="150" alt="DistributionsInference logo" align="right">

<!-- badges:start -->
| **Documentation** | **Build Status** | **Code Quality** | **License & DOI** | **Downloads** |
|:-----------------:|:----------------:|:----------------:|:-----------------:|:-------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://distributionsinference.epiaware.org/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://distributionsinference.epiaware.org/dev/) | [![Test](https://github.com/EpiAware/DistributionsInference.jl/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/EpiAware/DistributionsInference.jl/actions/workflows/test.yaml) [![codecov](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg)](https://codecov.io/gh/EpiAware/DistributionsInference.jl) [![AD](https://github.com/EpiAware/DistributionsInference.jl/actions/workflows/ad.yaml/badge.svg?branch=main)](https://github.com/EpiAware/DistributionsInference.jl/actions/workflows/ad.yaml) | [![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FDistributionsInference&query=total_requests&label=Downloads)](https://juliapkgstats.com/pkg/DistributionsInference) [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FDistributionsInference&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/DistributionsInference) |

| ForwardDiff | ReverseDiff (tape) | Enzyme forward | Enzyme reverse | Mooncake reverse | Mooncake forward |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [![cov ForwardDiff](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-forwarddiff)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-forwarddiff) | [![cov ReverseDiff](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-reversediff)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-reversediff) | [![cov Enzyme forward](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-enzyme-forward)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-enzyme-forward) | [![cov Enzyme reverse](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-enzyme-reverse)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-enzyme-reverse) | [![cov Mooncake reverse](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-mooncake-reverse)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-mooncake-reverse) | [![cov Mooncake forward](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-mooncake-forward)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-mooncake-forward) |
<!-- badges:end -->

The inference layer for the EpiAware composable-modelling stack: a PPL-neutral fit protocol and log-density engine, so any object that names its own parameters is fittable without committing to a probabilistic programming language.

## Why DistributionsInference?

- Fitting an object today usually means committing to one PPL's macros; here
  a type opts in by naming its own scalar parameters, and it becomes
  fittable everywhere, hand-rolled sampler and PPL alike.
- The log-density this produces has no PPL dependency, so it evaluates
  through whatever `LogDensityProblems`-compatible sampler a project already
  uses.
- A parameter becomes estimated by attaching a prior at the row level;
  nothing else about a type needs to change to be fitted.
- Reading a fitted chain back onto a concrete object is the same one call
  whether the chain came from a hand-rolled sampler or from Turing.
- Turing and Bijectors support are opt-in layers over the same protocol, not
  requirements, so a project can start with the bare log-density and add a
  PPL later without rewriting its model.
- Ported from ComposedDistributions.jl's own fit protocol, so a composed
  distribution and a plain hand-written type share one estimation surface.

## Getting started

See [documentation](https://distributionsinference.epiaware.org/dev/) for a full walkthrough.

A type becomes fittable by naming its scalar parameters and how to rebuild
itself from a flat vector — no other change needed.

```julia
using DistributionsInference, Distributions, Random

struct ToyDelay
    shape::Float64
    scale::Float64
end

Distributions.logpdf(d::ToyDelay, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::ToyDelay)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::ToyDelay, x::AbstractVector)
    return ToyDelay(x[1], d.scale)
end
```

`as_logdensity` packages a template object and data into a log-density over
just the one estimated parameter (`shape`; `scale` stays fixed).

```julia
leaf = ToyDelay(2.0, 1.0)
data = [1.5, 2.0, 3.2, 1.8, 2.6]
prob = DistributionsInference.as_logdensity(leaf, data)
DistributionsInference.flat_dimension(leaf)
```

Any `LogDensityProblems`-compatible sampler can drive `prob`; here is the
tiniest one, a ten-line random-walk Metropolis, so this stays self-contained.

```julia
function toy_sample(prob, x0, n; step = 0.2, rng = Xoshiro(1))
    x, lp = copy(x0), DistributionsInference.logdensity(prob, x0)
    draws = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        prop = x .+ step .* randn(rng, length(x))
        if all(>(0), prop)
            lp_prop = DistributionsInference.logdensity(prob, prop)
            log(rand(rng)) < lp_prop - lp && ((x, lp) = (prop, lp_prop))
        end
        draws[i] = copy(x)
    end
    return draws
end

draws = toy_sample(prob, [2.0], 500)
```

`readback` reduces the draws to a fitted `ToyDelay`, through the same
dotted-name chain a real PPL's sampler would hand back.

```julia
chain = DistributionsInference.to_flexichain(leaf, draws)
fit = DistributionsInference.readback(leaf, chain)
fit.shape
```

The [getting started guide](https://distributionsinference.epiaware.org/dev/getting-started/)
carries this same object further: reading every draw with `readback_draws`,
and sampling with Turing instead of the toy sampler above.

## Where to learn more

- Want to get started running code? See the [getting started guide](https://distributionsinference.epiaware.org/dev/getting-started/).
- Want to understand the API? See the [API reference](https://distributionsinference.epiaware.org/dev/lib/public).
- Want to see the code? Check out our [GitHub repository](https://github.com/EpiAware/DistributionsInference.jl).

## Getting help

For usage questions, ask on the [Julia Discourse](https://discourse.julialang.org)
(the SciML or usage categories) or the [epinowcast community forum](https://community.epinowcast.org),
our home for epidemiological modelling questions.
Please use [GitHub issues](https://github.com/EpiAware/DistributionsInference.jl/issues)
for bug reports and feature requests only.

<!-- standard-sections:start -->
<!-- MANAGED by EpiAwarePackageTools.scaffold — do not edit between the
     markers. These standard sections are re-rendered on every scaffold_update;
     edit the package-owned sections outside them, or CITATION.cff. -->

## Contributing

We welcome contributions and new contributors! Please open an issue or pull request on [GitHub](https://github.com/EpiAware/DistributionsInference.jl). This package follows [ColPrac](https://github.com/SciML/ColPrac) and the [SciML style](https://github.com/SciML/SciMLStyle).

## How to cite

If you use DistributionsInference in your work, please cite it. Citation metadata lives in [`CITATION.cff`](https://github.com/EpiAware/DistributionsInference.jl/blob/main/CITATION.cff), which GitHub renders as a "Cite this repository" button on the repository page.

## Code of conduct

Please note that the DistributionsInference project is released with a [Contributor Code of Conduct](https://github.com/EpiAware/.github/blob/main/CODE_OF_CONDUCT.md). By contributing, you agree to abide by its terms.
<!-- standard-sections:end -->
