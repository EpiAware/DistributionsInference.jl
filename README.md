# DistributionsInference <img src="docs/src/assets/logo.svg" width="150" alt="DistributionsInference logo" align="right">

<!-- badges:start -->
| **Documentation** | **Build Status** | **Code Quality** | **License & DOI** | **Downloads** |
|:-----------------:|:----------------:|:----------------:|:-----------------:|:-------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://distributionsinference.epiaware.org/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://distributionsinference.epiaware.org/dev/) | [![Test](https://github.com/EpiAware/DistributionsInference.jl/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/EpiAware/DistributionsInference.jl/actions/workflows/test.yaml) [![codecov](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg)](https://codecov.io/gh/EpiAware/DistributionsInference.jl) [![AD](https://github.com/EpiAware/DistributionsInference.jl/actions/workflows/ad.yaml/badge.svg?branch=main)](https://github.com/EpiAware/DistributionsInference.jl/actions/workflows/ad.yaml) | [![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FDistributionsInference&query=total_requests&label=Downloads)](https://juliapkgstats.com/pkg/DistributionsInference) [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FDistributionsInference&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/DistributionsInference) |

| ForwardDiff | ReverseDiff (tape) | Enzyme forward | Enzyme reverse | Mooncake reverse | Mooncake forward |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [![cov ForwardDiff](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-forwarddiff)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-forwarddiff) | [![cov ReverseDiff](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-reversediff)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-reversediff) | [![cov Enzyme forward](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-enzyme-forward)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-enzyme-forward) | [![cov Enzyme reverse](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-enzyme-reverse)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-enzyme-reverse) | [![cov Mooncake reverse](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-mooncake-reverse)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-mooncake-reverse) | [![cov Mooncake forward](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-mooncake-forward)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-mooncake-forward) |
<!-- badges:end -->

A PPL-neutral fit protocol for any object with parameters: name a type's parameters once, and get a `LogDensityProblems` log-density, a dotted-name posterior readback, and, when Turing is loaded, a ready-made `DynamicPPL` model, all from the same declaration.

## Why DistributionsInference?

- **Any object is fittable, not just a `Distribution`** — implement `parameter_rows` and `reconstruct` once, and a plain struct gets the same fitting surface as a built-in distribution.
- **No PPL required** — `as_logdensity` builds a `LogDensityProblems`-compatible log-density directly, so a hand-rolled sampler, or any other library that speaks the interface, can fit an object with no Turing.jl in the loop.
- **Turing is a thin layer on top, not a rewrite** — `as_turing` wraps the same log-density as a `DynamicPPL` model when a full PPL is wanted.
- **Dotted-name readback** — `readback` and `readback_draws` reduce a chain from either sampling route straight back onto the object, keyed by the same names the protocol declared.
- **Works on `ComposedDistributions` trees out of the box** — a package extension maps a tree's own parameter table onto this protocol, so an event tree is fittable with the same calls as a hand-written type.

## Getting started

See [documentation](https://distributionsinference.epiaware.org/stable/) for a full walkthrough.

Any type becomes fittable by naming its own parameters and how to rebuild itself from a flat vector, no probabilistic programming language required to reach a log-density:

```julia
using DistributionsInference, Distributions

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

leaf = ToyDelay(2.0, 1.0)
data = [1.5, 2.0, 3.2, 1.8, 2.6]
```

`shape` carries a prior so it is estimated; `scale`'s prior is `nothing`, so it stays fixed at its template value.
`as_logdensity` packages the object and data into a log-density over just the estimated parameters:

```julia
prob = DistributionsInference.as_logdensity(leaf, data)
DistributionsInference.logdensity(prob, [2.0])
```

The [getting started guide](https://distributionsinference.epiaware.org/stable/getting-started/) carries this same object further: sampling it with a hand-rolled sampler and with Turing, reading a fitted chain back onto the object either way, and running the same calls against a `ComposedDistributions` tree in place of a hand-written type.

## Where to learn more

- Want to get started running code? See the [getting started guide](https://distributionsinference.epiaware.org/stable/getting-started/).
- Want the full interface? See the [API reference](https://distributionsinference.epiaware.org/stable/lib/public).
- [GitHub Discussions](https://github.com/EpiAware/DistributionsInference.jl/discussions)
- [GitHub Repository](https://github.com/EpiAware/DistributionsInference.jl)

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
