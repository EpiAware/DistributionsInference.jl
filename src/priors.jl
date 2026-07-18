# Default-prior assembly over the fit protocol: the generic, params-first
# analogue of ComposedDistributions' `build_priors`/`default_prior`
# (CD#195/DI#20), but over any object implementing `parameter_rows` rather
# than a `ComposedDistributions` tree specifically.
#
# This is a separate implementation, not a thin wrapper over CD's: DI depends
# on ComposedDistributions (weakly, via an eventual extension), not the
# reverse, so CD's own `default_prior`/`build_priors` cannot delegate here
# without inverting that dependency edge. The two packages carry independent
# copies of the same brms-style, parameter-name-driven heuristic, ported by
# hand — see ComposedDistributions' `default_prior`/`_is_positive_param`/
# `_is_location_param` (`src/composers/introspection.jl`) for the sibling
# copy this mirrors.

# A `parameter_rows` row's own parameter name, stripping any dotted path
# prefix (`Symbol("onset.shape")` -> `:shape`): the classification below
# looks at the parameter's own natural domain, not the path it sits at.
function _own_param_name(name::Symbol)
    s = string(name)
    i = findlast('.', s)
    return i === nothing ? name : Symbol(s[(i + 1):end])
end

# Location parameters live on the whole line, so they get an unconstrained
# default even under a positive-support row.
function _is_location_param(p::Symbol)
    p === :mu || p === :location || p === :loc || p === :lower || p === :upper
end

# Scale/shape/rate-type parameters are positive by construction, so they get
# a positive-truncated default regardless of the row's own `support`.
function _is_positive_param(p::Symbol)
    p === :sigma || p === :scale || p === :rate || p === :shape ||
        p === :alpha || p === :beta || p === :theta || p === :nu ||
        p === :k || p === :df || p === :mean || p === :sd
end

@doc "

Pick a default prior for one `parameter_rows` row, brms-style.

`default_prior(row)` is [`distribution_priors`](@ref)'s per-row default for
rows the caller does not override. `row` is a [`parameter_rows`](@ref)-shaped
`NamedTuple` `(; name, value, prior, support)`; the prior family follows the
parameter's own name (the last dotted segment of `name`, e.g. `:shape` from
`Symbol(\"onset.shape\")`), not the row's `support`:

- a `[0, 1]`-support row (a simplex/probability parameter) -> `Uniform(0, 1)`.
- a scale/shape/rate-type name (`:sigma`, `:scale`, `:shape`, `:rate`, ...) ->
  `truncated(Normal(value, scale); lower = 0)`, positive by construction.
- a location name (`:mu`, `:location`, a bound) -> `Normal(value, scale)`,
  unconstrained even under a positive-support row.
- otherwise, falls back to `support`: non-negative ->
  `truncated(Normal(value, scale); lower = 0)`, else `Normal(value, scale)`.

The spread `scale` is `max(abs(value), 1)`, a weakly-informative width that
scales with the parameter's magnitude.

Mirrors ComposedDistributions' `default_prior` (same heuristic, ported by
hand — DI cannot depend on CD's copy; see this file's header note).

# Arguments
- `row`: a [`parameter_rows`](@ref) row `(; name, value, prior, support)`.

# See also
- [`distribution_priors`](@ref): assembles a full row set from this default.
"
function default_prior(row)
    lo, hi = row.support
    scale = max(abs(float(row.value)), one(float(row.value)))
    own = _own_param_name(row.name)
    if lo == 0 && hi == 1
        return Distributions.Uniform(0, 1)
    elseif _is_positive_param(own)
        return Distributions.truncated(Distributions.Normal(row.value, scale); lower = 0)
    elseif _is_location_param(own)
        return Distributions.Normal(row.value, scale)
    elseif lo >= 0 && isinf(hi)
        return Distributions.truncated(Distributions.Normal(row.value, scale); lower = 0)
    else
        return Distributions.Normal(row.value, scale)
    end
end

@doc "

Assemble a fully-specified row set from a fittable object, brms-style.

`distribution_priors(obj; priors, default)` reads [`parameter_rows`](@ref)`(obj)`
and returns the same row shape with every row's `prior` field filled: a
`priors` override for that row's dotted `name`, if given, else the row's own
attached `prior` if it is already set, else `default(row)` (support-derived,
[`default_prior`](@ref) unless a different `default` is given). The result is
directly usable as `obj`'s replacement row set — e.g. feeding
[`reconstruct`](@ref)/[`estimated_rows`](@ref)/[`as_logdensity`](@ref) through
the bare-row-vector fittable-object identity ([`parameter_rows`](@ref)`(rows)
=== rows`) — so `distribution_priors(obj)` alone is the estimate-everything
path for any fit-protocol object, generalising ComposedDistributions'
`param_priors`/`uncertain(tree)` (CD#195) beyond composed-distribution trees.

# Arguments
- `obj`: the fittable object (or a bare row vector).

# Keyword Arguments
- `priors`: per-parameter overrides, an `AbstractDict{Symbol}` keyed by a
  row's dotted `name` (default: empty).
- `default`: a function `row -> prior` for rows not overridden and not
  already carrying a `prior` (default: [`default_prior`](@ref)).

# Examples
```@example
using DistributionsInference, Distributions

rows = [(name = :shape, value = 2.0, prior = nothing, support = (0.0, Inf)),
    (name = :scale, value = 1.0, prior = nothing, support = (0.0, Inf))]
priored = DistributionsInference.distribution_priors(rows)
priored[1].prior
```

# See also
- [`default_prior`](@ref): the support-derived per-row default.
- [`parameter_rows`](@ref): the row inventory read and replaced.
"
function distribution_priors(
        obj; priors = Dict{Symbol, Any}(), default = default_prior)
    rows = collect(parameter_rows(obj))
    return map(rows) do row
        prior = if haskey(priors, row.name)
            priors[row.name]
        elseif row.prior !== nothing
            row.prior
        elseif default !== nothing
            default(row)
        else
            throw(ArgumentError(
                "no prior for $(row.name) and no default supplied"))
        end
        return merge(row, (; prior = prior))
    end
end
