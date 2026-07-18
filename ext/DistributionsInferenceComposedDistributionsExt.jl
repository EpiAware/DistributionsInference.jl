# DistributionsInference × ComposedDistributions: the fit protocol
# (`parameter_rows`/`estimated_rows`/`flat_dimension`/`reconstruct`/
# `extra_logprior`) over a composed distribution's existing public codec
# (`params_table`, `flat_dimension`, `unflatten`, `reconstruct`). This is a
# thin translation layer, not new inference logic — CD's generated codec
# (ComposedDistributions#178/#190) already carries the flat-vector layout
# (stick-breaking `Resolve` coordinates, pooled z-latents/hyperparameters,
# shared-tag dedup); this extension only maps CD's `edge`/`param`-columned
# `params_table` onto DistributionsInference's dotted-`name` row schema.
# Ported from ComposedDistributions' own `as_logdensity`/`ComposedLogDensity`
# machinery (ComposedDistributions#185), which this extension supersedes.
#
# Once `parameter_rows`/`reconstruct` are correct, DistributionsInference's
# existing generic machinery (`to_flexichain`, `readback`, `readback_draws`,
# and the `DynamicPPL` extension's `as_turing`) needs no ComposedDistributions-
# specific code at all: a stick coordinate, a pooled z-latent and a shared tag
# are each already a scalar dotted row, so the chain arm of a composed tree's
# read-back falls out of the core protocol for free.
#
# A composed tree containing a leaf whose leaf-protocol methods (`free_leaf`,
# `_leaf_free_type`, ...) come from a DIFFERENT package extension (e.g.
# ModifiedDistributions' `Affine`/`Weighted`/`Transformed`/`Modified`) can hit
# ComposedDistributions#189's world-age hazard in the generated codec if that
# extension is not yet loaded when `unflatten`'s `@generated` body is first
# compiled — this extension calls straight into CD's `unflatten`/`reconstruct`
# and inherits that hazard unchanged; it is not worked around here (tracked on
# CD#189/PR4, a load-order-independent registration table CD's own codec
# generator needs regardless of this extension).
module DistributionsInferenceComposedDistributionsExt

using ComposedDistributions: ComposedDistributions, AbstractComposedDistribution
import DistributionsInference: parameter_rows, estimated_rows, flat_dimension,
                               reconstruct, extra_logprior

# A CD `params_table` row's `prior` column carries a `CentredPoolPrior` marker
# (ComposedDistributions.Pool.jl) for a centred pooled parameter: the
# population term is scored against the RECONSTRUCTED hyperparameters, not the
# flat value alone, so it is not a fixed distribution `logdensity` can score
# per-row. DistributionsInference's row schema handles exactly this case by
# convention (`parameter_rows`'s docstring in protocol.jl): an object-dependent
# prior carries `prior = nothing` at the row level and is scored instead
# through `extra_logprior`. Translating here, rather than leaving the marker
# in the DI-facing row, keeps the row schema to the four documented fields for
# every tree, pooled or not.
_di_prior(prior) = prior isa ComposedDistributions.CentredPoolPrior ? nothing : prior

function _di_row(edge::Symbol, param::Symbol, value, support, prior)
    (
        name = Symbol(edge, ".", param), value = value,
        prior = _di_prior(prior), support = support)
end

# CD's own `as_logdensity` gates a tree's pool-group consistency (every member
# of a group shares one population) and its pool/shared/root-edge namespace
# (#177) once, eagerly, before scoring. Neither check is a params_table walk,
# so `params_table(d)` alone does not run them; this extension's entry points
# are the only place a DI caller touches the tree before `reconstruct`, so the
# checks are run here rather than silently skipped (a malformed tree would
# otherwise surface as a confusing failure deep inside `_update`/`_pool_hyper`
# instead of this clear, up-front error).
function _validated_params_table(d::AbstractComposedDistribution)
    ComposedDistributions._validate_pool_groups(d)
    ComposedDistributions._validate_tree_names(d)
    return ComposedDistributions.params_table(d)
end

# `parameter_rows`: one DI row per `params_table` row, dotted `edge.param`
# name, `value`/`support` passed through, `prior` translated per `_di_prior`.
function parameter_rows(d::AbstractComposedDistribution)
    tbl = _validated_params_table(d)
    edges, params_col = tbl.edge, tbl.param
    values, supports, priors = tbl.value, tbl.support, tbl.prior
    return [_di_row(edges[i], params_col[i], values[i], supports[i], priors[i])
            for i in eachindex(edges)]
end

# `estimated_rows`/`flat_dimension` MUST override the generic (`prior !==
# nothing`) default: a centred pooled row's DI-facing `prior` is translated to
# `nothing` above (matching the object-dependent-prior convention), so the
# generic filter would silently drop it from the flat vector even though CD's
# own generated codec DOES consume an `x` slot for it (the member's own
# latent). Built from `params_table`'s own `prior !== nothing` test — CD's own
# `_estimated_rows`/`as_logdensity` convention (logdensity.jl) — so this
# cannot drift from CD's internal notion of "estimated", independent of the
# DI-facing translation above.
function estimated_rows(d::AbstractComposedDistribution)
    tbl = _validated_params_table(d)
    edges, params_col = tbl.edge, tbl.param
    values, supports, priors = tbl.value, tbl.support, tbl.prior
    idx = findall(!isnothing, priors)
    return [_di_row(edges[i], params_col[i], values[i], supports[i], priors[i])
            for i in idx]
end

# `flat_dimension`: CD's own generated count, read straight off the codec
# rather than re-derived from `estimated_rows(d)`'s length, so the two cannot
# drift apart.
flat_dimension(d::AbstractComposedDistribution) = ComposedDistributions.flat_dimension(d)

# `reconstruct`: a direct delegation. CD's own `reconstruct` already collapses
# every uncertain leaf at `x` (stick-breaking coordinates, pooled latents and
# hyperparameters, shared tags all included) and holds every fixed parameter
# at `d`'s template value — nothing left for this extension to add.
function reconstruct(d::AbstractComposedDistribution, x::AbstractVector)
    return ComposedDistributions.reconstruct(d, x)
end

# `extra_logprior`: the centred-pooled population term, `0.0` when `d` has no
# centred pooling (the common case, no extra cost). Recomputes `unflatten`
# independently of `reconstruct`'s own internal call — DistributionsInference's
# protocol calls the two separately with no shared cache slot for a package's
# object-dependent extra state, so a tree with centred pooling pays
# `unflatten` twice per evaluation. Worth a follow-up if profiling ever flags
# it; not optimised here.
function extra_logprior(d::AbstractComposedDistribution, ::Any, x::AbstractVector)
    rows = ComposedDistributions._centred_pool_rows(d)
    isempty(rows) && return 0.0
    nt = ComposedDistributions.unflatten(d, x)
    return ComposedDistributions._pool_centred_logprior(rows, nt)
end

end # module DistributionsInferenceComposedDistributionsExt
