# DistributionsInference x ComposedDistributions: the fit protocol over a
# composed distribution's public codec (`params_table`, `flat_dimension`,
# `unflatten`, `reconstruct`). A translation layer only. CD's generated codec
# (CD#178, CD#190) already carries the flat-vector layout (stick-breaking
# coordinates, pooled latents and hyperparameters, shared-tag dedup); this
# maps its `edge`/`param` columns onto DistributionsInference's
# dotted-`name` row schema.
#
# Calling straight into that codec also inherits CD#189's world-age hazard: a
# leaf whose protocol methods come from another package's extension can fail
# inside the `@generated` `unflatten` if that extension loads late. The fix
# belongs upstream (CD#189), not here.
module DistributionsInferenceComposedDistributionsExt

using ComposedDistributions: ComposedDistributions, AbstractComposedDistribution
import DistributionsInference: parameter_rows, estimated_rows, flat_dimension,
                               reconstruct, extra_logprior, extra_prior_state

# A centred pooled row's prior is a `CentredPoolPrior` marker, not a fixed
# distribution: it is scored against the reconstructed hyperparameters, so it
# takes the protocol's object-dependent-prior path (`prior = nothing` at the
# row level, scored through `extra_logprior`).
_di_prior(prior) = prior isa ComposedDistributions.CentredPoolPrior ? nothing : prior

function _di_row(edge::Symbol, param::Symbol, value, support, prior)
    (
        name = Symbol(edge, ".", param), value = value,
        prior = _di_prior(prior), support = support)
end

# `params_table` does not itself run CD's pool-group and namespace (CD#177)
# checks, and these entry points are the only place a DI caller touches the
# tree before `reconstruct`, so run them here to fail early on a malformed
# tree.
function _validated_params_table(d::AbstractComposedDistribution)
    ComposedDistributions._validate_pool_groups(d)
    ComposedDistributions._validate_tree_names(d)
    return ComposedDistributions.params_table(d)
end

function parameter_rows(d::AbstractComposedDistribution)
    tbl = _validated_params_table(d)
    edges, params_col = tbl.edge, tbl.param
    values, supports, priors = tbl.value, tbl.support, tbl.prior
    return [_di_row(edges[i], params_col[i], values[i], supports[i], priors[i])
            for i in eachindex(edges)]
end

# Must override the generic `prior !== nothing` default: a centred pooled row
# is translated to `prior = nothing` above but still consumes a slot in CD's
# flat vector, so estimation is decided from CD's own untranslated priors.
function estimated_rows(d::AbstractComposedDistribution)
    tbl = _validated_params_table(d)
    edges, params_col = tbl.edge, tbl.param
    values, supports, priors = tbl.value, tbl.support, tbl.prior
    idx = findall(!isnothing, priors)
    return [_di_row(edges[i], params_col[i], values[i], supports[i], priors[i])
            for i in idx]
end

# The same must-override as above, and CD's own count rather than a re-derived
# one, so the two cannot drift.
flat_dimension(d::AbstractComposedDistribution) = ComposedDistributions.flat_dimension(d)

function reconstruct(d::AbstractComposedDistribution, x::AbstractVector)
    return ComposedDistributions.reconstruct(d, x)
end

# The centred-pooled rows depend only on `d`, not on `x`, so this walk runs
# once at `distribution_to_logdensity` construction rather than per
# evaluation (DI#28).
function extra_prior_state(d::AbstractComposedDistribution)
    ComposedDistributions.centred_pool_rows(d)
end

# The centred-pooled population term, `0.0` when `d` has no centred pooling.
function extra_logprior(d::AbstractComposedDistribution, ::Any,
        x::AbstractVector, state)
    isempty(state) && return 0.0
    nt = ComposedDistributions.unflatten(d, x)
    return ComposedDistributions.pool_centred_logprior(state, nt)
end

end # module DistributionsInferenceComposedDistributionsExt
