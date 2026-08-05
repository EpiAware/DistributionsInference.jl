@testitem "Package loads" begin
    using DistributionsInference

    @test isdefined(DistributionsInference, :DistributionsInference)
end

@testitem "the exported and public surfaces are the declared ones" begin
    using DistributionsInference

    # `export` carries the names a caller writes; `public` carries the ones a
    # caller implements or that a package loaded alongside this one owns.
    # Pinning both catches a name that drifts across the line. Two names are
    # load-bearing on the `public` side: `ComposedDistributions` exports
    # `default_prior` and `LogDensityProblems` exports `logdensity`, so
    # exporting either here makes it ambiguous in a session holding both.
    exported = sort(filter(
        s -> Base.isexported(DistributionsInference, s),
        names(DistributionsInference)))
    public_only = sort(setdiff(names(DistributionsInference), exported))

    @test exported == [
        :DistributionsInference,
        :distribution_draws,
        :distribution_params,
        :distribution_to_logdensity,
        :distribution_to_turing,
        :logdensity_to_objective,
        :objective_to_distribution,
        :optimise_distribution,
        :parameter_rows,
        :point_estimate,
        :with_priors]
    @test public_only == [
        :FitLogDensity,
        :default_prior,
        :estimated_rows,
        :extra_logprior,
        :extra_prior_state,
        :flat_dimension,
        :logdensity,
        :minimise,
        :reconstruct,
        :to_constrained,
        :to_unconstrained]
end

@testitem "an extension-backed stub names the package to load" begin
    using DistributionsInference

    # Without its trigger package each stub raises an `ArgumentError` naming
    # the package and the extension, rather than a `MethodError` that names
    # neither. Sibling items load `Bijectors` and `DynamicPPL`, so this is
    # only reachable in a fresh process. The `FlexiChains` readback stubs
    # have the same coverage in `test/readback.jl`.
    script = """
    using DistributionsInference, Distributions

    rows = [(name = :shape, value = 2.0, prior = LogNormal(0.0, 0.2),
        support = (0.0, Inf))]
    prob = DistributionsInference.distribution_to_logdensity(rows, [1.5, 2.0])

    cases = [
        (() -> DistributionsInference.logdensity_to_objective(prob),
            "Bijectors", "DistributionsInferenceBijectorsExt"),
        (() -> DistributionsInference.to_constrained(prob, [0.0]),
            "Bijectors", "DistributionsInferenceBijectorsExt"),
        (() -> DistributionsInference.to_unconstrained(prob, [2.0]),
            "Bijectors", "DistributionsInferenceBijectorsExt"),
        (() -> DistributionsInference.objective_to_distribution(prob, [0.0]),
            "Bijectors", "DistributionsInferenceBijectorsExt"),
        (() -> DistributionsInference.minimise(sum, [0.0], nothing),
            "Optim", "DistributionsInferenceOptimExt"),
        (() -> DistributionsInference.distribution_to_turing(rows, [1.5]),
            "DynamicPPL", "DistributionsInferenceDynamicPPLExt")]
    for (call, pkg, ext) in cases
        thrown = try
            call()
            nothing
        catch e
            e
        end
        thrown isa ArgumentError ||
            error("expected an ArgumentError, got \$(repr(thrown))")
        occursin(pkg, thrown.msg) ||
            error("the message does not name \$pkg: \$(thrown.msg)")
        occursin(ext, thrown.msg) ||
            error("the message does not name \$ext: \$(thrown.msg)")
    end
    print("stub-error-ok")
    """

    out = IOBuffer()
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project())
           --startup-file=no -e $script`
    ok = success(pipeline(cmd; stdout = out, stderr = out))
    output = String(take!(out))
    ok || println(output)
    @test ok
    @test occursin("stub-error-ok", output)
end
