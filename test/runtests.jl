# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Main test entry. `:ad`-tagged items live under `test/ad/` with their own
# environment and are excluded here (see test/ad/runtests.jl).
#
# ARGS filters:
#   skip_quality  — skip the QA testset (fast local iteration)
#   quality_only  — run only the QA testset
#   readme_only   — run only `:readme`-tagged items (README/tutorial tests)

using EpiAwarePackageTools: run_package_tests

# `run_package_tests` roots discovery at `test/` rather than the package root,
# so a nested worktree checkout is never scanned (kit #191). Otherwise a
# drop-in for TestItemRunner's `@run_package_tests`.

if "skip_quality" in ARGS
    run_package_tests(@__DIR__;
        filter = ti -> !(:quality in ti.tags) && !(:ad in ti.tags))
elseif "quality_only" in ARGS
    run_package_tests(@__DIR__; filter = ti -> :quality in ti.tags)
elseif "readme_only" in ARGS
    run_package_tests(@__DIR__; filter = ti -> :readme in ti.tags)
else
    run_package_tests(@__DIR__; filter = ti -> !(:ad in ti.tags))
end
