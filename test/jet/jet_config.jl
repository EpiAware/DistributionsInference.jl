# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Optional JET configuration for the isolated runner (test/jet/runtests.jl).
# Defining `JET_REPORT_FILTER` (a `report -> Bool` predicate, keeping a report
# when it returns `true`) switches the runner from `test_package` to
# `report_package` + filter. With no filter the strict default applies: fail on
# any report.
#
# A DynamicPPL `@model` package needs this, since the tilde macro hides `~`
# assignments from JET's static analysis and yields false
# `UndefVarErrorReport`s (and `MethodErrorReport`s through the `:=` tracker).
# To drop exactly those:
#
# const JET_REPORT_FILTER = dynamicppl_model_filter
