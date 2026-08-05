# Shared wording for a function whose only methods live in a package
# extension. Julia's own `MethodError` reports that no method matched but
# names no package to load, which is the one thing a caller needs, so each
# extension-backed stub answers that itself.
#
# The `FlexiChains` readback has a richer version of this in `readback.jl`: it
# also separates "you have not loaded FlexiChains" from "the extension failed
# to load", which the chain-free stubs here have no need for.

function _extension_required(
        f::Symbol, pkg::AbstractString, ext::AbstractString)
    throw(ArgumentError(
        "`$f` needs `$pkg`: its implementation lives in the `$ext` package " *
        "extension, which loads only once `$pkg` is in the session. Run " *
        "`using $pkg` first (and `Pkg.add(\"$pkg\")` if it is not installed " *
        "yet)."))
end
