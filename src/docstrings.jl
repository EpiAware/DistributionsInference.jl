# Standard EpiAware docstring conventions: DocStringExtensions `@template`
# blocks giving every function, type, and the module a consistent layout.
#
# PACKAGE-OWNED: scaffold writes this once and never overwrites it. `include`
# it near the TOP of the package module, AFTER the module's
# `using DocStringExtensions: ...` and BEFORE any docstrings are defined (a
# `@template` only applies to docstrings written after it in the same module).

@template (FUNCTIONS, METHODS, MACROS) = """
                                         $(TYPEDSIGNATURES)
                                         $(DOCSTRING)
                                         """

@template TYPES = """
                  $(TYPEDEF)
                  $(DOCSTRING)

                  ---
                  ## Fields
                  $(TYPEDFIELDS)
                  """

@template MODULES = """
                    $(DOCSTRING)

                    ---
                    ## Exports
                    $(EXPORTS)
                    ---
                    ## Imports
                    $(IMPORTS)
                    """
