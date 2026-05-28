using Test
using AcousticDisguising

# Naming convention in this folder:
#
#   test_*.jl   — Pkg.test() unit tests; runtests.jl `include`s them.
#                 Most must run quickly + headlessly on a CPU (CI-friendly);
#                 `test_extrapolation_writecount.jl` is the one exception
#                 (see note at the include site below).
#
#   Everything else is a diagnostic CLI script run manually — see each
#   file's header for usage. The prefix is informational:
#
#     check_*    — single-question correctness check (does X match Y?)
#     compare_*  — side-by-side comparison of two artefacts
#     sweep_*    — parameter sweep that writes a set of outputs
#     plot_*     — read sweep outputs and produce a figure
#     run_*      — kick off an expensive reference simulation
#     inspect_*  — interactive / one-off introspection tool
#     calibrate_*— derive a scaling constant or fit
#
# Both .jl and .py live here. Python unit tests for the python/mdd
# package live separately under python/mdd/tests/ (pytest-discovered).

@testset "AcousticDisguising" begin
    include("test_reciprocity.jl")
    # Heavier than the `test_*` convention suggests (~600 LOC, includes
    # a parameter sweep); included here for CI reproducibility.
    include("test_extrapolation_writecount.jl")
end
