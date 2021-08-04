# load solvers
using HALeqO
using NLPModelsIpopt
using NCL # from https://github.com/JuliaSmoothOptimizers/NCL.jl, using IPOPT
using Percival

# load problems
using CUTEst
probnames = CUTEst.select(
    min_var = 1, # default 1
    min_con = 1, # default 1
    max_var = 2, # default 100
    max_con = 2, # default 100
    only_free_var = true,
    only_equ_con = true,
)

# load tools
using SolverBenchmark
using CSV, DataFrames, Printf
using Plots
include("time_profiles.jl")

# setup benchmarking
filename = "cutest_eq_tol5"
TOL = 1e-5 # default 1e-6
MAXITER = 100 # default 3000

problems = (CUTEstModel(probname) for probname in probnames)
solvers = Dict{Symbol,Function}(
    :HALeqO => prob -> haleqo(prob; tol = TOL, max_iter = MAXITER),
    :NCL => prob -> NCLSolve(prob, opt_tol = TOL, feas_tol = TOL, max_iter = MAXITER),
    :IPOPT => prob -> ipopt(prob; tol = TOL, print_level = 0, max_iter = MAXITER),
    :Percival => prob -> percival(prob; atol = TOL, rtol = 0.0, ctol = TOL, max_iter = MAXITER),
)

# run solvers!
stats = bmark_solvers(solvers, problems)

# get statistics
for solver ∈ keys(stats)
    @info "$solver statuses" count_unique(stats[solver].status)
end

statuses, avgs = quick_summary(stats; cols = [:iter, :elapsed_time])
for solver ∈ keys(stats)
    @info "statistics for" solver statuses[solver] avgs[solver]
end

num_solved_problems = Dict{Symbol,Int}()
for solver ∈ keys(stats)
    num_solved_problems[solver] = sum(stats[solver].status .== :first_order)
end

# performance and time profiles
cost(df) = (df.status .!= :first_order) * Inf + df.elapsed_time
pprof = performance_profile(stats, cost)
tprof = time_profile(stats, cost)

# store data
for solver ∈ keys(stats)
    CSV.write("data/" * filename * "_" * String(solver) * ".csv", stats[solver], header = true)
end
