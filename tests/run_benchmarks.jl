
# load solvers
using HALeqO # may need ``push!(LOAD_PATH,"src/")`` or similar
using NLPModelsIpopt
using NCL

# load problems
using CUTEst
probnames = CUTEst.select(
    min_var = 1,
    min_con = 1,
    max_var = 100,
    max_con = 100,
    only_free_var = true,
    only_equ_con = true,
)

# load tools
using SolverBenchmark
using CSV, DataFrames, Printf
using Plots
pyplot()
include("time_profiles.jl")

# setup benchmarking
TOL = 1e-6
MAXITER = 3000

problems = (CUTEstModel(probname) for probname in probnames)
solvers = Dict{Symbol,Function}(
    :NCL => prob -> NCLSolve(prob, opt_tol = TOL, feas_tol = TOL, max_iter = MAXITER),
    :HALeqO => prob -> haleqo(prob; tol = TOL, max_iter = MAXITER),
    :IPOPT => prob -> ipopt(prob; tol = TOL, print_level = 0, max_iter = MAXITER),
)
stats = bmark_solvers(solvers, problems)

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
datastore = Dict{Symbol,DataFrame}()
for solver ∈ keys(stats)
    datastore[solver] = DataFrame(
        id = stats[solver].id,
        time = stats[solver].elapsed_time,
        iter = stats[solver].iter,
        optim = stats[solver].dual_feas,
        cviol = stats[solver].primal_feas,
        solved = Int.(stats[solver].status .== :first_order),
    )
end

filename = "cutest_eq"
for solver ∈ keys(stats)
    CSV.write("../data/" * filename * "_" * String(solver) * ".csv", datastore[solver], header = false)
end
