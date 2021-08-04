# load solver
using HALeqO

# load tools
using SolverBenchmark

# load problems
using CUTEst
probnames = CUTEst.select(
    min_var = 1,
    min_con = 1,
    max_var = 2,
    max_con = 2,
    only_free_var = true,
    only_equ_con = true,
)

# setup testing
problems = (CUTEstModel(probname) for probname in probnames)
solver = prob -> haleqo(prob; tol = 1e-4, max_iter = 1000, max_time = 100.0)

# run solver!
stats = solve_problems(solver, problems)

# get statistics
@info "HALeqO statuses" count_unique(stats.status)
num_solved_problems = sum(stats.status .== :first_order)
