
# TODO
# - separate computation and flow control
# - improve direction calculation (H not necessarily pos. def.)

# load solver
using HALeqO # may need ``push!(LOAD_PATH,"src/")`` or similar

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

# setup testing
TOL = 1e-4
MAXITER = 1000
MAXTIME = 100.0
problems = (CUTEstModel(probname) for probname in probnames)
solver = prob -> haleqo(prob; tol = TOL, max_iter = MAXITER, max_time = MAXTIME)
stats = solve_problems(solver, problems)

@info "HALeqO statuses" count_unique(stats.status)
num_solved_problems = sum(stats.status .== :first_order)
