# HALeqO.jl
Homotopy Augmented Lagrangian method for EQuality-constrained Optimization

HALeqO.jl is a pure Julia implementation of a solver for continuous nonlinear equality-constrained optimization problems of the form

    min f(x)  over x in R^n  subject to c(x) = 0

based on a homotopy augmented Lagrangian method and globalised Newton's steps with Armijo's linesearch. To invoke the ```haleqo``` solver, you have to pass it an [NLPModel](https://github.com/JuliaSmoothOptimizers/NLPModels.jl); it returns a [GenericExecutionStats](https://github.com/JuliaSmoothOptimizers/SolverTools.jl).

    using NLPModels, HALeqO
    out = haleqo(nlp)

You can solve an JuMP model `m` by using NLPModels to convert it.

    using NLPModelsJuMP, HALeqO
    nlp = MathOptNLPModel(m)
    out = haleqo(nlp)

### Prerequisites

These codes were written for Julia v1.5.3. The package dependencies are from March 2021, when our experiments were run. You may get Julia from [julialang.org](https://julialang.org/). We compared HALeqO against [IPOPT](https://coin-or.github.io/Ipopt/), via the wrapper [NLPModelsIpopt](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl), and [NCL.jl](https://github.com/JuliaSmoothOptimizers/NCL.jl) invoking IPOPT.
