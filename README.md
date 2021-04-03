# HALeqO.jl
Homotopy Augmented Lagrangian method for EQuality-constrained Optimization

HALeqO.jl is a pure Julia implementation of a solver for continuous nonlinear equality-constrained optimization problems of the form

    min f(x)  over x in R^n  subject to c(x) = 0

based on a homotopy augmented Lagrangian method and globalised Newton's steps with Armijo's linesearch. To invoke the ```haleqo``` solver, you have to pass it an [NLPModel](https://github.com/JuliaSmoothOptimizers/NLPModels.jl); it returns a [GenericExecutionStats](https://github.com/JuliaSmoothOptimizers/SolverCore.jl).

    using NLPModels, HALeqO
    out = haleqo(nlp)

You can solve an JuMP model `m` by using NLPModels to convert it.

    using NLPModelsJuMP, HALeqO
    nlp = MathOptNLPModel(m)
    out = haleqo(nlp)

### Linear solver

HALeqO.jl uses [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl)'s `MA57` as main linear solver, based on [HSL](https://www.hsl.rl.ac.uk/). It also uses [PositiveFactorizations.jl](https://github.com/timholy/PositiveFactorizations.jl) for regularizing the Hessian matrix. These could be replaced by, or complemented with, [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl).

### Benchmarking

We compared HALeqO against [IPOPT](https://coin-or.github.io/Ipopt/), via the wrapper provided by [NLPModelsIpopt](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl), and [NCL.jl](https://github.com/JuliaSmoothOptimizers/NCL.jl) invoking IPOPT. See `run_benchmarks.jl` in the `tests` folder.
