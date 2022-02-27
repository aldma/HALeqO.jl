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

HALeqO.jl uses the free [QDLDL.jl](https://github.com/osqp/QDLDL.jl) routines as main linear solver and [PositiveFactorizations.jl](https://github.com/timholy/PositiveFactorizations.jl) for regularizing the Hessian matrix. These could be replaced by, or complemented with, [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) and [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl)'s `MA57` based on [HSL](https://www.hsl.rl.ac.uk/).

### Citing

If you are using HALeqO for your work, we encourage you to

* Cite the related [paper](https://doi.org/10.1109/CDC45484.2021.9683199),
* Put a star on this repository.

### Bug reports and support

Please report any issues via the [issue tracker](https://github.com/aldma/HALeqO.jl/issues). All types of issues are welcome including bug reports, typos, feature requests and so on.

### Benchmarks

We compared HALeqO against [IPOPT](https://coin-or.github.io/Ipopt/), via the wrapper provided by [NLPModelsIpopt](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl), and [NCL.jl](https://github.com/JuliaSmoothOptimizers/NCL.jl) invoking IPOPT. See `run_benchmarks.jl` in the `tests` folder.
