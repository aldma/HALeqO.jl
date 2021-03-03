# HALeqO.jl
Homotopy Augmented Lagrangian method for EQuality-constrained Optimization

HALeqO.jl is a pure Julia implementation of a solver for continuous nonlinear equality-constrained optimization problems of the form

    min f(x)  over x in R^n  subject to c(x) = 0

based on a homotopy augmented Lagrangian method and globalised Newton's steps with Armijo's linesearch. To use ```haleqo```, you have to pass it an [NLPModel](https://github.com/JuliaSmoothOptimizers/NLPModels.jl); it returns a [GenericExecutionStats](https://github.com/JuliaSmoothOptimizers/SolverTools.jl).

### Prerequisites

These codes were written for Julia 1.5.3. The package dependencies are from March 2021, when our experiments were run. You may get Julia from julialang.org.

### Use with NLPModels and JuMP

You can solve an JuMP model `m` by using NLPModels to convert it.
```
using NLPModelsJuMP, HALeqO
nlp = MathOptNLPModel(m)
out = haleqo(nlp)
```
