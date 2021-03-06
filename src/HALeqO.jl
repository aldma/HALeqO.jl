module HALeqO

export haleqo

using SolverCore
using LinearAlgebra
using NLPModels
using HSL
using PositiveFactorizations

"""
    normsq(v)
"""
function normsq(v)
    return dot(v, v)
end

"""
    merit(x, y, yhat, μ, fx, cx)
"""
function merit(x, y, yhat, μ, fx, cx)
    return fx + 0.25 * μ * normsq(y) + normsq(cx + μ .* (yhat - 0.5 * y)) / μ
end

"""
    haleqo( nlp )
"""
function haleqo(
    nlp::AbstractNLPModel;
    x::AbstractVector = copy(nlp.meta.x0),
    y::AbstractVector = copy(nlp.meta.y0),
    tol::Real = 1e-8,
    σ::Real = 1e-3,
    μ::Real = 1e-3,
    max_iter::Int = 1000,
    max_time::Real = 100.0,
)

    start_time = time()
    nx, ny = nlp.meta.nvar, nlp.meta.ncon
    T = eltype(x)

    if has_bounds(nlp) || inequality_constrained(nlp)
        error("Problem has inequalities, can't solve it")
    end

    # parameters
    θ = 0.5
    β = 0.5
    κμminus = 0.1
    κσplus = 10.0
    η = 1e-4

    # initialization
    iter = 0
    eltime = 0.0
    status = :unknown
    fx = obj(nlp, x)
    cx = cons(nlp, x)
    yhat = copy(y)
    σhat = copy(σ)
    subres = zeros(T, nx + ny)
    subres[1:nx] .= grad(nlp, x) + jtprod(nlp, x, y)
    subres[nx+1:nx+ny] .= cx
    optimality = norm(subres[1:nx], Inf)
    cviolation = norm(cx, Inf)
    cviol__old = cviolation
    residy = cviolation
    converged = optimality ≤ tol && cviolation ≤ tol
    tired = iter ≥ max_iter || eltime ≥ max_time
    dir = zeros(T, nx + ny)
    xold = zeros(T, nx)
    yold = zeros(T, ny)

    @info log_header(
        [:iter, :fx, :cviol, :optim, :σ, :μ, :resy],
        [Int, T, T, T, T, T, T],
        hdr_override = Dict(:fx => "f(x)", :optim => "‖∇L‖", :cviol => "‖c(x)‖"),
    )
    @info log_row(Any[iter, fx, cviolation, optimality])

    while !(converged || tired)

        # sub-problem update
        if optimality ≤ tol && residy ≤ tol
            # check improvement in constraint violation
            if cviolation > θ * cviol__old
                # update dual regularization parameter
                μ *= κμminus
            end
            # update dual estimate
            yhat .= y
            # re-evaluate some quantities
            subres[nx+1:nx+ny] .= cx
            cviol__old = cviolation
            residy = cviolation
            # print
            @info log_row(Any[iter, fx, cviolation, optimality, σ, μ])
        else
            # print
            @info log_row(Any[iter, fx, cviolation, optimality, σ, μ, residy])
        end

        # search direction
        # compute Newton direction for the regularized sub-problem, possibly with
        # some perturbation of the Hessian matrix to obtain a descent direction
        # for the merit function
        H = hess(nlp, x, y)
        J = jac(nlp, x)

        # TODO
        # - linear solver refactoring
        # - set up sparse matrix, lower/upper triangle
        # - variant: (i) Try to factorize KKT matrix. If successfull go to (ii),
        #                otherwise go to (iv).
        #           (ii) Solve linear system, compute slope, and go to (iii).
        #          (iii) If slope < 0.0, take as search direction and quit, otherwise
        #                go to (iv).
        #           (iv) If σ = 0.0, set σ = σinit, otherwise set σ *= κσplus. Go to (i).

        H = H + H' - triu(H)
        σ = σhat
        HσI = H + UniformScaling(σ)
        Hf, Hd = ldlt(Positive, HσI, Val{true})
        while any(Hd .< 1)
            σ *= κσplus
            HσI = H + UniformScaling(σ)
            Hf, Hd = ldlt(Positive, HσI, Val{true})
        end
        KKT = [HσI J'; J UniformScaling(-μ)]
        LDLT = Ma57(KKT)
        ma57_factorize(LDLT)
        dir .= -subres
        ma57_solve!(LDLT, dir)

        # gradient of merit function
        # ∇x merit = resx + (2/μ) resy
        # ∇y merit = - resy
        xold .= subres[1:nx] + (2.0 / μ) .* jtprod(nlp, x, subres[nx+1:nx+ny])
        # slope of merit along search direction
        # slope = ∇merit ⋅ dir
        slope = dot(dir[1:nx], xold) - dot(dir[nx+1:nx+ny], subres[nx+1:nx+ny])
        if slope ≥ 0.0
            status = :not_desc
            break
        end

        # line-search
        # backtracking line-search along the primal-dual augmented Lagrangian
        # merit function with Armijo sufficient decrease condition
        xold .= x
        yold .= y
        mold = merit(x, y, yhat, μ, fx, cx)
        τ = one(T)
        x .+= dir[1:nx]
        y .+= dir[nx+1:nx+ny]
        while true
            fx = obj(nlp, x)
            cons!(nlp, x, cx)
            m = merit(x, y, yhat, μ, fx, cx)
            # check Armijo's condition
            if m ≤ mold + η * τ * slope
                break
            else
                τ *= β
                x .= xold + τ .* dir[1:nx]
                y .= yold + τ .* dir[nx+1:nx+ny]
            end
        end

        # evaluate residuals and check termination
        subres[1:nx] .= grad(nlp, x) + jtprod(nlp, x, y)
        optimality = norm(subres[1:nx], Inf)
        cviolation = norm(cx, Inf)
        converged = optimality ≤ tol && cviolation ≤ tol

        iter += 1
        if !converged
            eltime = time() - start_time
            tired = iter ≥ max_iter || eltime ≥ max_time
            if !tired
                subres[nx+1:nx+ny] .= cx + μ .* (yhat - y)
                residy = norm(subres[nx+1:nx+ny], Inf)
            end
        end

    end

    status = if converged
        :first_order
    elseif tired
        if iter ≥ max_iter
            :max_iter
        else
            :max_time
        end
    end

    # output
    eltime = time() - start_time
    return GenericExecutionStats(
        status,
        nlp,
        solution = x,
        objective = fx,
        multipliers = y,
        dual_feas = optimality,
        primal_feas = cviolation,
        iter = iter,
        elapsed_time = eltime,
    )
end

end # module
