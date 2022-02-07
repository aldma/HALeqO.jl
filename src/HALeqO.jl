module HALeqO

export haleqo

using SolverCore, NLPModels
using LinearAlgebra, SparseArrays
using HSL, PositiveFactorizations

"""
    normsq(v)
"""
normsq(v) = dot(v, v)

"""
    merit(y, yhat, μ, fx, cx)
"""
merit(y, yhat, μ, fx, cx) =
    fx + 0.25 * μ * normsq(y) + normsq(cx + μ .* (yhat - 0.5 * y)) / μ

"""
    haleqo( nlp )
"""
function haleqo(
    nlp::AbstractNLPModel;
    x::AbstractVector = copy(nlp.meta.x0),
    y::AbstractVector = copy(nlp.meta.y0),
    tol::Real = 1e-8,
    μ::Real = 0.1,
    max_iter::Int = 3000,
    max_time::Real = 300.0,
    max_eval::Int = 100000,
    use_filter::Bool = false,
)

    start_time = time()
    nx, ny = nlp.meta.nvar, nlp.meta.ncon
    T = eltype(x)

    if has_bounds(nlp) || inequality_constrained(nlp)
        error("Problem has inequalities, can't solve it")
    end

    # parameters
    θ = 0.5
    ls_beta = 0.5
    κμminus = 0.1
    ls_eta = 1e-4
    μmin = 1e-16
    subtol = sqrt(tol) # assuming tol < 1
    kappa_e = 0.1

    # initialization
    iter = 0
    eltime = 0.0
    status = :unknown
    fx = obj(nlp, x)
    cx = cons(nlp, x)
    yhat = copy(y)
    subres = zeros(T, nx + ny)
    subres[1:nx] .= grad(nlp, x) + jtprod(nlp, x, y)
    subres[nx+1:nx+ny] .= cx
    optimality = norm(subres[1:nx], Inf)
    cviolation = norm(cx, Inf)
    cviol__old = cviolation
    residy = cviolation
    is_solved = optimality ≤ tol && cviolation ≤ tol
    is_infeasible = false
    is_tired = iter ≥ max_iter || eltime ≥ max_time || neval_obj(nlp) > max_eval
    dir = zeros(T, nx + ny)
    xold = zeros(T, nx)
    yold = zeros(T, ny)

    rhoBM = (1.0 / μ) * max(1.0, fx) / max(1.0, 0.5*normsq(cx)) # without abs()
    rhoBM = max(1e-4, min(rhoBM, 1e4))
    μ = 1.0 / rhoBM

    if use_filter
        phi_beta = 0.1 # 0 < beta \le 1
        phi_V(cviol, optim) = cviol + phi_beta * optim
        phi_O(cviol, optim) = cviol * phi_beta + optim
        phi_theta = 0.1 # 0 < theta < 1
        phi_V_max = phi_V(cviolation, optimality)
        phi_O_max = phi_O(cviolation, optimality)
        phi_V_max *= phi_theta
        phi_O_max *= phi_theta
    end

    @info log_header(
        [:iter, :fx, :cviol, :optim, :μ, :resy],
        [Int, T, T, T, T, T],
        hdr_override = Dict(:fx => "f(x)", :optim => "‖∇L‖", :cviol => "‖c(x)‖"),
    )
    @info log_row(Any[iter, fx, cviolation, optimality])

    while !(is_solved || is_infeasible || is_tired)

        # search direction
        # compute Newton direction for the regularized sub-problem, possibly with
        # some perturbation of the Hessian matrix to obtain a descent direction
        # for the merit function
        H = hess(nlp, x, y)
        H = Symmetric(SparseMatrixCSC(Matrix(cholesky(Positive, H, Val{false}))))
        J = jac(nlp, x)

        KKT = [H J'; J UniformScaling(-μ)]
        dir .= -subres
        LDLT = Ma57(KKT)
        ma57_factorize(LDLT)
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
        mold = merit(y, yhat, μ, fx, cx)
        slope *= ls_eta
        τ = one(T)
        x .+= dir[1:nx]
        y .+= dir[nx+1:nx+ny]
        while true
            fx = obj(nlp, x)
            cons!(nlp, x, cx)
            m = merit(y, yhat, μ, fx, cx)
            # check Armijo's condition
            if m ≤ mold + τ * slope
                break
            end
            τ *= ls_beta
            x .= xold + τ .* dir[1:nx]
            y .= yold + τ .* dir[nx+1:nx+ny]
        end

        # evaluate residuals and check termination
        subres[1:nx] .= grad(nlp, x) + jtprod(nlp, x, y)
        optimality = norm(subres[1:nx], Inf)
        cviolation = norm(cx, Inf)
        is_solved = optimality ≤ tol && cviolation ≤ tol

        iter += 1
        if !is_solved
            eltime = time() - start_time
            is_tired = iter ≥ max_iter || eltime ≥ max_time || neval_obj(nlp) > max_eval
            if !is_tired
                is_infeasible =
                    μ < μmin && norm(jtprod(nlp, x, cx), Inf) < cviolation * √tol
                if !is_infeasible
                    subres[nx+1:nx+ny] .= cx + μ .* (yhat - y)
                    residy = norm(subres[nx+1:nx+ny], Inf)
                end
            end
        end

        # sub-problem update
        if optimality ≤ subtol && residy ≤ subtol
            iter_type = :M
            # check improvement in constraint violation
            if cviolation > max(θ * cviol__old, tol)
                # update dual regularization parameter
                μ = max(μmin, κμminus * μ)
            end
            # update subproblem tolerance
            if cviolation ≤ sqrt(tol)
                subtol = max(tol, kappa_e*subtol)
            end
            # update dual estimate
            yhat .= y
            # re-evaluate some quantities
            subres[nx+1:nx+ny] .= cx
            cviol__old = cviolation
            residy = cviolation
            @info log_row(Any[iter, fx, cviolation, optimality, μ, "$(iter_type)"])
        elseif use_filter && phi_V(cviolation, optimality) ≤ phi_V_max
            iter_type = :V
            phi_V_max *= phi_theta
            yhat .= y
            subres[nx+1:nx+ny] .= cx
            residy = cviolation
            @info log_row(Any[iter, fx, cviolation, optimality, μ, "$(iter_type)"])
        elseif use_filter && phi_O(cviolation, optimality) ≤ phi_O_max
            iter_type = :O
            phi_O_max *= phi_theta
            yhat .= y
            subres[nx+1:nx+ny] .= cx
            residy = cviolation
            @info log_row(Any[iter, fx, cviolation, optimality, μ, "$(iter_type)"])
        else
            iter_type = :F
            @info log_row(Any[iter, fx, cviolation, optimality, μ, residy])
        end

    end

    status = if is_solved
        :first_order
    elseif is_infeasible
        :infeasible
    elseif is_tired
        if iter ≥ max_iter
            :max_iter
        elseif eltime ≥ max_time
            :max_time
        elseif neval_obj(nlp) > max_eval
            :max_eval
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
