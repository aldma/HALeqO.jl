module HALeqO

export haleqo

using SolverCore, NLPModels
using LinearAlgebra, SparseArrays
using PositiveFactorizations
using QDLDL

"""
    merit(y, yhat, mu, fx, cx)
"""
function merit(y, yhat, mu, fx, cx)
    m = fx
    m += 0.25 * sum( mu .* y.^2 )
    m += sum( (cx + mu .* (yhat - 0.5 * y)).^2 ./ mu )
    return m
end

"""
    haleqo( nlp )
"""
function haleqo(
    nlp::AbstractNLPModel;
    x::AbstractVector = copy(nlp.meta.x0),
    y::AbstractVector = copy(nlp.meta.y0),
    tol::Real = 1e-8,
    mu::Real = 0.1,
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
    κmuminus = 0.1
    ls_eta = 1e-4
    mumin = 1e-16
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
    dfx = similar(x)
    jtv = similar(x)
    grad!(nlp, x, dfx)
    jtprod!(nlp, x, y, jtv)
    subres[1:nx] .= dfx .+ jtv
    subres[nx+1:nx+ny] .= cx
    optimality = norm(subres[1:nx], Inf)
    cviolation = norm(cx, Inf)
    cviol__old = cviolation
    residy = cviolation
    is_solved = optimality ≤ tol && cviolation ≤ tol
    is_infeasible = false
    is_tired = iter ≥ max_iter || eltime ≥ max_time || neval_obj(nlp) > max_eval
    dir = zeros(T, nx + ny)
    xold = similar(x)
    yold = similar(y)

    rhoBM = max(1.0, fx) / max(1.0, 0.5*dot(cx, cx)) # without abs()
    rhoBM = max(1e-8, min(rhoBM, 1e8))
    mu = 1.0 / rhoBM

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
        H = cholesky(Positive, H) # without pivoting
        H = Symmetric(SparseMatrixCSC(Matrix(H)))
        J = jac(nlp, x)

        KKT = [H J'; J UniformScaling(-mu)]
        dir .= -subres
        LDLT = QDLDL.qdldl(KKT)
        QDLDL.solve!(LDLT, dir)

        # gradient of merit function
        # ∇x merit = resx + (2/mu) resy
        # ∇y merit = - resy
        jtprod!(nlp, x, subres[nx+1:nx+ny], jtv)
        xold .= subres[1:nx] + (2.0 / mu) .* jtv
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
        mold = merit(y, yhat, mu, fx, cx)
        slope *= ls_eta
        τ = T(1)
        x .+= dir[1:nx]
        y .+= dir[nx+1:nx+ny]
        while true
            fx = obj(nlp, x)
            cons!(nlp, x, cx)
            m = merit(y, yhat, mu, fx, cx)
            # check Armijo's condition
            tol = 10 * eps(T) * (1 + abs(m))
            if m ≤ mold + τ * slope + tol
                break
            end
            τ *= ls_beta
            x .= xold + τ .* dir[1:nx]
            y .= yold + τ .* dir[nx+1:nx+ny]
        end

        # evaluate residuals and check termination
        grad!(nlp, x, dfx)
        jtprod!(nlp, x, y, jtv)
        subres[1:nx] .= dfx .+ jtv
        optimality = norm(subres[1:nx], Inf)
        cviolation = norm(cx, Inf)
        is_solved = optimality ≤ tol && cviolation ≤ tol

        iter += 1
        if !is_solved
            eltime = time() - start_time
            is_tired = iter ≥ max_iter || eltime ≥ max_time || neval_obj(nlp) > max_eval
            if !is_tired
                jtprod!(nlp, x, cx, jtv)
                is_infeasible =
                    mu < mumin && norm(jtv, Inf) < cviolation * √tol
                if !is_infeasible
                    subres[nx+1:nx+ny] .= cx + mu .* (yhat - y)
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
                mu = max(mumin, κmuminus * mu)
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
            @info log_row(Any[iter, fx, cviolation, optimality, mu, "$(iter_type)"])
        elseif use_filter && phi_V(cviolation, optimality) ≤ phi_V_max
            iter_type = :V
            phi_V_max *= phi_theta
            yhat .= y
            subres[nx+1:nx+ny] .= cx
            residy = cviolation
            @info log_row(Any[iter, fx, cviolation, optimality, mu, "$(iter_type)"])
        elseif use_filter && phi_O(cviolation, optimality) ≤ phi_O_max
            iter_type = :O
            phi_O_max *= phi_theta
            yhat .= y
            subres[nx+1:nx+ny] .= cx
            residy = cviolation
            @info log_row(Any[iter, fx, cviolation, optimality, mu, "$(iter_type)"])
        else
            iter_type = :F
            @info log_row(Any[iter, fx, cviolation, optimality, mu, residy])
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
    else
        :exception
    end

    # output
    eltime = time() - start_time
    return GenericExecutionStats(
        nlp,
        status = status,
        solution = x,
        objective = fx,
        dual_feas = optimality,
        primal_feas = cviolation,
        multipliers = y,
        iter = iter,
        elapsed_time = eltime,
    )
end

end # module
