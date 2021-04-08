# A. De Marchi, Feb 2021.

using DataFrames
using Printf
using Plots

"""
    time_profile(stats, cost)
Produce a time profile comparing solvers in `stats` using the `cost` function.
Inputs:
- `stats::Dict{Symbol,DataFrame}`: pairs of `:solver => df`;
- `cost::Function`: cost function applyed to each `df`. Should return a vector with the cost of solving the problem at each row;
  - 0 cost is not allowed;
  - If the solver did not solve the problem, return Inf, a negative number, or NaN.
Examples of cost functions:
- `cost(df) = df.elapsed_time`: Simple `elapsed_time` cost. Assumes the solver solved the problem.
- `cost(df) = (df.status .!= :first_order) * Inf + df.elapsed_time`: (default) Takes into consideration the status of the solver.
"""
function time_profile(
    stats::Dict{Symbol,DataFrame},
    cost::Function = df -> (df.status .!= :first_order) * Inf + df.elapsed_time,
    args...;
    kwargs...,
)
    solvers = keys(stats)
    dfs = (stats[s] for s in solvers)
    T = hcat([cost(df) for df in dfs]...)
    time_profile(T, string.(solvers), args...; kwargs...)
end

"""Produce a time profile.
Each column of the matrix `T` defines the cost for a solver (smaller is better).
Failures on a given problem are represented by a negative value, an infinite value, or `NaN`.
The optional argument `logscale` is used to produce a logarithmic (base 10) time plot.
"""
function time_profile(
    T::Array{Float64,2},
    labels::Vector{AbstractString};
    logscale::Bool = true,
    title::AbstractString = "",
    kwargs...,
)
    kwargs = Dict{Symbol,Any}(kwargs)
    # compute time profiles
    (fmat, tvec) = time_fractions(T)
    (nt, ns) = size(fmat)
    # setup style
    length(labels) == 0 && (labels = [@sprintf("column %d", col) for col = 1:ns])
    xlabel = pop!(kwargs, :xlabel, "Runtime" * (logscale ? " (log scale)" : ""))
    ylabel = pop!(kwargs, :ylabel, "Fraction of problems")
    linestyles = pop!(kwargs, :linestyles, Symbol[])
    # initial plot
    profile = Plots.plot(
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        xlims = ((logscale && (tvec[1] == 0.0)) ? tvec[2] : tvec[1], tvec[nt]),
        ylims = (0.0, 1.0),
        xscale = (logscale ? :log10 : :none),
    )
    # plot solvers
    for s = 1:ns
        if length(linestyles) > 0
            kwargs[:linestyle] = linestyles[s]
        end
        Plots.plot!(tvec, fmat[:, s], t = :steppost, label = labels[s]; kwargs...)
    end
    return profile
end

time_profile(
    T::Array{Tn,2},
    labels::Vector{S};
    kwargs...,
) where {Tn<:Number,S<:AbstractString} = time_profile(
    convert(Array{Float64,2}, T),
    convert(Vector{AbstractString}, labels);
    kwargs...,
)

"""
    Compute time fractions used to produce a time profile.
    There is normally no need to call this function directly.
"""
function time_fractions(T::Array{Float64,2})

    # number of problems, number of solvers
    (np, ns) = size(T)

    # manage NaN, Inf, and negative values
    T[isinf.(T)] .= NaN
    T[T.<0] .= NaN
    failures = isnan.(T)

    # create time vector (x-axis)
    tvec = sort(unique(T[.!failures]))
    if tvec[1] == 0.0
        tvec = tvec[2:end]
    end
    append!(tvec, 1.1 * tvec[end])
    nt = length(tvec)

    # Compute fractions for each solver (y-axis)
    fmat = zeros(nt, ns)
    for is = 1:ns
        for it = 1:nt
            fmat[it, is] = sum(T[:, is] .≤ tvec[it]) / np
        end
    end

    return (fmat, tvec)
end

time_fractions(T::Array{Td,2}) where {Td<:Number} =
    time_fractions(convert(Array{Float64,2}, T))
