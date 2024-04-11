
struct OTCompactModel{T} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    data::OTData{T}
    δ::T

    res::Vector{T}

    j_i::Vector{Int}
    j_j::Vector{Int}
    j_z::Vector{T}

    h_i::Vector{Int}
    h_j::Vector{Int}
    h_z::Vector{T}

    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    counters::NLPModels.Counters
end

OTCompactModel(w, q, d, delta; options...) = OTCompactModel(OTData(w, q, d), delta; options...)
function OTCompactModel(data::OTData, delta; eval_hessian=false)
    S, L = data.S, data.L
    res = zeros(S)

    # Assemble Jacobian as COO
    j_i = ones(S * L)
    j_j = collect(1:S*L)
    j_z = copy(data.d)
    for (i, j, v) in zip(findnz(data.A1)...)
        push!(j_i, i+1)
        push!(j_j, j)
        push!(j_z, v)
    end

    # Assemble Hessian as COO
    h_i, h_j, h_z = Int[], Int[], Float64[]
    if eval_hessian
        W = data.A2' * data.A2
        for (i, j, v) in zip(findnz(W)...)
            if j <= i
                push!(h_i, i)
                push!(h_j, j)
                push!(h_z, v)
            end
        end
    end

    # Initial variable
    x0 = zeros(S * L)
    # x0 .= 1e-4
    for s in 1:S, l in 1:L
        x0[s + S * (l-1)] = data.w[s] / L
    end
    y0 = fill(0.0, L+1)

    meta = NLPModels.NLPModelMeta(
        S * L,     #nvar
        ncon = L+1,
        nnzj = length(j_z),
        nnzh = length(h_z),
        x0 = x0,
        y0 = y0,
        lvar = zeros(S * L),
        lcon = [-Inf; data.q],
        ucon = [delta; data.q],
        minimize = true
    )

    return OTCompactModel{Float64}(
        data, delta, res,
        j_i, j_j, j_z,
        h_i, h_j, h_z,
        meta, NLPModels.Counters(),
    )
end

function NLPModels.obj(ot::OTCompactModel, x::Vector{T}) where T
    A2 = ColumnOperator{Float64}(ot.data.L, ot.data.S)
    residual = ot.res
    residual .= ot.data.w
    mul!(residual, A2, x, 1.0, -1.0)
    return 0.5 * dot(residual, residual)
end

function NLPModels.cons!(ot::OTCompactModel, x::Vector{T}, c::Vector{T}) where T
    A1 = RowOperator{Float64}(ot.data.L, ot.data.S)
    L = ot.data.L
    c[1] = dot(ot.data.d, x)
    ca1 = view(c, 2:L+1)
    mul!(ca1, A1, x)
    return c
end

function NLPModels.grad!(ot::OTCompactModel, x::Vector{T}, g::Vector{T}) where T
    A2 = ColumnOperator{Float64}(ot.data.L, ot.data.S)
    residual = ot.res
    residual .= ot.data.w
    mul!(residual, A2, x, 1.0, -1.0)
    mul!(g, A2', residual)
    return g
end

function NLPModels.jac_structure!(ot::OTCompactModel, I::AbstractVector{Ti}, J::AbstractVector{Ti}) where Ti
    copyto!(I, ot.j_i)
    copyto!(J, ot.j_j)
    return (I, J)
end

function NLPModels.jac_coord!(ot::OTCompactModel, x::Vector{T}, jac::Vector{T}) where T
    copyto!(jac, ot.j_z)
    return jac
end

function NLPModels.jtprod!(ot::OTCompactModel, x::Vector{T}, v::Vector{T}, jtv::Vector{T}) where T
    L = ot.data.L
    jtv .+= v[1] .* ot.data.d
    va1 = view(v, 2:L+1)
    mul!(jtv, ot.data.A1', va1, 1.0, 1.0)
    return jtv
end

function NLPModels.hess_structure!(ot::OTCompactModel, I::AbstractVector{Ti}, J::AbstractVector{Ti}) where Ti
    copyto!(I, ot.h_i)
    copyto!(J, ot.h_j)
    return (I, J)
end

function NLPModels.hess_coord!(ot::OTCompactModel, x, y, hess; obj_weight=1.0)
    copyto!(hess, ot.h_z)
    return
end
