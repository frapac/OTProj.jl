
#=
    OTKKTSystem
=#

struct OTKKTSystem{T, VI, VT, MT} <: MadNLP.AbstractReducedKKTSystem{T, VT, MT, MadNLP.ExactHessian{T, VT}}
    aug_com::MT
    data::OTData{T}
    K::OTCondensedBlock{T}
    pr_diag::VT
    du_diag::VT
    # Buffers
    z1::VT # L x S
    z2::VT # L x S
    b1::VT # L
end

function OTKKTSystem{T, VI, VT, MT}(nlp::OTCompactModel, ind_cons=MadNLP.get_index_constraints(nlp)) where {T, VI, VT, MT}
    L, S = nlp.data.L, nlp.data.S
    aug_com = zeros(L, L)
    pr_diag = zeros(L*S+1)
    du_diag = zeros(L+1)
    K = OTCondensedBlock(nlp.data, zeros(L*S))

    z1  = zeros(L*S)
    z2  = zeros(L*S)
    b1  = zeros(S)
    return OTKKTSystem{T, VI, VT, MT}(
        aug_com, nlp.data, K,
        pr_diag, du_diag,
        z1, z2, b1,
    )
end

MadNLP.num_variables(kkt::OTKKTSystem) = kkt.data.L
MadNLP.get_hessian(kkt::OTKKTSystem) = aug_com
MadNLP.get_jacobian(kkt::OTKKTSystem) = 0
MadNLP.is_reduced(kkt::OTKKTSystem) = true
MadNLP.nnz_jacobian(kkt::OTKKTSystem) = 0

function update_internal!(kkt::OTKKTSystem)
    L, S = kkt.data.L, kkt.data.S
    kkt.K.B.Σ .= kkt.pr_diag[1:L*S]
    kkt.K.ρ[1] = kkt.pr_diag[L*S+1]
    init!(kkt.K)
    return
end

function MadNLP.initialize!(kkt::OTKKTSystem{T}) where T
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    return
end

function MadNLP.get_raw_jacobian(kkt::OTKKTSystem)
    S, L = kkt.data.S, kkt.data.L
    j_i = ones(Int, S * L)
    j_j = collect(1:S*L)
    j_z = copy(kkt.data.d)
    for (i, j, v) in zip(findnz(kkt.data.A1)...)
        push!(j_i, i+1)
        push!(j_j, j)
        push!(j_z, v)
    end
    return MadNLP.SparseMatrixCOO(L+1, S*L, j_i, j_j, j_z)
end

function MadNLP.set_jacobian_scaling!(kkt::OTKKTSystem{T,VI,VT,MT}, constraint_scaling::AbstractVector) where {T,VI,VT,MT}
    # TODO
end

function _mul_expanded!(y::AbstractVector, kkt::OTKKTSystem, x::AbstractVector)
    S, L = kkt.data.S, kkt.data.L
    A1 = RowOperator{Float64}(L, S)
    A2 = ColumnOperator{Float64}(L, S)

    n = L * S

    yx = view(y, 1:n)
    yy = view(y, n+3:n+2+L)
    xx = view(x, 1:n)
    xy = view(x, n+3:n+2+L)
    Σₓ = view(kkt.pr_diag, 1:n)
    Ax = kkt.b1

    # / x
    yx .= Σₓ .* xx
    mul!(Ax, A2, xx)
    mul!(yx, A2', Ax, 1.0, 1.0)
    mul!(yx, A1', xy, 1.0, 1.0)
    axpy!(x[n+2], kkt.data.d, yx)
    # / s
    y[n+1] = kkt.pr_diag[n+1] * x[n+1] - x[n+2]
    # / z
    y[n+2] = dot(kkt.data.d, xx) - x[n+1]
    # / y
    mul!(yy, A1, xx)
    return
end


function MadNLP.mul!(y::AbstractVector, kkt::OTKKTSystem, x::AbstractVector)
    if size(kkt.aug_com, 1) == length(x) == length(y)
        mul!(y, kkt.aug_com, x)
    else
        _mul_expanded!(y, kkt, x)
    end
end

function MadNLP.jtprod!(
    y::AbstractVector,
    kkt::OTKKTSystem{T, VI, VT, MT},
    x::AbstractVector,
) where {T, VI, VT, MT}
    S, L = kkt.data.S, kkt.data.L
    A1 = RowOperator{Float64}(L, S)

    vd = x[1]
    y_x = view(y, 1:S*L)
    y_x .= vd .* kkt.data.d
    # Slack term
    y[S*L+1] = -vd

    vA1 = view(x, 2:L+1)
    mul!(y_x, A1', vA1, 1.0, 1.0)
    return y
end

function MadNLP.compress_jacobian!(kkt::OTKKTSystem)
    return
end

function MadNLP.compress_hessian!(kkt::OTKKTSystem)
    return
end

function MadNLP.eval_jac_wrapper!(
    solver::MadNLP.MadNLPSolver,
    kkt::OTKKTSystem{T, VI, VT, MT},
    x::MadNLP.PrimalVector{T},
) where {T, VI, VT, MT}
    return # do nothing
end

function MadNLP.eval_lag_hess_wrapper!(
    solver::MadNLP.MadNLPSolver,
    kkt::OTKKTSystem{T, VI, VT, MT},
    x::MadNLP.PrimalVector{T},
    l::Vector{T};
    is_resto=false,
) where {T, VI, VT, MT}
    return # do nothing
end

function MadNLP.build_kkt!(kkt::OTKKTSystem)
    L, S = kkt.data.L, kkt.data.S
    update_internal!(kkt)
    # Load buffers
    z1 = kkt.z1
    z2 = kkt.z2
    b1 = kkt.b1
    # Assemble Schur-complement
    assemble_multithreads!(kkt.aug_com, kkt.K)
    # Symmetrize
    @inbounds for i in 1:L, j in (i+1):L
        kkt.aug_com[i, j] = kkt.aug_com[j, i]
    end
    return
end

function backsolve!(x::AbstractVector, solver::MadNLP.MadNLPSolver, b::AbstractVector)
    kkt = solver.kkt
    S, L = kkt.data.S, kkt.data.L
    z1 = kkt.z1  # buffer L x S

    nx = S * L

    Σ = view(kkt.pr_diag, 1:nx)   # X^{-1} S
    ρ = kkt.pr_diag[nx+1]         # t / r

    # Decompose left-hand side
    Δx = view(x, 1:nx)
    Δy = view(x, nx+3:nx+2+L)
    Δyy = zeros(L)

    # Decompose right-hand side
    v1 = view(b, 1:nx)
    v2 = b[nx+1]
    v3 = b[nx+2]
    v4 = view(b, nx+3:nx+2+L)

    # Schur-complement w.r.t inequality constraint
    w1 = v1 .+ (ρ * v3 + v2) .* kkt.data.d

    Δyy .= v4
    ldiv!(z1, kkt.K, w1)
    mul!(Δyy, kkt.data.A1, z1, 1.0, -1.0)

    solver.cnt.linear_solver_time += @elapsed begin
        status = MadNLP.solve!(solver.linear_solver, Δyy)
        Δy .= Δyy
    end

    # Recover Δx
    z1 .= w1
    mul!(z1, kkt.data.A1', Δy, -1.0, 1.0)
    ldiv!(Δx, kkt.K, z1)

    # Recover solution (Δr, Δt)
    Δr = -v3 + dot(kkt.data.d, Δx)
    x[nx+1] = Δr             # Δr
    x[nx+2] = -v2 + ρ * Δr   # Δt

    return true
end

function MadNLP.solve_refine_wrapper!(
    solver::MadNLP.MadNLPSolver{T, <:OTKKTSystem{T,VI,VT,MT}},
    xm::MadNLP.AbstractKKTVector,
    bm::MadNLP.AbstractKKTVector,
) where {T, VI, VT, MT}
    x = MadNLP.primal_dual(xm)
    b = MadNLP.primal_dual(bm)
    # return backsolve!(x, solver, b)
    res = similar(b)
    rep = similar(b)
    norm_b = norm(b, Inf)

    fill!(x, zero(T))
    fill!(res, zero(T))

    axpy!(-1, b, res)                        # ϵ = -b

    iter = 0
    residual_ratio = Inf
    noprogress = 0

    max_iter = 10

    while true
        if (iter > max_iter) || (residual_ratio < 1e-12)
            break
        end
        iter += 1

        rep .= res
        backsolve!(res, solver, rep)        # ϵ = -A⁻¹ b
        axpy!(-1, res, x)                   # x = x + A⁻¹ b
        mul!(res, solver.kkt, x)            # ϵ = A x
        axpy!(-1, b, res)                   # ϵ = Ax - b

        norm_res = norm(res, Inf)
        residual_ratio = norm_res / (one(T)+norm_b)
    end
    return true
end

