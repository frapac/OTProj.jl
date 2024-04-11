
#=
    OTKKTSystem
=#

struct OTKKTSystem{T, VI, VT, MT, LS} <: MadNLP.AbstractReducedKKTSystem{T, VT, MT, MadNLP.ExactHessian{T, VT}}
    aug_com::MT
    data::OTData{T}
    K::OTCondensedBlock{T}
    reg::Vector{T}
    pr_diag::VT
    du_diag::VT
    l_diag::Vector{T}
    u_diag::Vector{T}
    l_lower::Vector{T}
    u_lower::Vector{T}
    ind_lb::Vector{Int}
    ind_ub::Vector{Int}
    # Buffers
    z1::VT # L x S
    z2::VT # L x S
    b1::VT # L
    linear_solver::LS
end

function MadNLP.create_kkt_system(
    ::Type{OTKKTSystem{T, VI, VT, MT}},
    cb::MadNLP.AbstractCallback{T, Vector{T}},
    ind_cons,
    linear_solver;
    opt_linear_solver=MadNLP.default_options(linear_solver),
    hessian_approximation=MadNLP.ExactHessian,
) where {T, VI, VT, MT}
    # Load original model
    nlp = cb.nlp
    L, S = nlp.data.L, nlp.data.S
    nlb, nub = length(ind_cons.ind_lb), length(ind_cons.ind_ub)

    aug_com = zeros(L, L)
    pr_diag = zeros(L*S + 1)
    du_diag = zeros(L+1)
    reg = zeros(L*S + 1)
    l_diag = zeros(nlb)
    u_diag = zeros(nub)
    l_lower = zeros(nlb)
    u_lower = zeros(nub)
    K = OTCondensedBlock(nlp.data, zeros(L*S))

    linear_solver_ = linear_solver(aug_com; opt=opt_linear_solver)

    z1  = zeros(L*S)
    z2  = zeros(L*S)
    b1  = zeros(S)
    return OTKKTSystem{T, VI, VT, MT, typeof(linear_solver_)}(
        aug_com, nlp.data, K,
        reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        ind_cons.ind_lb, ind_cons.ind_ub,
        z1, z2, b1,
        linear_solver_,
    )
end

MadNLP.num_variables(kkt::OTKKTSystem) = kkt.data.L
MadNLP.get_hessian(kkt::OTKKTSystem) = kkt.aug_com
MadNLP.get_jacobian(kkt::OTKKTSystem) = 0

function update_internal!(kkt::OTKKTSystem)
    L, S = kkt.data.L, kkt.data.S
    kkt.K.B.Σ .= kkt.pr_diag[1:L*S]
    kkt.K.ρ[1] = kkt.pr_diag[L*S+1]
    init!(kkt.K)
    return
end

function MadNLP.initialize!(kkt::OTKKTSystem{T}) where T
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.l_lower, zero(T))
    fill!(kkt.u_lower, zero(T))
    fill!(kkt.l_diag, one(T))
    fill!(kkt.u_diag, one(T))
    return
end

function MadNLP.eval_jac_wrapper!(
    solver::MadNLP.MadNLPSolver,
    kkt::OTKKTSystem,
    x::MadNLP.PrimalVector{T},
) where T
    return
end

function MadNLP.eval_lag_hess_wrapper!(
    solver::MadNLP.MadNLPSolver,
    kkt::OTKKTSystem,
    x::MadNLP.PrimalVector{T},
    l::Vector{T};
    is_resto=false,
) where T
    return
end

# TODO
function _mul_expanded!(y::AbstractVector, kkt::OTKKTSystem, x::AbstractVector, alpha::Number, beta::Number)
    S, L = kkt.data.S, kkt.data.L
    A1 = RowOperator{Float64}(L, S)
    A2 = ColumnOperator{Float64}(L, S)

    n = L * S

    yx = view(y, 1:n)
    yy = view(y, n+3:n+2+L)
    xx = view(x, 1:n)
    xy = view(x, n+3:n+2+L)
    Ax = kkt.b1

    # / x
    mul!(Ax, A2, xx)
    mul!(yx, A2', Ax, alpha, beta)
    mul!(yx, A1', xy, alpha, 1.0)
    axpy!(x[n+2] * alpha, kkt.data.d, yx)
    # / s
    y[n+1] = - alpha * x[n+2] + beta * y[n+1]
    # / z
    y[n+2] = alpha * (dot(kkt.data.d, xx) - x[n+1]) + beta * y[n+2]
    # / y
    mul!(yy, A1, xx, alpha, beta)
    return y
end

function MadNLP.mul!(y::VT, kkt::OTKKTSystem, x::VT, alpha, beta) where VT <: MadNLP.AbstractKKTVector
    _mul_expanded!(y.values, kkt, x.values, alpha, beta)
    MadNLP._kktmul!(y, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return y
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

function MadNLP.build_kkt!(kkt::OTKKTSystem)
    L, S = kkt.data.L, kkt.data.S
    update_internal!(kkt)
    # Load buffers
    z1 = kkt.z1
    z2 = kkt.z2
    b1 = kkt.b1
    # Assemble Schur-complement
    @time assemble_multithreads!(kkt.aug_com, kkt.K)
    # Symmetrize
    @inbounds for i in 1:L, j in (i+1):L
        kkt.aug_com[i, j] = kkt.aug_com[j, i]
    end
    return
end

function MadNLP.solve!(kkt::OTKKTSystem, w::MadNLP.AbstractKKTVector)
    # Build reduced KKT vector.
    MadNLP.reduce_rhs!(w.xp_lr, MadNLP.dual_lb(w), kkt.l_diag, w.xp_ur, MadNLP.dual_ub(w), kkt.u_diag)

    w_ = MadNLP.primal_dual(w)
    S, L = kkt.data.S, kkt.data.L
    z1 = kkt.z1  # buffer L x S

    nx = S * L

    Σ = view(kkt.pr_diag, 1:nx)   # X^{-1} S
    ρ = kkt.pr_diag[nx+1]         # t / r

    # Decompose left-hand side
    Δyy = zeros(L)

    # Decompose right-hand side
    wx = view(w_, 1:nx)
    wr = w_[nx+1]
    wt = w_[nx+2]
    wy = view(w_, nx+3:nx+2+L)

    # Schur-complement w.r.t inequality constraint
    v1 = wx .+ (ρ * wt + wr) .* kkt.data.d

    Δyy .= wy
    ldiv!(z1, kkt.K, v1)
    mul!(Δyy, kkt.data.A1, z1, 1.0, -1.0)

    status = MadNLP.solve!(kkt.linear_solver, Δyy)
    wy .= Δyy

    # Recover Δx
    z1 .= v1
    mul!(z1, kkt.data.A1', wy, -1.0, 1.0)
    ldiv!(wx, kkt.K, z1)

    # Recover solution (Δr, Δt)
    Δr = -wt + dot(kkt.data.d, wx)
    w_[nx+1] = Δr             # Δr
    w_[nx+2] = -wr + ρ * Δr   # Δt

    MadNLP.finish_aug_solve!(kkt, w)
    return true
end

