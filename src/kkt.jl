
#=
    OTKKTSystem
=#

struct OTKKTSystem{T, VI, VT, MT, LS, LS2} <: MadNLP.AbstractReducedKKTSystem{T, VT, MT, MadNLP.ExactHessian{T, VT}}
    aug_com::MT
    data::OTData{T}
    K::OTCondensedBlock{T}
    AKA::NormalBlockOperator{T}
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
    z3::VT # L
    b1::VT # S
    linear_solver::LS
    cg_solver::LS2
    strategy::Symbol
    sparsity::Ref{T}
    threshold::T
end

function MadNLP.create_kkt_system(
    ::Type{OTKKTSystem{T, VI, VT, MT}},
    cb::MadNLP.AbstractCallback{T, Vector{T}},
    ind_cons,
    linear_solver;
    opt_linear_solver=MadNLP.default_options(linear_solver),
    hessian_approximation=MadNLP.ExactHessian,
    strategy=:exact,
    threshold=0.00,
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
    AKA = NormalBlockOperator(K)

    linear_solver_ = linear_solver(aug_com; opt=opt_linear_solver)
    cg_solver = if strategy == :mixed
        Krylov.CgSolver(L, L, VT)
    else
        nothing
    end


    z1  = zeros(L*S)
    z2  = zeros(L*S)
    z3  = zeros(L)
    b1  = zeros(S)
    return OTKKTSystem{T, VI, VT, MT, typeof(linear_solver_), typeof(cg_solver)}(
        aug_com, nlp.data, K, AKA,
        reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        ind_cons.ind_lb, ind_cons.ind_ub,
        z1, z2, z3, b1,
        linear_solver_,
        cg_solver,
        strategy,
        Ref(0.1),
        threshold,
    )
end

MadNLP.num_variables(kkt::OTKKTSystem) = kkt.data.L
MadNLP.get_hessian(kkt::OTKKTSystem) = kkt.aug_com
MadNLP.get_jacobian(kkt::OTKKTSystem) = 0

Base.eltype(kkt::OTKKTSystem{T}) where T = T

function Base.size(kkt::OTKKTSystem)
    L, S = kkt.data.L, kkt.data.S
    n_ineq = 1
    n_eq = L
    n_ub = length(kkt.u_lower)
    n_lb = length(kkt.l_lower)
    n = L*S + 2*n_ineq + n_eq + n_ub + n_lb
    return (n, n)
end

function update_internal!(kkt::OTKKTSystem)
    L, S = kkt.data.L, kkt.data.S
    @turbo for i in 1:L*S
        kkt.K.B.Σ[i] = kkt.pr_diag[i]
    end
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

function LinearAlgebra.mul!(y::Vector{T}, kkt::OTKKTSystem, x::Vector{T}, alpha::Number, beta::Number) where T
    L, S = kkt.data.L, kkt.data.S
    n = L * S + 1
    m = L + 1
    nub = length(kkt.u_lower)
    nlb = length(kkt.l_lower)
    # Wrap vectors in memory
    y_ = MadNLP.UnreducedKKTVector(y, n, m, nlb, nub, kkt.ind_lb, kkt.ind_ub)
    x_ = MadNLP.UnreducedKKTVector(x, n, m, nlb, nub, kkt.ind_lb, kkt.ind_ub)

    mul!(y_, kkt, x_, alpha, beta)
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
    if kkt.strategy == :exact
        assemble_kkt_exact!(kkt.aug_com, kkt.K)
    elseif kkt.strategy == :mixed
        θ = kkt.K.B.invΣ
        tau = 1e-2
        cnt = count_nnz(θ, tau, L, S)
        kkt.sparsity[] = cnt / (L*S)
        println(cnt / (L*S))
        if kkt.sparsity[] < kkt.threshold
            Bp, Bj, Bx = theta2csr(θ, tau, L, S)
            assemble_kkt_sparse!(kkt.aug_com, kkt.K, Bp, Bj, Bx)
        end
    end
    # Symmetrize
    @inbounds for i in 1:L, j in (i+1):L
        kkt.aug_com[i, j] = kkt.aug_com[j, i]
    end
    return
end

function MadNLP.solve!(kkt::OTKKTSystem, w::MadNLP.AbstractKKTVector)
    S, L = kkt.data.S, kkt.data.L
    # Build reduced KKT vector.
    MadNLP.reduce_rhs!(w.xp_lr, MadNLP.dual_lb(w), kkt.l_diag, w.xp_ur, MadNLP.dual_ub(w), kkt.u_diag)

    A1 = RowOperator{Float64}(L, S)

    w_ = MadNLP.primal_dual(w)
    z1 = kkt.z1  # buffer L x S
    z2 = kkt.z2
    Δyy = kkt.z3

    nx = S * L

    ρ = kkt.pr_diag[nx+1]         # t / r

    # Decompose right-hand side
    wx = view(w_, 1:nx)
    wr = w_[nx+1]
    wt = w_[nx+2]
    wy = view(w_, nx+3:nx+2+L)

    # Schur-complement w.r.t inequality constraint
    z2 .= wx .+ (ρ * wt + wr) .* kkt.data.d

    Δyy .= wy
    ldiv!(z1, kkt.K, z2)
    mul!(Δyy, A1, z1, 1.0, -1.0)

    use_exact = (kkt.strategy == :exact) || (kkt.strategy == :mixed && kkt.sparsity[] < kkt.threshold)
    if use_exact
        b = copy(Δyy)
        status = MadNLP.solve!(kkt.linear_solver, Δyy)
        wy .= Δyy
        K = kkt.aug_com
    else
        @time Krylov.solve!(
            kkt.cg_solver,
            kkt.AKA,
            Δyy;
            atol=0.0,
            rtol=1e-10,
            verbose=0,
        )
        copyto!(wy, kkt.cg_solver.x)
        cg_iter = kkt.cg_solver.stats.niter
        println(cg_iter)
    end

    # Recover Δx
    z1 .= z2
    mul!(z1, A1', wy, -1.0, 1.0)
    ldiv!(wx, kkt.K, z1)

    # Recover solution (Δr, Δt)
    Δr = -wt + dot(kkt.data.d, wx)
    w_[nx+1] = Δr             # Δr
    w_[nx+2] = -wr + ρ * Δr   # Δt

    MadNLP.finish_aug_solve!(kkt, w)

    return true
end

function MadNLP.solve_refine_wrapper!(
    d,
    solver::MadNLP.MadNLPSolver{T, Vector{T}, Vector{Int}, KKT},
    p,
    w,
) where {T, KKT <: OTKKTSystem{T, Vector{Int}, Vector{T}, Matrix{T}}}
    result = false
    kkt = solver.kkt

    use_ir = ((kkt.strategy == :exact) || (kkt.strategy == :mixed && kkt.sparsity[] < kkt.threshold))
    use_ir &= false #(solver.mu <= 1e-10 && solver.cnt.k >= 20)
    solver.cnt.linear_solver_time += @elapsed begin
        if use_ir
            result =  MadNLP.solve_refine!(d, solver.iterator, p, w)
        else
            copyto!(MadNLP.full(d), MadNLP.full(p))
            MadNLP.solve!(kkt, d)
            # (sol, stats) = Krylov.gmres(solver.kkt, d.values; verbose=0, atol=1e-5, rtol=1e-9)
            # println(stats.niter)
            # d.values .= sol
        end
    end

    return result
end

# Fast MadNLP.set_aug_diagonal!
function MadNLP.set_aug_diagonal!(kkt::OTKKTSystem{T}, solver::MadNLP.MadNLPSolver{T}) where T
    x = MadNLP.full(solver.x)
    xl = MadNLP.full(solver.xl)
    xu = MadNLP.full(solver.xu)
    zl = MadNLP.full(solver.zl)
    zu = MadNLP.full(solver.zu)

    kkt.l_diag .= solver.xl_r .- solver.x_lr
    kkt.u_diag .= solver.x_ur .- solver.xu_r
    copyto!(kkt.l_lower, solver.zl_r)
    copyto!(kkt.u_lower, solver.zu_r)

    @inbounds for (k, i) in enumerate(kkt.ind_lb)
        kkt.pr_diag[i] = - kkt.l_lower[k] / kkt.l_diag[k]
    end
    @inbounds for (k, i) in enumerate(kkt.ind_ub)
        kkt.pr_diag[i] = - kkt.u_lower[k] / kkt.u_diag[k]
    end
    return
end

