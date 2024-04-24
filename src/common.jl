
function projection_simplex(w)
    nw = length(w)
    u = sort(w; rev=true)
    ucum = cumsum(u)
    ind = collect(1:nw)
    ks = findall((ucum .- 1.0) ./ ind .<= u)
    K = maximum(ks)
    τ = (sum(u[1:K]) - 1.0) / K
    return max.(w .- τ, 0.0)
end

function build_optimal_transport(data::OTData)
    S, L = data.S, data.L

    model = Model()
    @variable(model, 0.0 <= x[1:L*S])
    @constraint(model, con1, data.A1 * x == data.q)
    @constraint(model, con2, data.A2 * x == data.w)
    @objective(model, Min, dot(data.d, x))

    return model
end

function build_projection_wasserstein_qp(data::OTData, delta)
    S, L = data.S, data.L

    model = Model()
    @variable(model, 0.0 <= x[1:L*S])
    @variable(model, p[1:S])
    @constraint(model, dot(data.d, x) <= delta)
    @constraint(model, data.A1 * x == data.q)
    @constraint(model, data.A2 * x == p)
    @objective(model, Min, 0.5 * dot(p - data.w, p - data.w))

    return model
end

function theta2csr(θ, tau, L, S)
    nnz_ = 0
    Bp = zeros(Int, L+1)
    @inbounds for i in 1:L, j in 1:S
        if (θ[j + S * (i-1)] > tau)
            nnz_ += 1
            Bp[i] += 1
        end
    end

    Bj = zeros(Int, nnz_)
    Bx = zeros(Float64, nnz_)

    cumsum = 1
    @inbounds for i in 1:L
        tmp = Bp[i]
        Bp[i] = cumsum
        cumsum += tmp
    end
    Bp[L+1] = nnz_ + 1

    @inbounds for i in 1:L, j in 1:S
        if (θ[j + S * (i-1)] > tau)
            dest = Bp[i]
            Bj[dest] = j
            Bx[dest] = θ[j + S * (i-1)]
            Bp[i] += 1
        end
    end
    last = 1
    @inbounds for i in 1:L
        tmp = Bp[i]
        Bp[i] = last
        last = tmp
    end
    return (Bp, Bj, Bx)
end

function count_nnz(θ, tau, L, S)
    cnt = 0
    @inbounds for i in 1:L, j in 1:S
        if (θ[j + S * (i-1)] > tau)
            cnt += 1
        end
    end
    return cnt
end

function MadNLP.UnreducedKKTVector(
    values, n::Int, m::Int, nlb::Int, nub::Int, ind_lb, ind_ub
)
    x = MadNLP._madnlp_unsafe_wrap(values, n + m) # Primal-Dual
    xp = MadNLP._madnlp_unsafe_wrap(values, n) # Primal
    xl = MadNLP._madnlp_unsafe_wrap(values, m, n+1) # Dual
    xzl = MadNLP._madnlp_unsafe_wrap(values, nlb, n + m + 1) # Lower bound
    xzu = MadNLP._madnlp_unsafe_wrap(values, nub, n + m + nlb + 1) # Upper bound

    xp_lr = view(xp, ind_lb)
    xp_ur = view(xp, ind_ub)

    return MadNLP.UnreducedKKTVector(values, x, xp, xp_lr, xp_ur, xl, xzl, xzu)
end

