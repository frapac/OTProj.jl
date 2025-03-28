
struct SoftMaxModel{T}
    data::OTData{T}
    hash::Vector{UInt64}
    expu::Matrix{T}
    sumexp::Vector{T}
    u_max::Vector{T}
end
function SoftMaxModel(data::OTData{T}) where T
    return SoftMaxModel{T}(data, zeros(UInt64, 1), zeros(T, data.S, data.L), zeros(T, data.L), zeros(T, data.L))
end

function _expterm!(res, u, d, el_max, eps, i, S)
    @turbo for j in 1:S
        res[j, i] = exp((u[j] - d[(i-1)*S + j] - el_max) / eps)
    end
    return
end

function _update!(smd::SoftMaxModel, u, eps)
    hash_u = hash(u)
    if hash_u != smd.hash[1]
        data = smd.data
        S, L = data.S, data.L
        @inbounds for i in 1:L
            # Compute maximum elements
            el_max = -Inf
            for j in 1:S
                elj = u[j] - data.d[(i-1)*S + j]
                if elj > el_max
                    el_max = elj
                end
            end
            # Compute exponential elements with sum
            _expterm!(smd.expu, u, data.d, el_max, eps, i, S)
            sumexp = 0.0
            for j in 1:S
                sumexp += smd.expu[j, i]
            end
            smd.sumexp[i] = sumexp
            smd.u_max[i] = el_max
        end
        smd.hash[1] = hash_u
    end
end

function objective(u, p, eps, smd::SoftMaxModel)
    _update!(smd, u, eps)
    data = smd.data
    res = 0.0
    @inbounds for i in 1:data.L
        res += (smd.u_max[i] + eps * log(smd.sumexp[i])) * data.q[i]
    end
    return res - dot(p, u)
end

function gradient!(grad, u, p, eps, smd::SoftMaxModel)
    _update!(smd, u, eps)
    data = smd.data
    fill!(grad, 0.0)
    for i in 1:data.L, j in 1:data.S
        @inbounds grad[j] += smd.expu[j, i] / smd.sumexp[i] * data.q[i]
    end
    grad .-= p
    return grad
end

function bundle_problem(optimizer, w, pk, delta, G, rhs)
    nw = length(w)

    model = Model(optimizer)
    @variable(model, p[1:nw] >= 0)
    @constraint(model, sum(p) == 1.0)
    con_bundle = @constraint(model, G*p .<= delta .+ rhs)
    @constraint(model, dot(w .- pk, p) <= dot(w .- pk, pk))
    @objective(model, Min, 0.5 * dot(p - w, p - w))
    JuMP.set_silent(model)
    JuMP.optimize!(model)

    return (JuMP.value.(p), JuMP.dual.(con_bundle))
end

function ots(model::JuMP.Model, u, p, target)
    JuMP.set_normalized_rhs.(model[:con2], p)
    JuMP.optimize!(model)
    return (JuMP.objective_value(model), JuMP.value.(model[:x]), JuMP.dual.(model[:con2]))
end

function ots(smd::SoftMaxModel, u, p, target; eps=1e-4, eta=1e-5, kMax=10, iMax=100)
    data = smd.data
    S, L = data.S, data.L

    tot_it = 0

    f = x -> objective(x, p, eta, smd)
    g! = (g, x) -> gradient!(g, x, p, eta, smd)

    g = zeros(S)
    optimizer = LBFGSB.L_BFGS_B(S, 10)
    bounds = zeros(3, S)
    lb, ub = fill(-Inf, S), fill(Inf, S)
    bounds[2, :] .= lb
    bounds[3, :] .= ub
    bounds[1, :] .= LBFGSB.typ_bnd.(lb, ub)

    for k in 1:kMax
        smd.hash[1] = UInt64(0)
        vals, u = optimizer(f, g!, u, bounds; m=10, maxiter=iMax, pgtol=eps, iprint=-1)
        it_cur = optimizer.isave[30]

        tot_it += it_cur
        # We exit as soon as the constraint is satisfied
        if -vals > target
            break
        end
    end

    u = u .- mean(u)
    v = zeros(L)
    for j in 1:L
        dj = @view(data.d[(j-1)*S+1:j*S])
        v[j] = minimum(dj .- u)
    end
    val = dot(p, u) + dot(data.q, v)

    return val, v, u
end

function proj_wass_bundle(
    data::OTData, delta, optimizer;
    eps0=1e-4,
    eta0=1e-4,
    tol=1e-6,
    kMax=100,
    verbose=true,
)
    L, S = data.L, data.S
    p = projection_simplex(data.w)

    eta, eps = eta0, eps0

    smd = SoftMaxModel(data)

    nbun = kMax
    u = zeros(S)
    target = delta + 2*eps

    c, _, u = ots(smd, u, p, target; eps=1e-5)
    G = zeros(1, S)
    G[1, :] .= u

    rhs = Float64[dot(u, p) - c]

    if verbose
        @printf(" %3s %7s %11s %5s %7s %5s \n", "it", "obj", "inf_pr", "eps", "eta", "nK")
    end

    n_iter = 0
    for k in 1:kMax
        p, mu = bundle_problem(optimizer, data.w, p, delta, G, rhs)
        c, _, u = ots(smd, u, p, target; eps=eps, eta=eta)

        gap = abs(c - delta)
        if gap <= tol
            break
        elseif gap <= eps
            eps *= 0.1
            eta *= 0.1
        end

        nk = length(mu)

        if nk >= nbun
            mu ./= sum(mu)
            J = findall(abs.(mu) .> 1e-7)
            na = length(J)

            if na >= nk
                J = J[2:end]
            end

            G = G[J, :]
            rhs = rhs[J]
        end
        G = [G; u']
        rhs = [rhs; dot(u, p) - c]
        verbose && @printf(" %3d %2.5e %2.1e %2.1e %2.1e %3d \n", k, 0.5*norm(p - data.w)^2, c - delta, eps, eta, nk)
        n_iter += 1
    end
    return p, 0.5 * norm(p - data.w)^2, n_iter
end

