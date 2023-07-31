
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

# C = alpha * A / B + beta * C
function scaldiv!(C, A, B, alpha, beta, N)
    @inbounds for i in 1:N
        C[i] = beta * C[i] + alpha * A[i] / B[i]
    end
end

function axdiv!(A, B)
    N = length(A)
    for i in 1:N
        @inbounds A[i] = A[i] / B[i]
    end
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
    @variable(model, p[1:L])
    @constraint(model, dot(data.d, x) <= delta)
    @constraint(model, data.A1 * x == data.q)
    @constraint(model, data.A2 * x == p)
    @objective(model, Min, 0.5 * dot(p - data.w, p - data.w))

    return model
end

