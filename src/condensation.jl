#=
    Compute inverse explicitly.
=#

function _sum_vector(Bd, i,S)
    s1 = 0.0
    @turbo for k in 1:S
        s1 += Bd[k + S * (i-1)]
    end
    return s1
end

# TODO: Store sum of pr_diag
function assemble_simd!(C::AbstractMatrix, K::OTCondensedBlock)
    L, S = K.data.L, K.data.S
    # We assume all inner values have been updated previously with init!
    # Load buffers
    d = K.data.d
    Bd = K.a3
    θ = K.B.invΣ       # Σ⁻¹
    buffer = K.a1
    D = K.B.V
    ρ = K.ρ[1]

    # NB: inverse operation is expensive
    # Evaluate coefficient for inequality inverse block
    γ = ρ / (1.0 + ρ * dot(d, Bd))      # dᵀ B⁻¹ d

    # Reset matrix
    fill!(C, 0.0)

    # Add Σ⁻¹ diagonal terms
    @inbounds for l in 1:L
        for s in 1:S
            C[l, l] += θ[s + S*(l-1)]
        end
    end

    @inbounds for i in 1:L
        for k in 1:S
            buffer[k] = θ[k + S * (i-1)] / D[k]
        end
        for j in 1:i
            acc = 0.0
            @turbo for k in 1:S
                acc += buffer[k] * θ[k + S*(j-1)]
            end
            C[i, j] -= acc
        end
    end

    @inbounds for k in 1:L
        buffer[k] = _sum_vector(Bd, k, S)
    end
    @inbounds for i in 1:L, j in 1:i
        C[i, j] -= γ * buffer[i] * buffer[j]
    end

    return
end

@kernel function _assemble_kernel!(
    C,
    @Const(θ),
    @Const(D),
    @Const(Bd),
    γ,
    L,
    S,
)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if i == j
            for s in 1:S
                C[i, i] += θ[s + S * (i - 1)]
            end
        end
        if j <= i
            acc = 0.0
            @simd for s in 1:S
                acc += θ[s + S * (i-1)] * θ[s + S * (j-1)] / D[s]
            end
            C[i, j] -= acc
            acc1, acc2 = 0.0, 0.0
            @simd for s in 1:S
                acc1 += Bd[s + S*(i-1)]
                acc2 += Bd[s + S*(j-1)]
            end
            C[i, j] -= γ * acc1 * acc2
        end
    end
end

function assemble_multithreads!(C::AbstractMatrix, K::OTCondensedBlock)
    L, S = K.data.L, K.data.S
    # We assume all inner values have been updated previously with init!
    # Load buffers
    d = K.data.d
    Bd = K.a1
    θ = K.a2
    buffer = K.a3
    D = K.B.V
    ρ = K.ρ[1]

    # NB: inverse operation is expensive
    @inbounds for i in 1:L*S
        θ[i] = 1.0 / K.B.Σ[i]
    end

    fill!(D, 1.0)
    @inbounds for s in 1:S, l in 1:L
        D[s] += θ[s + S*(l-1)]
    end

    # Evaluate coefficient for inequality inverse block
    ldiv!(Bd, K.B, d)                   # B⁻¹ d
    γ = ρ / (1.0 + ρ * dot(d, Bd))      # dᵀ B⁻¹ d

    # Reset matrix
    workgroup_size = 1024
    backend = CPU()
    fill!(C, 0.0)

    ndrange = (L, L)
    _assemble_kernel!(backend)(C, θ, D, Bd, γ, L, S, ndrange=ndrange)
    KA.synchronize(backend)

    return
end

