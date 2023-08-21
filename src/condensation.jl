#=
    Compute inverse explicitly.
=#

function _sum_vector(Bd, i, j, S)
    s1 = 0.0
    s2 = 0.0
    @turbo for k in 1:S
        s1 += Bd[k + S * (i-1)]
        s2 += Bd[k + S * (j-1)]
    end
    return s1 * s2
end

# TODO: Store sum of pr_diag
function assemble!(C::AbstractMatrix, K::OTCondensedBlock)
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

    @inbounds for i in 1:L, j in 1:i
        acc = _sum_vector(Bd, i, j, S)
        C[i, j] -= γ * acc
    end

    return
end

