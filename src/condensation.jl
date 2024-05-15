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

function assemble_preconditioner!(P::AbstractVector, K::OTCondensedBlock)
    L, S = K.data.L, K.data.S
    θ = K.B.invΣ       # Σ⁻¹
    @inbounds for l in 1:L
        for s in 1:S
            P[l] += θ[s + S*(l-1)]
        end
    end
end

# TODO: Store sum of pr_diag
function assemble_kkt_exact!(C::AbstractMatrix, K::OTCondensedBlock)
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
        acc = 0.0
        @turbo for s in 1:S
             acc += θ[s + S*(l-1)]
        end
        C[l, l] = acc
    end

    @inbounds for k in 1:L
        buffer[k] = _sum_vector(Bd, k, S)
    end
    @inbounds for i in 1:L, j in 1:i
        C[i, j] -= γ * buffer[i] * buffer[j]
    end

    # Matrix multiplication
    k = 0
    @inbounds for i in 1:L
        shift = S * (i-1)
        for j in 1:S
            buffer[k += 1] = θ[j + shift] / D[j]
        end
    end
    U = reshape(buffer, S, L)
    V = reshape(θ, S, L)
    mul!(C, V', U, -1.0, 1.0)

    return
end

function assemble_kkt_sparse!(C::AbstractMatrix, K::OTCondensedBlock, Bp::Vector{Int}, Bj::Vector{Int}, Bx::Vector{Float64})
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
        for l in 1:S
            buffer[l] = 0.0
        end
        for nc in Bp[i]:Bp[i+1]-1
            k = Bj[nc]
            buffer[k] = Bx[nc] / D[k]
        end
        for j in 1:i
            for nd in Bp[j]:Bp[j+1]-1
                k = Bj[nd]
                C[i, j] -= buffer[k] * Bx[nd]
            end
        end
    end

    @inbounds for k in 1:L
        buffer[k] = _sum_vector(Bd, k, S)
    end
    @inbounds for i in 1:L, j in 1:i
        v = γ * buffer[i] * buffer[j]
        C[i, j] -= v
    end

    return
end

