
#=
    A₁
=#

struct RowOperator{T}
    L::Int
    S::Int
end
Base.size(A::RowOperator) = (A.L, A.S * A.L)
LinearAlgebra.issymmetric(A::RowOperator) = false
LinearAlgebra.adjoint(A::RowOperator{T}) where T = LinearAlgebra.Adjoint{T, RowOperator{T}}(A)

function LinearAlgebra.mul!(y::AbstractVector{T}, A::RowOperator{T}, x::AbstractVector{T}) where T
    S, L = A.S, A.L
    fill!(y, zero(T))
    for l in 1:L
        @inbounds for s in 1:S
            y[l] += x[s + S*(l-1)]
        end
    end
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::RowOperator{T}, x::AbstractVector{T}, alpha::Number, beta::Number) where T
    S, L = A.S, A.L
    @inbounds for l in 1:L
        y[l] *= beta
    end
    for l in 1:L
        @inbounds for s in 1:S
            y[l] += alpha * x[s + S*(l-1)]
        end
    end
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::Adjoint{T, RowOperator{T}}, x::AbstractVector{T}) where T
    S, L = A.parent.S, A.parent.L
    for l in 1:L
        @turbo for s in 1:S
            y[s + S*(l-1)] = x[l]
        end
    end
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::Adjoint{T, RowOperator{T}}, x::AbstractVector{T}, alpha::Number, beta::Number) where T <: Number
    S, L = A.parent.S, A.parent.L
    for l in 1:L
        @turbo for s in 1:S
            y[s + S*(l-1)] = alpha * x[l] + beta * y[s + S*(l-1)]
        end
    end
end

#=
    A₂
=#

struct ColumnOperator{T}
    L::Int
    S::Int
end
Base.size(A::ColumnOperator) = (A.S, A.S * A.L)
LinearAlgebra.adjoint(A::ColumnOperator{T}) where T = LinearAlgebra.Adjoint{T, ColumnOperator{T}}(A)
LinearAlgebra.issymmetric(A::ColumnOperator) = false

function LinearAlgebra.mul!(y::AbstractVector{T}, A::ColumnOperator{T}, x::AbstractVector{T}) where T
    S, L = A.S, A.L
    fill!(y, zero(T))
    for l in 1:L
        @turbo for s in 1:S
            y[s] += x[s + S*(l-1)]
        end
    end
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::ColumnOperator{T}, x::AbstractVector{T}, alpha::Number, beta::Number) where T
    S, L = A.S, A.L
    @inbounds for s in 1:S
        y[s] *= beta
    end
    for l in 1:L
        @turbo for s in 1:S
            y[s] += alpha * x[s + S*(l-1)]
        end
    end
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::Adjoint{T, ColumnOperator{T}}, x::AbstractVector{T}) where T <: Number
    S, L = A.parent.S, A.parent.L
    for l in 1:L
        @turbo for s in 1:S
            y[s + S*(l-1)] = x[s]
        end
    end
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::Adjoint{T, ColumnOperator{T}}, x::AbstractVector{T}, alpha::Number, beta::Number) where T <: Number
    S, L = A.parent.S, A.parent.L
    for l in 1:L
        @turbo for s in 1:S
            y[s + S*(l-1)] = alpha * x[s] + beta * y[s + S * (l-1)]
        end
    end
end

#=
    B = A2 * A2' + Σ
=#
struct OTQuadraticBlock{T}
    data::OTData{T}
    Σ::Vector{T}
    invΣ::Vector{T}
    # Buffers
    a1::Vector{T}  # L x S
    b1::Vector{T}  # S
    # Inverse
    V::Vector{T}  # S
end

function OTQuadraticBlock(data::OTData, sigma)
    L, S = data.L, data.S
    invΣ = similar(sigma)
    a1 = zeros(L*S)
    b1 = zeros(S)
    V  = zeros(S)
    return OTQuadraticBlock(
        data, sigma, invΣ, a1, b1, V,
    )
end

Base.size(B::OTQuadraticBlock) = (B.data.S * B.data.L, B.data.S * B.data.L)
LinearAlgebra.issymmetric(B::OTQuadraticBlock) = true

# Evaluate (I + A₂ Σ⁻¹ A₂')
function _inv_sigma!(dest::Vector{T}, src::Vector{T}) where T
    @turbo for i in eachindex(src)
        dest[i] = one(T) / src[i]
    end
end

function _update_inner_block!(B::OTQuadraticBlock)
    S, L = B.data.S, B.data.L
    _inv_sigma!(B.invΣ, B.Σ)
    fill!(B.V, 1.0)
    @inbounds for s in 1:S
        for l in 1:L
            B.V[s] += B.invΣ[s + S*(l-1)]
        end
    end
    return
end

function init!(B::OTQuadraticBlock)
    _update_inner_block!(B)
end

function LinearAlgebra.mul!(y, B::OTQuadraticBlock, x)
    A2 = ColumnOperator{Float64}(B.data.L, B.data.S)
    # Σ x
    @turbo for k in eachindex(y)
        y[k] = x[k] * B.Σ[k]
    end
    mul!(B.b1, A2, x)              # A₂ x
    mul!(y, A2', B.b1, 1.0, 1.0)   # Σx + A₂ᵀA₂ x
    return y
end

# Implement Woodbury formula
function _solve!(B::OTQuadraticBlock, x)
    L, S = B.data.L, B.data.S
    θ = B.invΣ

    fill!(B.b1, 0.0)
    # A₂ Σ⁻¹ x
    @inbounds for l in 1:L
        @turbo for s in 1:S
            B.b1[s] += θ[s + S * (l-1)] * x[s + S * (l-1)]
        end
    end
    # V⁻¹A₂Σ⁻¹ x
    @inbounds for s in 1:S
        B.b1[s] = B.b1[s] / B.V[s]
    end
    # Σ⁻¹ x - Σ⁻¹A₂ᵀV⁻¹A₂Σ⁻¹ x
    @inbounds for l in 1:L
        @turbo for s in 1:S
            θi = θ[s + S * (l-1)]
            x[s + S * (l-1)] = θi * (x[s + S * (l-1)] - B.b1[s])
        end
    end
    return x
end

function LinearAlgebra.ldiv!(y, B::OTQuadraticBlock, x)
    L, S = B.data.L, B.data.S
    copyto!(y, x)
    _solve!(B, y)
    return y
end

#=
    K = ρ ddᵀ + B
=#
struct OTCondensedBlock{T}
    data::OTData{T}
    B::OTQuadraticBlock{T}
    ρ::Vector{T}
    # Buffers
    a1::Vector{T}   # L x S
    a2::Vector{T}   # L x S
    a3::Vector{T}   # L x S
end

function OTCondensedBlock(data::OTData, sigma)
    L, S = data.L, data.S
    B = OTQuadraticBlock(data, sigma)
    ρ = ones(1)
    a1 = zeros(L*S)
    a2 = zeros(L*S)
    a3 = zeros(L*S)
    return OTCondensedBlock(data, B, ρ, a1, a2, a3)
end

Base.size(K::OTCondensedBlock) = (K.data.S * K.data.L, K.data.S * K.data.L)
LinearAlgebra.issymmetric(K::OTCondensedBlock) = true

function init!(K::OTCondensedBlock)
    init!(K.B)
    ldiv!(K.a3, K.B, K.data.d)          # B⁻¹ d
    return
end

function LinearAlgebra.mul!(y, K::OTCondensedBlock, x)
    mul!(y, K.B, x)                     # B x
    rdx = dot(K.data.d, x) * K.ρ[1]     # dᵀx * ρ
    @turbo for k in eachindex(y)
        y[k] += rdx * K.data.d[k]               # B x + ddᵀx * ρ
    end
    return y
end

# Implement Sherman-Morisson formula
function LinearAlgebra.ldiv!(y, K::OTCondensedBlock, x)
    r = 1.0 / K.ρ[1] + dot(K.data.d, K.a3) # ρ + d' B⁻¹ d
    ρ = K.ρ[1]

    w = K.a2
    y .= 0
    copyto!(w, x)
    # Custom iterative refinement
    for i in 1:10
        # B⁻¹ x
        _solve!(K.B, w)
        # dᵀB⁻¹ x
        dBx = dot(K.data.d, w)
        # B⁻¹ x - (B⁻¹ddᵀB⁻¹ x) / r
        axpy!(-dBx / r, K.a3, w)
        axpy!(1.0, w, y)

        mul!(w, K, y)
        w .= x .- w

        if norm(w) <= 1e-6
            break
        end
    end

    return y
end

#=
    A₁ K⁻¹ A₁ᵀ
=#

struct NormalBlockOperator{T}
    K::OTCondensedBlock{T}
    A1::RowOperator{T}
    z1::Vector{T}
    z2::Vector{T}
end

function NormalBlockOperator(K::OTCondensedBlock{T}) where T
    L, S = K.data.L, K.data.S
    A1 = RowOperator{T}(L, S)
    return NormalBlockOperator{T}(K, A1, zeros(L*S), zeros(L*S))
end

Base.size(N::NormalBlockOperator) = (N.A1.L, N.A1.L)
Base.eltype(N::NormalBlockOperator{T}) where T = T

function LinearAlgebra.mul!(y, N::NormalBlockOperator, x)
    mul!(N.z1, N.A1', x)
    ldiv!(N.z2, N.K, N.z1)
    mul!(y, N.A1, N.z2)
    return y
end
