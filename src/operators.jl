
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

function LinearAlgebra.mul!(y::AbstractVector{T}, A::RowOperator{T}, x::AbstractVector{T}, alpha::T, beta::T) where T
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

function LinearAlgebra.mul!(y::AbstractVector{T}, A::Adjoint{T, RowOperator{T}}, x::AbstractVector{T}, alpha::T, beta::T) where T <: Number
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

function LinearAlgebra.mul!(y::AbstractVector{T}, A::ColumnOperator{T}, x::AbstractVector{T}, alpha::T, beta::T) where T
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

function LinearAlgebra.mul!(y::AbstractVector{T}, A::Adjoint{T, ColumnOperator{T}}, x::AbstractVector{T}, alpha::T, beta::T) where T <: Number
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
    # Buffers
    a1::Vector{T}  # L x S
    b1::Vector{T}  # S
    # Inverse
    V::Vector{T}  # S
end

function OTQuadraticBlock(data::OTData, sigma)
    L, S = data.L, data.S
    a1 = zeros(L*S)
    b1 = zeros(S)
    V  = zeros(S)
    return OTQuadraticBlock(
        data, sigma, a1, b1, V,
    )
end

Base.size(B::OTQuadraticBlock) = (B.data.S * B.data.L, B.data.S * B.data.L)
LinearAlgebra.issymmetric(B::OTQuadraticBlock) = true

# Evaluate (I + A₂ Σ⁻¹ A₂')
function _update_inner_block!(B::OTQuadraticBlock)
    S, L = B.data.S, B.data.L
    fill!(B.V, 1.0)
    @inbounds for s in 1:S
        for l in 1:L
            B.V[s] += 1.0 / B.Σ[s + S*(l-1)]
        end
    end
    return
end

function init!(B::OTQuadraticBlock)
    _update_inner_block!(B)
end

function LinearAlgebra.mul!(y, B::OTQuadraticBlock, x)
    y .= x .* B.Σ                         # Σ x
    mul!(B.b1, B.data.A2, x)              # A₂ x
    mul!(y, B.data.A2', B.b1, 1.0, 1.0)   # Σx + A₂ᵀA₂ x
    return y
end


# Implement Woodbury formula
function LinearAlgebra.ldiv!(y, B::OTQuadraticBlock, x)
    A2 = ColumnOperator{Float64}(B.data.L, B.data.S)
    LS = B.data.L * B.data.S
    scaldiv!(y, x, B.Σ, 1.0, 0.0, LS)     # Σ⁻¹ x
    mul!(B.b1, A2, y)                     # A₂ Σ⁻¹ x
    axdiv!(B.b1, B.V)
    # B.b1 ./= B.V                        # V⁻¹A₂Σ⁻¹ x
    mul!(B.a1, A2', B.b1)                 # A₂ᵀV⁻¹A₂Σ⁻¹ x
    scaldiv!(y, B.a1, B.Σ, -1.0, 1.0, LS) # Σ⁻¹ x - Σ⁻¹A₂ᵀV⁻¹A₂Σ⁻¹ x
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
end

function LinearAlgebra.mul!(y, K::OTCondensedBlock, x)
    mul!(y, K.B, x)                     # B x
    rdx = dot(K.data.d, x) * K.ρ[1]     # dᵀx * ρ
    y .+= rdx .* K.data.d               # B x + ddᵀx * ρ
    return y
end

# Implement Sherman-Morisson formula
function LinearAlgebra.ldiv!(y, K::OTCondensedBlock, x)
    ldiv!(y, K.B, x)                       # B⁻¹ x
    ldiv!(K.a3, K.B, K.data.d)             # B⁻¹ d
    r = 1.0 / K.ρ[1] + dot(K.data.d, K.a3) # ρ + d' B⁻¹ d
    dBx = dot(K.data.d, y)                 # dᵀB⁻¹ x
    y .-= K.a3 .* (dBx / r)                # B⁻¹ x - (B⁻¹ddᵀB⁻¹ x) / r
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
