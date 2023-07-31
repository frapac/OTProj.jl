
struct OTData{T}
    S::Int
    L::Int
    A1::SparseMatrixCSC{T, Int}
    A2::SparseMatrixCSC{T, Int}
    w::Vector{T}
    q::Vector{T}
    d::Vector{T}
end

function OTData(w, q, d)
    S, L = length(w), length(q)
    A1 = kron(spdiagm(ones(L)), ones(1, S))
    A2 = kron(ones(1, L), spdiagm(ones(S)))
    return OTData(S, L, A1, A2, w, q, d)
end

function preprocess_image(raw)
    w = raw[:]
    return w ./ sum(w)
end

function compute_distance_matrix_grid(n, m)
    A1 = kron(ones(1, m), diagm(1:n))
    A2 = kron(diagm(1:m), ones(1, n))
    X = [A1; A2]
    return sqrt.(pairwise(Euclidean(), X; dims=2))
end

function compute_distance_2(n, m, distance)
    A1 = kron(ones(m), 1:n)
    A2 = kron(1:m, ones(n))
    X = [A1'; A2']

    N, M = size(X)

    D = zeros(M, M)
    for i in 1:M, j in 1:M
        xi = @view X[:, i]
        xj = @view X[:, j]
        D[i, j] = norm(xi .- xj, distance)
    end
    return D
end

function OTData(src_folder, class, img1, img2, resolution; distance=2)
    img1_file = joinpath(src_folder, class, "data$(resolution)_$(img1).csv")
    img2_file = joinpath(src_folder, class, "data$(resolution)_$(img2).csv")
    # Import data
    w_raw = readdlm(img1_file, ',')
    q_raw = readdlm(img2_file, ',')
    # Preprocessing
    w = preprocess_image(w_raw)
    q = preprocess_image(q_raw)
    # Generate problem
    S, L = length(w), length(q)
    D = compute_distance_2(resolution, resolution, distance)
    d = D[:] ./ median(D)
    return OTData(w, q, d)
end

function Base.show(io::IO, data::OTData)
    M = convert(Int, sqrt(length(data.w)))
    print("$M x $M optimal transport problem")
    return
end

