
using Revise
using NLPModels
using MadNLP
using MadQP
using MAT
using OTProj
using LinearAlgebra
using SparseArrays

reduction = true
max_iter = 100

h = 1.0
a = 1.0e1
b = 1

src_dir = "/home/fpacaud/dev/ot/sdap/sdap_10648_1000"
src_file = "1-data2FP.mat"

file = matopen(joinpath(src_dir, src_file))
w = read(file, "w")[:]
delta = read(file, "delta")
d = read(file, "d")[:]
S = read(file, "S") |> Int
L = read(file, "L") |> Int
q = ones(L) ./ L
close(file)

data = OTProj.OTData(
    w .* a,
    q .* a,
    d ./ (a * b),
)
nlp = OTProj.OTCompactModel(data, h * delta / b; eval_hessian=false)

####

KKT = OTProj.OTKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}

solver = MadNLP.MadNLPSolver(
    nlp;
    kkt_system=KKT,
    mu_min=1e-11,
    max_iter=300,
    dual_initialized=true,
    nlp_scaling=false,
    richardson_tol=1e-12,
    richardson_max_iter=5,
    tol=1e-8,
    # richardson_acceptable_tol=1e-1,
    linear_solver=LapackCPUSolver,
    lapack_algorithm=MadNLP.CHOLESKY,
    print_level=MadNLP.INFO,
    # bound_push=1e2,
    # bound_fac=0.02,
    hessian_constant=true,
    jacobian_constant=true,
    # mu_linear_decrease_factor=.99,
    # mu_superlinear_decrease_power=1.05,
)

BLAS.set_num_threads(12)

MadQP.solve!(solver)
theta = solver.kkt.K.B.invΣ
CC = reshape(theta, L, S)
heatmap(CC)

