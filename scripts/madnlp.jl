
using NLPModels
using MadNLP
using Profile
using MadQP
using LinearAlgebra
using SparseArrays

reduction = true
max_iter = 100

h = 0.1
a = 1.0e1
b = 0.1
# a,b=1, 1

data2 = OTProj.OTData(data.w .* a, data.q .* a, data.d ./ (a * b) )
nlp = OTProj.OTCompactModel(data2, h * valLP / b; eval_hessian=false)

###

KKT = OTProj.OTKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}

solver = MadNLP.MadNLPSolver(
    nlp;
    kkt_system=KKT,
    mu_min=1e-11,
    max_iter=100,
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

Profile.clear()
Profile.init()
# MadQP.initialize!(solver)
MadQP.solve!(solver)

K = solver.kkt.aug_com

theta = solver.kkt.K.B.invΣ
