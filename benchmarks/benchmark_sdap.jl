
using LinearAlgebra
using OTProj
using NLPModels
using MadNLP
using MadQP
using DelimitedFiles
using JuMP
using Gurobi
using MAT

DATA = joinpath(@__DIR__, "..", "..", "sdap")
RESULTS = joinpath(@__DIR__, "..", "results")

GRB_ENV = nothing
# if isnothing(GRB_ENV)
#     GRB_ENV = Gurobi.Env(output_flag=0)
# end

const NTHREADS = 12

function import_data(src_dir)
    file = matopen(joinpath(DATA, src_dir, "1-data2FP.mat"))
    w = read(file, "w")[:]
    delta = read(file, "delta")
    d = read(file, "d")[:]
    S = read(file, "S") |> Int
    L = read(file, "L") |> Int
    q = ones(L) ./ L
    close(file)

    return OTProj.OTData(w, q, d), delta
end

function update_w!(data, src_dir, n)
    file = matopen(joinpath(DATA, src_dir, "$(n)-data2FP.mat"))
    w = read(file, "w")[:]
    close(file)
    data.w .= w
    return
end

function solve_ipm(data::OTProj.OTData, delta)
    max_iter = 300

    # Scaling
    a = 10.0
    b = 1.0
    data2 = OTProj.OTData(data.w .* a, data.q .* a, data.d ./ (a * b))
    nlp = OTProj.OTCompactModel(data2, delta / b; eval_hessian=false)

    # Build MadNLP
    KKT = OTProj.OTKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}
    solver = MadNLP.MadNLPSolver(
        nlp;
        kkt_system=KKT,
        mu_min=1e-11,
        max_iter=100,
        dual_initialized=true,
        nlp_scaling=false,
        tol=1e-8,
        linear_solver=LapackCPUSolver,
        lapack_algorithm=MadNLP.CHOLESKY,
        print_level=MadNLP.ERROR,
        hessian_constant=true,
        jacobian_constant=true,
    )

    BLAS.set_num_threads(NTHREADS)
    MadQP.solve!(solver)

    sol = solver.obj_val
    p = (data.A2 * solver.x.x) ./ a
    niter = solver.cnt.k
    println(solver.status, " ", niter)
    t_ipm = solver.cnt.total_time
    return p, sol, niter, t_ipm
end

function benchmark_sdap(dir, N)
    results = zeros(45, 12)
    # optimizer = () -> Gurobi.Optimizer(GRB_ENV)

    data, delta = import_data(dir)
    cnt = 1
    for i in 1:N
        @info "$(dir)-$(i)"
        update_w!(data, dir, i)

        # OT

        # Gurobi
        # modelQP = OTProj.build_projection_wasserstein_qp(data, 0.5 * valLP)
        # JuMP.set_optimizer(modelQP, optimizer)
        # JuMP.set_attribute(modelQP, "Threads", NTHREADS)
        # JuMP.set_attribute(modelQP, "Method", 2)      # barrier
        # JuMP.set_attribute(modelQP, "Presolve", 0)    # presolve=off
        # t_gurobi = @elapsed JuMP.optimize!(modelQP)
        # solGur = JuMP.objective_value(modelQP)
        # pGur = JuMP.value.(modelQP[:p])

        # # Bundle
        # t_bundle = @elapsed begin
        #     pBundle, solBundle, iterBundle = OTProj.proj_wass_bundle(data, 0.5 * valLP, optimizer)
        # end

        # IPM
        pIpm, solIpm, iterIpm, t_ipm = solve_ipm(data, delta)

        #
        results[cnt, 1] = i
        # results[cnt, 3] = t_gurobi
        # results[cnt, 4] = solGur
        # results[cnt, 5] = t_bundle
        # results[cnt, 6] = solBundle
        # results[cnt, 7] = iterBundle
        # results[cnt, 8] = norm(pGur .- pBundle, Inf) / norm(pGur, Inf)
        results[cnt, 9] = t_ipm
        results[cnt, 10] = solIpm
        results[cnt, 11] = iterIpm
        # results[cnt, 12] = norm(pGur .- pIpm, Inf) / norm(pGur, Inf)

        cnt += 1
    end
    return results
end


classes = [
    "sdap_1000_125",
    # "sdap_1000_250",
    # "sdap_1000_500",
    # "sdap_10648_1000",
    # "sdap_8000_1000",
    # "sdap_8000_500",
]
N = 10

for class in classes
    @info "\n $class"
    results = benchmark_sdap(class, N)
    dest = joinpath(RESULTS, "$(class).txt")
    writedlm(dest, results)
end

