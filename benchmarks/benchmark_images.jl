
using OTProj
using NLPModels
using MadNLP
using DelimitedFiles
using JuMP
using Gurobi

DATA = joinpath(@__DIR__, "..", "..", "data", "Data")
RESULTS = joinpath(@__DIR__, "..", "results")

GRB_ENV = nothing
if isnothing(GRB_ENV)
    GRB_ENV = Gurobi.Env(output_flag=0)
end

const NTHREADS = 12

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
        richardson_tol=1e-12,
        richardson_max_iter=5,
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

function benchmark(class, resolution; distance=2)
    results = zeros(45, 12)
    optimizer = () -> Gurobi.Optimizer(GRB_ENV)

    cnt = 1
    for k in 1001:1010, l in (k+1):1010
        @info "Instance $k x $l"
        data_ref = OTProj.OTData(DATA, class, k, l, 32; distance=distance)

        # OT
        modelLP = OTProj.build_optimal_transport(data_ref)
        JuMP.set_optimizer(modelLP, optimizer)
        JuMP.set_attribute(modelLP, "Method", 1)       # dual simplex
        JuMP.set_attribute(modelLP, "Presolve", 0)     # presolve=off
        JuMP.optimize!(modelLP)
        valLP = JuMP.objective_value(modelLP)

        data = OTProj.OTData(DATA, class, k, l, resolution; distance=distance)
        # Gurobi
        modelQP = OTProj.build_projection_wasserstein_qp(data, 0.5 * valLP)
        JuMP.set_optimizer(modelQP, optimizer)
        JuMP.set_attribute(modelQP, "Threads", NTHREADS)
        JuMP.set_attribute(modelQP, "Method", 2)      # barrier
        JuMP.set_attribute(modelQP, "Presolve", 0)    # presolve=off
        t_gurobi = @elapsed JuMP.optimize!(modelQP)
        solGur = JuMP.objective_value(modelQP)
        pGur = JuMP.value.(modelQP[:p])

        # Bundle
        t_bundle = @elapsed begin
            pBundle, solBundle, iterBundle = OTProj.proj_wass_bundle(data, 0.5 * valLP, optimizer)
        end

        # IPM
        pIpm, solIpm, iterIpm, t_ipm = solve_ipm(data, 0.5 * valLP)

        #
        results[cnt, 1] = k
        results[cnt, 2] = l
        results[cnt, 3] = t_gurobi
        results[cnt, 4] = solGur
        results[cnt, 5] = t_bundle
        results[cnt, 6] = solBundle
        results[cnt, 7] = iterBundle
        results[cnt, 8] = norm(pGur .- pBundle, Inf) / norm(pGur, Inf)
        results[cnt, 9] = t_ipm
        results[cnt, 10] = solIpm
        results[cnt, 11] = iterIpm
        results[cnt, 12] = norm(pGur .- pIpm, Inf) / norm(pGur, Inf)

        cnt += 1
    end
    return results
end

results = benchmark("Shapes", 32)
# classes = [
#     "CauchyDensity",
#     "ClassicImages",
#     "GRFmoderate",
#     "GRFrough",
#     "GRFsmooth",
#     "LogGRF",
#     "LogitGRF",
#     "MicroscopyImages",
#     "Shapes",
#     "WhiteNoise",
# ]
# resolution = 32

# for class in classes
#     @info "\n $class"
#     results = benchmark(class, resolution)
#     dest = joinpath(RESULTS, "$(class)_$(resolution).txt")
#     writedlm(dest, results)
# end

