

using Revise

using LinearAlgebra
using SparseArrays
using OTProj
using JuMP
using HiGHS
using Gurobi

using NLPModels
using MadNLP
using MadQP


DATA = joinpath(@__DIR__, "..", "..", "data", "Data")

function load_data(class, img1, img2, resolution)
    return OTProj.OTData(DATA, class, img1, img2, resolution; distance=2)
end

function solve_ot(data)
    modelLP = OTProj.build_optimal_transport(data)
    JuMP.set_optimizer(modelLP, HiGHS.Optimizer)
    JuMP.optimize!(modelLP)
    return JuMP.objective_value(modelLP)
end

function solve_proj(
    data,
    valLP,
    h;
    a=1e1,
    b=0.1,
    nthreads=12,
    max_iter=200,
)
    a, b = 1e1, 0.1
    data2 = OTProj.OTData(data.w .* a, data.q .* a, data.d ./ (a * b) )
    nlp = OTProj.OTCompactModel(data2, h * valLP / b; eval_hessian=false)

    KKT = OTProj.OTKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}

    solver = MadNLP.MadNLPSolver(
        nlp;
        kkt_system=KKT,
        mu_min=1e-11,
        max_iter=max_iter,
        dual_initialized=true,
        nlp_scaling=false,
        tol=1e-8,
        linear_solver=LapackCPUSolver,
        lapack_algorithm=MadNLP.CHOLESKY,
        print_level=MadNLP.INFO,
        # bound_push=1e1,
        # bound_fac=0.02,
        hessian_constant=true,
        jacobian_constant=true,
    )
    BLAS.set_num_threads(nthreads)
    MadQP.solve!(solver)
    return solver
end

function benchmark_delta()
    class = "ClassicImages"
    img1 = "1001"
    img2 = "1003"
    resolution = 32
    data = load_data(class, img1, img2, resolution)
    valLP = solve_ot(data)
    a, b = 1e1, 0.1

    all_deltas = 0.1:0.1:1.0
    nexp = length(all_deltas)
    results = zeros(nexp, 3)
    for (cnt, h) in enumerate(all_deltas)
        solver = solve_proj(data, valLP, h; a=a, b=b)

        x = solver.x.x
        sol = solver.obj_val / a^2
        x_active = findall(x .> 1e-8)

        results[cnt, 1] = h
        results[cnt, 2] = sol
        results[cnt, 3] = length(x_active)
    end
    return results
end

function benchmark_threads()
    class = "ClassicImages"
    k = "1001"
    l = "1003"
    resolution = 64
    env = Gurobi.Env(output_flag=0)
    optimizer = () -> Gurobi.Optimizer(env)

    cnt = 1
    data_ref = OTProj.OTData(DATA, class, k, l, 32; distance=distance)
    # OT
    modelLP = OTProj.build_optimal_transport(data_ref)
    JuMP.set_optimizer(modelLP, optimizer)
    JuMP.set_attribute(modelLP, "Method", 1)       # dual simplex
    JuMP.set_attribute(modelLP, "Presolve", 0)     # presolve=off
    JuMP.optimize!(modelLP)
    valLP = JuMP.objective_value(modelLP)

    data = OTProj.OTData(DATA, class, k, l, resolution)

    nthreads = [1, 2, 4, 6, 8, 12]
    nexp = length(nthreads)

    results = zeros(nexp, 12)
    for (cnt, nth) in enumerate(nthreads)
        # Gurobi
        modelQP = OTProj.build_projection_wasserstein_qp(data, 0.5 * valLP)
        JuMP.set_optimizer(modelQP, optimizer)
        JuMP.set_attribute(modelQP, "Threads", nth)
        JuMP.set_attribute(modelQP, "Method", 2)      # barrier
        JuMP.set_attribute(modelQP, "Presolve", 0)    # presolve=off
        t_gurobi = @elapsed JuMP.optimize!(modelQP)

        # IPM
        solver = solve_proj(data, valLP, 0.5; nthreads=nth)
        t_ipm = solver.cnt.total_time

        results[cnt, 1] = nth
        results[cnt, 3] = t_gurobi
        results[cnt, 9] = t_ipm
    end
    return results
end

function decompose_ipm_solving_time()
    class = "ClassicImages"
    img1 = "1001"
    img2 = "1003"
    resolution = 32
    data = load_data(class, img1, img2, resolution)
    valLP = solve_ot(data)
    a, b = 1e1, 0.1

    solver = solve_proj(data, valLP, 0.5; max_iter=10)
    timings = MadNLP.timing_linear_solver(solver)

end

function plot_heatmap(data, x)
    X = reshape(x, data.L, data.S)
    fig = plot()
    heatmap!(log10.(max.(X, 1e-8)))
    title!("Solution returned by IPM")
    return fig
end


results = benchmark_delta()





# x = solver.x.x
# fig = plot_heatmap(data, x)
# savefig("fig/heatmap_h0.10.png")


