
using OTProj

DATA = joinpath(@__DIR__, "..", "..", "data", "Data")

function solve_ipm(data::OTProj.OTData, delta)
    max_iter = 300

    nlp = OTProj.OTCompactModel(data, delta; eval_hessian=false)
    # Build MadNLP
    madnlp_options = Dict{Symbol, Any}()
    madnlp_options[:linear_solver] = LapackCPUSolver
    madnlp_options[:lapack_algorithm] = MadNLP.CHOLESKY
    madnlp_options[:dual_initialized] = true
    madnlp_options[:max_iter] = max_iter
    madnlp_options[:print_level] = MadNLP.ERROR
    madnlp_options[:tol] = 1e-9
    opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)

    KKT = OTProj.OTKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}
    solver = MadNLP.MadNLPSolver{Float64, KKT}(nlp, opt_ipm, opt_linear; logger=logger)
    MadNLP.solve!(solver)

    sol = solver.obj_val
    p = data.A2 * solver.x.x
    niter = solver.cnt.k
    t_ipm = solver.cnt.total_time
    return p, sol, niter, t_ipm
end

function benchmark(class, resolution; distance=2)
    results = zeros(45, 10)
    optimizer = () -> Gurobi.Optimizer(GRB_ENV)

    cnt = 1
    for k in 1001:1010, l in (k+1):1010
        @info "Instance $k x $l"
        data = OTProj.OTData(DATA, class, k, l, resolution; distance=distance)

        # OT
        modelLP = OTProj.build_optimal_transport(data)
        JuMP.set_optimizer(modelLP, optimizer)
        JuMP.set_attribute(modelLP, "Method", 1)       # dual simplex
        JuMP.set_attribute(modelLP, "Presolve", 0)     # presolve=off
        JuMP.optimize!(modelLP)
        valLP = JuMP.objective_value(modelLP)

        # Gurobi
        modelQP = OTProj.build_projection_wasserstein_qp(data, 0.5 * valLP)
        JuMP.set_optimizer(modelQP, optimizer)
        t_gurobi = @elapsed JuMP.optimize!(modelQP)
        solGur = JuMP.objective_value(modelQP)

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
        results[cnt, 8] = t_ipm
        results[cnt, 9] = solIpm
        results[cnt, 10] = iterIpm

        cnt += 1
    end
    return results
end

results = benchmark("Shapes", 32)
