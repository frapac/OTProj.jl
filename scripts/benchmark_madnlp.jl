
using NLPModels
using MadNLP
using OTProj
using DelimitedFiles

DATA = joinpath(@__DIR__, "..", "..", "data", "Data")
BUCKET = joinpath(@__DIR__, "..", "values")
RESULTS = joinpath(@__DIR__, "..", "results")

include(joinpath(@__DIR__, "solver.jl"))

function fetch_value(class, k, l, resolution)
    values = readdlm(joinpath(BUCKET, "$(class)_$(resolution).txt"))
    ncols = size(values, 1)
    valLP = NaN
    for c in 1:ncols
        if (values[c, 1] == k) && (values[c, 2] == l)
            valLP = values[c, 4]
            break
        end
    end
    return valLP
end

function solve_ipm(data::OTProj.OTData, delta)
    max_iter = 300

    # Scaling
    a = 10.0
    b = 1.0
    data2 = OTProj.OTData(data.w .* a, data.q .* a, data.d ./ (a * b))
    nlp = OTProj.OTCompactModel(data2, delta / b; eval_hessian=false)

    # Build MadNLP
    madnlp_options = Dict{Symbol, Any}()
    madnlp_options[:linear_solver] = LapackCPUSolver
    madnlp_options[:lapack_algorithm] = MadNLP.CHOLESKY
    madnlp_options[:dual_initialized] = true
    madnlp_options[:max_iter] = max_iter
    madnlp_options[:print_level] = MadNLP.ERROR
    madnlp_options[:tol] = 1e-8      # 1e-7
    madnlp_options[:mu_min] = 1e-11  # 1e-10
    madnlp_options[:bound_fac] = 0.2
    madnlp_options[:bound_push] = 100.0

    opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)

    KKT = OTProj.OTKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}
    solver = MadNLP.MadNLPSolver{Float64, KKT}(nlp, opt_ipm, opt_linear; logger=logger)
    l_solve!(solver)

    sol = solver.obj_val
    niter = solver.cnt.k
    t_ipm = solver.cnt.total_time
    return sol, niter, t_ipm
end

function benchmark_madnlp(class, resolution; distance=2)
    results = zeros(45, 5)
    cnt = 1
    for k in 1001:1010, l in (k+1):1010
        @info "Instance $k x $l"
        data = OTProj.OTData(DATA, class, k, l, resolution; distance=distance)
        valLP = fetch_value(class, k, l, resolution)
        solIpm, iterIpm, t_ipm = solve_ipm(data, 0.5 * valLP)
        results[cnt, 1] = k
        results[cnt, 2] = l
        results[cnt, 3] = t_ipm
        results[cnt, 4] = solIpm
        results[cnt, 5] = iterIpm
        cnt += 1
    end
    return results
end

classes = [
    # "CauchyDensity",
    # "ClassicImages",
    # "GRFmoderate",
    # "GRFrough",
    # "GRFsmooth",
    # "LogGRF",
    # "LogitGRF",
    # "MicroscopyImages",
    "Shapes",
    # "WhiteNoise",
]
resolution = 32

for class in classes
    @info "\n $class"
    results = benchmark_madnlp(class, resolution)
    dest = joinpath(RESULTS, "$(class)_$(resolution)_madnlp.txt")
    writedlm(dest, results)
end
