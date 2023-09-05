
using NLPModels
using MadNLP

include("solver.jl")
reduction = true
max_iter = 200

a = 1.0e1
b = 1

data2 = OTProj.OTData(data.w .* a, data.q .* a, data.d ./ (a * b) )
nlp = OTProj.OTCompactModel(data2, 0.5 * valLP / b; eval_hessian=false)
# Build MadNLP
madnlp_options = Dict{Symbol, Any}()
madnlp_options[:linear_solver] = LapackCPUSolver
madnlp_options[:lapack_algorithm] = MadNLP.CHOLESKY
madnlp_options[:dual_initialized] = true
madnlp_options[:max_iter] = max_iter
madnlp_options[:print_level] = MadNLP.DEBUG
madnlp_options[:hessian_constant] = true
madnlp_options[:jacobian_constant] = true
madnlp_options[:tol] = 1e-9

### REFERENCE
madnlp_options[:mu_min] = 1e-11
madnlp_options[:bound_fac] = 0.2
madnlp_options[:bound_push] = 100.0
madnlp_options[:mu_linear_decrease_factor] = 0.99
madnlp_options[:mu_superlinear_decrease_power] = 1.05

###
opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)

KKT = OTProj.OTKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}
solver = MadNLP.MadNLPSolver{Float64, KKT}(nlp, opt_ipm, opt_linear; logger=logger)
# l_solve!(solver)
MadNLP.solve!(solver)
println(solver.status)

pM = nlp.data.A2 * solver.x.x
