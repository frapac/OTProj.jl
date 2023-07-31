
using MadNLP

reduction = true
max_iter = 100

nlp = OTProj.OTCompactModel(data, 0.5 * valLP; eval_hessian=false)
# Build MadNLP
madnlp_options = Dict{Symbol, Any}()
madnlp_options[:linear_solver] = LapackCPUSolver
madnlp_options[:lapack_algorithm] = MadNLP.CHOLESKY
madnlp_options[:dual_initialized] = true
madnlp_options[:max_iter] = max_iter
madnlp_options[:print_level] = MadNLP.DEBUG
madnlp_options[:tol] = 1e-9
opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)

KKT = OTProj.OTKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}
solver = MadNLP.MadNLPSolver{Float64, KKT}(nlp, opt_ipm, opt_linear; logger=logger)
MadNLP.solve!(solver)

# pM = nlp.data.A2 * solver.x.x
