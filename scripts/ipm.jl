
using OTProj
using JuMP
using Gurobi

GRB_ENV = nothing
if isnothing(GRB_ENV)
    GRB_ENV = Gurobi.Env(output_flag=0)
end

# Params
DATA = joinpath(@__DIR__, "..", "..", "data", "Data")
# Picture class
class = "Shapes"
# Origin picture
img1 = "1001"
# Destination picture
img2 = "1010"
# Resolution ∈ [32, 64, 128, 256, 512]
resolution = 32

data = OTProj.OTData(DATA, class, img1, img2, resolution)

modelLP = OTProj.build_optimal_transport(data)
JuMP.set_optimizer(modelLP, () -> Gurobi.Optimizer(GRB_ENV))
JuMP.set_attribute(modelLP, "Method", 1)       # dual simplex
JuMP.set_attribute(modelLP, "Presolve", 0)     # presolve=off
JuMP.optimize!(modelLP)
valLP = JuMP.objective_value(modelLP)

if resolution <= 32
    modelQP = OTProj.build_projection_wasserstein_qp(data, 0.5 * valLP)
    JuMP.set_optimizer(modelQP, () -> Gurobi.Optimizer(GRB_ENV))
    JuMP.set_attribute(modelQP, "OutputFlag", 1)
    JuMP.optimize!(modelQP)
    solGur = JuMP.objective_value(modelQP)
end

