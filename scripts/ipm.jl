
using Revise
using OTProj
using JuMP
using Ipopt
using Gurobi

GRB_ENV = nothing
# if isnothing(GRB_ENV)
#     GRB_ENV = Gurobi.Env(output_flag=0)
# end

# Params
DATA = joinpath(@__DIR__, "..", "..", "data", "Data")
# Picture class
class = "WhiteNoise"
# Origin picture
img1 = "1001"
# Destination picture
img2 = "1009"
# Resolution ∈ [32, 64, 128, 256, 512]
resolution = 32

data = OTProj.OTData(DATA, class, img1, img2, resolution; distance=2)

modelLP = OTProj.build_optimal_transport(data)
JuMP.set_optimizer(modelLP, Gurobi.Optimizer)
JuMP.set_attribute(modelLP, "OutputFlag", 0)
JuMP.set_attribute(modelLP, "Method", 2)
JuMP.optimize!(modelLP)
valLP = JuMP.objective_value(modelLP)

if resolution <= 32
    model1 = OTProj.build_projection_wasserstein_qp(data, 0.5 * valLP)
    JuMP.set_optimizer(model1, () -> Gurobi.Optimizer(GRB_ENV))
    JuMP.set_attribute(model1, "OutputFlag", 1)
    JuMP.set_attribute(model1, "Threads", 1)
    JuMP.optimize!(model1)
    solGur = JuMP.objective_value(model1)
    xGur = JuMP.value.(model1[:x])
    pGur = JuMP.value.(model1[:p])

    # model3 = OTProj.build_projection_wasserstein_qp(data, 0.5 * valLP)
    # JuMP.set_optimizer(model3, Ipopt.Optimizer)
    # JuMP.set_attribute(model3, "mehrotra_algorithm", "yes")
    # JuMP.optimize!(model3)
    # solIpopt = JuMP.objective_value(model3)
    # xIpopt = JuMP.value.(model3[:x])
    # pIpopt = JuMP.value.(model3[:p])
end

