module OTProj

using DelimitedFiles
using Distances
using LinearAlgebra
using Printf
using SparseArrays
using Statistics

using JuMP

using KernelAbstractions
using LoopVectorization

const KA = KernelAbstractions

import NLPModels
import MadNLP
import LBFGSB

include("data.jl")
include("common.jl")
include("operators.jl")

# IPM
include("model.jl")
include("condensation.jl")
include("kkt.jl")

# Bundle
include("bundle.jl")

end # module OTProj
