# Naïve implementation of K-Means algorithm
# with experimental multi-process support implemented using SharedArray
# 孙斯哲 Sizhe Sun

include("shared_worker.jl")

module Shared_Kmeans

export kmeans, load_libraries

include("utils.jl")
include("shared_scheduler.jl")

function load_libraries(worker_set::Vector{Int}=workers())
  f = """include("./../src/shared_worker.jl");using Shared_Kmeans_Worker"""
  @sync for w in worker_set
    expr = Expr(:call, :eval, :Main, Expr(:call, :remotecall_fetch, :eval,
                                      w, Expr(:call, :parse, f)))
    @async eval(Main, expr)
  end
end

end