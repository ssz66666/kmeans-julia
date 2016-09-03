include("distributed_worker.jl")

module Distributed_Kmeans

export kmeans, load_libraries

include("distributed_scheduler.jl")

function load_libraries(worker_set::Vector{Int}=workers())
  #cpath = pwd()
  f = """include("./../src/distributed_worker.jl");using Distributed_Kmeans_Worker"""
  @sync for w in worker_set
    expr = Expr(:call, :eval, :Main, Expr(:call, :remotecall_fetch, :eval,
                                      w, Expr(:call, :parse, f)))
    @async eval(Main, expr)
  end
end

end
