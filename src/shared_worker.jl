# Naïve implementation of K-Means algorithm
# with experimental multi-process support implemented using SharedArray
# worker module
# 孙斯哲 Sizhe Sun

module Shared_Kmeans_Worker

export next_iteration, init_shared_worker

include("worker.jl")

type KMeansWorker{T<:AbstractFloat,N<:Integer}
  sums::AbstractArray{T,2}
  counts::AbstractArray{N,1}
end

type KMeansProc
  worker_on_this_proc::KMeansWorker
  KMeansProc() = new()
end

const global wproc = KMeansProc()

function next_iteration{T<:AbstractFloat,N<:Integer}(centres::SharedArray{T, 2},
                        data::SharedArray{T, 2},
                        dmat::SharedArray{T, 2},
                        assignments::SharedArray{N, 1},
                        dist::SharedArray{UnitRange{N}, 1},
                        k::N, d::N, n::N, pid::N)
  global wproc                      
  p_n = dist[pid]
  next_iteration!(centres,
                  data,
                  dmat,
                  assignments,
                  wproc.worker_on_this_proc.sums,
                  wproc.worker_on_this_proc.counts,
                  1:k,
                  p_n)
end

function init_shared_worker{N<:Integer}(ttype::Type,ntype::Type,d::N,k::N)
  global wproc
  wproc.worker_on_this_proc = KMeansWorker(Array{ttype}(d,k),Array{ntype}(k))
end

end
