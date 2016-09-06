module Distributed_Kmeans_Worker

# Naïve implementation of K-Means algorithm,
# with experimental parallel computing (multiple processes) support
# the worker module
# 孙斯哲 Sizhe Sun

include("worker.jl")

export next_iteration!, init_dist_worker, get_assignments

type KMeansWorker{T<:AbstractFloat,N<:Integer}
  data::AbstractArray{T,2}
  dmat::AbstractArray{T,2}
  assignments::AbstractArray{N,1}
  sums::AbstractArray{T,2}
  counts::AbstractArray{N,1}
  k::N
  d::N
  n::N
end

type KMeansProc
  worker_on_this_proc::KMeansWorker
  KMeansProc() = new()
end

const global wproc = KMeansProc()

function next_iteration!{T<:AbstractFloat}(centres::AbstractArray{T,2})
  global wproc
  worker_on_this_proc = wproc.worker_on_this_proc
  next_iteration!(centres,
                  worker_on_this_proc.data,
                  worker_on_this_proc.dmat,
                  worker_on_this_proc.assignments,
                  worker_on_this_proc.sums,
                  worker_on_this_proc.counts,
                  1:(worker_on_this_proc.k),
                  1:(worker_on_this_proc.n))
end

function init_dist_worker{T<:AbstractFloat,N<:Integer}(init_data::AbstractArray{T,2},
                    k::N,
                    d::N,
                    n::N)
  global wproc
  wproc.worker_on_this_proc = KMeansWorker(init_data,
                                           Array(T,k,n),
                                           Array(N,n),
                                           Array(T,d,k),
                                           Array(N,k),
                                           k,d,n)
end

function get_assignments()
  global wproc
  wproc.worker_on_this_proc.assignments
end

end
