module Threaded_Kmeans

# Naïve implementation of K-Means algorithm,
# with experimental threading support
# v0.4
# 孙斯哲 Sizhe Sun 2016-09-03

export kmeans

include("worker.jl")

function kmeans{T<:AbstractFloat,N<:Integer}(
  data::AbstractMatrix{T},             # matrix containing original data
  k::N,                         # number of clusters
  iter_count::N=100,
  test_convergence::Bool=true,
  init_centres::AbstractMatrix{T}=randomSelectCentroid(data,k))
                                  # initial centroids of clusters


  #single process version of kmeans
  # n - number of records
  # d - number of dimensions
  const d::N, n::N = size(data)
  if size(init_centres) != (d, k) error("incorrect initial centres"); end
  const centres::Array{T, 2} = copy(init_centres)
  const dmat::Array{T, 2} = Array(T, (k,n))
  const assignments::Array{N, 1} = Array(N, (n...))
  const sums::Array{T, 2} = Array(T, (d,k))
  const counts::Array{N, 1} = Array(N, (k...))
  const prev_cents::Array{T, 2} = Array(T,(d,k))
  const dist::Vector{UnitRange{N}} = Base.splitrange(n, Threads.nthreads())

  iter::N = 0
  while iter < iter_count
    fill!(sums, zero(eltype(sums)))
    fill!(counts, zero(eltype(counts)))
    next_iteration(centres, data, dmat,
                                assignments, sums, counts, k, d, n, dist)
    if test_convergence && ==(centres,prev_cents)
      break
    end
    test_convergence && copy!(prev_cents, centres)
    iter += 1
  end
  finalize(centres);finalize(dmat)
  finalize(sums);finalize(counts);finalize(prev_cents)
  assignments
end

function next_iteration{T<:AbstractFloat,N<:Integer}(centres::Array{T, 2},
                        data::Array{T, 2},
                        dmat::Array{T, 2},
                        assignments::Array{N, 1},
                        sums::Array{T, 2},
                        counts::Array{N, 1},
                        k::N, d::N, n::N, dist::Vector{UnitRange{N}})
  
  psums = Array{Array{T, 2}}(length(dist))
  pcounts = Array{Array{N, 1}}(length(dist))
  
  Threads.@threads for i in 1:length(dist)
    psums[i] = Array{T}(d, k)
    pcounts[i] = Array{N}(k)
    next_iteration!(centres, data, dmat, assignments, psums[i], pcounts[i], 1:k, dist[i])
  end
  for t in 1:length(dist)
    sums += psums[t]
    counts += pcounts[t]
  end
  
  Threads.@threads for ctr = 1 : k
    t_ctr = view(centres,:,ctr)
    t_sums = view(sums,:,ctr)
    t_counts = counts[ctr]
    for dim = eachindex(t_ctr,t_sums)
      @inbounds t_ctr[dim] = t_sums[dim] / t_counts
    end
  end
end

function randomSelectCentroid(dataSet::AbstractArray{Float64}, k::Int)
  return dataSet[1:end, randperm(size(dataSet,2))][1:end,1:k]
end

end # end of module
