module Serial_Kmeans

# Naïve implementation of K-Means algorithm
# 孙斯哲 Sizhe Sun

export kmeans

include("utils.jl")
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

  iter::N = 0
  while iter < iter_count
    next_iteration(centres, data, dmat,
                                assignments, sums, counts, k, d, n)
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
                        k::N, d::N, n::N)
  next_iteration!(centres, data, dmat, assignments, sums, counts, 1:k, 1:n)
  sums_to_centres!(centres, sums, counts, 1:d, 1:k)
end

end # end of module
