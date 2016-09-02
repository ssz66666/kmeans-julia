module Serial_Kmeans

# Naïve implementation of K-Means algorithm
# v0.4
# 孙斯哲 Sizhe Sun 2016-09-03

export kmeans

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
    fill!(sums, zero(eltype(sums)))
    fill!(counts, zero(eltype(counts)))
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
  for i in n
    rec = view(data, :, i)
    l_dist = typemax(eltype(data))
    currentAssignment = 0
    for j = 1 : k
      ctr = view(centres, :, j)
      c_dist = zero(Float64)
      @simd for I = eachindex(rec,ctr)
         @inbounds reci = rec[I]
         @inbounds ctri = ctr[I]
         c_dist += (reci - ctri) ^ 2
      end
      @inbounds dmat[j,i] = c_dist
      c_dist < l_dist && (l_dist = c_dist; currentAssignment = j)
    end
    @inbounds assignments[i] = currentAssignment
    current_col = rec
    current_sum = view(sums, :, currentAssignment)
    @simd for row = eachindex(current_col,current_sum)
       @inbounds current_sum[row] += current_col[row]
    end
     @inbounds counts[currentAssignment] += 1
  end
  for ctr = 1 : k
    t_ctr = view(centres,:,ctr)
    t_sums = view(sums,:,ctr)
    t_counts = counts[ctr]
    @simd for dim = eachindex(t_ctr,t_sums)
      @inbounds t_ctr[dim] = t_sums[dim] / t_counts
    end
  end
end

function randomSelectCentroid(dataSet::AbstractArray{Float64}, k::Int)
  return dataSet[1:end, randperm(size(dataSet,2))][1:end,1:k]
end

end # end of module
