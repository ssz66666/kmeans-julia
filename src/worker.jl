# k-means julia core implementation
# Sizhe Sun ssz66666@gmail.com

function next_iteration!{T<:AbstractFloat,N<:Integer}(centres::AbstractArray{T,2},
                        data::AbstractArray{T,2},
                        dmat::AbstractArray{T,2},
                        assignments::AbstractArray{N,1},
                        sums::AbstractArray{T,2},
                        counts::AbstractArray{N,1},
                        k_range::UnitRange{N},
                        n_range::UnitRange{N})

  # Calculate the 'coordinates' of the next centres points from data of previous iteration using K-means algorithm.
  # Returns the sums of coordinate values in each dimension of points which are classified as in one cluster (d * k), 'sums',
  # and the number of points in each cluster (n), 'counts'.
  # Notice that the new centres can be calculated by applying sums[dimension, :] / counts[dimension]
  #
  # centres     current 'coordinates' of centroids/centres of clusters of points, d * k
  # data        'coordinates' of all original data points, d * n
  # dmat        the distance matrix, k * n
  # assignments the array of assignments for each data point, n
  # sums, counts the arrays to store the returned 'sums' (d * k) and 'counts' (k) result as stated above.
  # k_range     the range of 'k', i.e. previous centres points to be used.
  # n_range     the range of 'n', i.e. data points to be used
                        
  fill!(sums, zero(eltype(sums)))
  fill!(counts, zero(eltype(counts)))
  
  d_range = Base.OneTo(size(data,1))
  l_dist::T = typemax(T)
  c_dist::T = zero(T)
  currentAssignment::N = zero(N)
  
  for n in n_range
    l_dist = typemax(T)
    currentAssignment = zero(N)
    for k in k_range
      c_dist = zero(T)
      for d in d_range
        @inbounds c_dist += (data[d,n] - centres[d,k]) ^ 2
      end
      dmat[k,n] = c_dist
      c_dist < l_dist && (l_dist = c_dist; currentAssignment = k)
    end
    @inbounds assignments[n] = currentAssignment
    for d in d_range
      @inbounds sums[d,currentAssignment] += data[d,n]
    end
    @inbounds counts[currentAssignment] += 1
  end
  return sums, counts
end

