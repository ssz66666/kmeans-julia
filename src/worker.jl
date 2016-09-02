# k-means julia implementation
# Sizhe Sun ssz66666@gmail.com
# 2016-09-03

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
  # sums, counts the arrays to store the returned 'sums' (d * k) and 'counts' (n) result as stated above.
  # k_range     the range of 'k', i.e. previous centres points to be used.
  # n_range     the range of 'n', i.e. data points to be used
                        
  fill!(sums, zero(eltype(sums)))
  fill!(counts, zero(eltype(counts)))
  for i in n_range
    rec = view(data, :, i)
    l_dist = typemax(eltype(data))
    currentAssignment = 0
    for j = k_range
      ctr = view(centres, :, j)
      c_dist = zero(T)
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
    for row = eachindex(current_col,current_sum)
       @inbounds current_sum[row] += current_col[row]
    end
    @inbounds counts[currentAssignment] += 1
  end
  return sums, counts
end
