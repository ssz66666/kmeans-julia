# k-means julia implementation
# Sizhe Sun ssz66666@gmail.com

function randomSelectCentroid{T<:AbstractFloat,N<:Integer}(dataSet::AbstractArray{T}, k::N)
  # randomly select k initial cluster centres from dataSet
  
  return dataSet[1:end, randperm(size(dataSet,2))][1:end,1:k]
end

function add_in_place(m1::AbstractArray,m2::AbstractArray)
  # equivalent to m1 += m2
  
  for i in eachindex(m1)
      m1[i] = m1[i] + m2[i]
  end
end

function sums_to_centres!{T<:AbstractFloat,N<:Integer}(
                          centres::AbstractArray{T, 2},
                          sums::AbstractArray{T, 2},
                          counts::AbstractArray{N, 1},
                          d_range::UnitRange{N},
                          k_range::UnitRange{N})
  # centres[:,ctr] = sums[:,ctr] / counts[ctr]
  # this is to calculate new centres from calculated sum(s) of coordinates of data points from each 'cluster'.
  
  for dim = d_range
    for ctr = k_range
      @inbounds centres[dim,ctr] = sums[dim,ctr] / counts[ctr]
    end
  end
end