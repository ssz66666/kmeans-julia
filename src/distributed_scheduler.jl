# k-means julia implementation
# Sizhe Sun ssz66666@gmail.com

importall Distributed_Kmeans_Worker

function kmeans{T<:AbstractFloat, N<:Integer}(data::Array{T, 2},
                                              k::N,
                                              iter_count::N=100,
                                              test_convergence::Bool=true,
                                              init_centres::AbstractMatrix{T}=randomSelectCentroid(data,k),
                                              worker_set::Vector{N}=workers())
  const nw::N, dist::Vector{UnitRange{N}} = dist_data(data,worker_set,k)
  const d::N, n::N = size(data)
  const assignments::Array{N, 1} = Array(N, (n...))
  const centres::Array{T, 2} = Array(T,(d,k))
  const sums::Array{T, 2} = Array(T,(d,k))
  const counts::Array{N, 1} = Array(N, k)
  test_convergence && const prev_cents::Array{T, 2} = copy(init_centres)
  copy!(centres, init_centres)

  iter::Int = 0
  while iter < iter_count
    get_next_centres!(worker_set, nw, centres, sums, counts)
    if test_convergence && ==(prev_cents,worker_set)
      break
    end
    test_convergence && copy!(prev_cents, centres)
    iter += 1
  end
  r = fetch_assignments(worker_set, assignments, dist)
  return r
end

function get_next_centres!{T<:AbstractFloat,N<:Integer}(worker_set::Vector{N},
                                       nw::N,
                                       centres::Array{T, 2},
                                       sums::Array{T,2},
                                       counts::Array{N,1})
  d,k = size(sums)
  fill!(sums,zero(eltype(sums)))
  fill!(counts,zero(eltype(counts)))
  prefs = Array{Future}(nw)
  for w = 1 : nw
      prefs[w] = remotecall(next_iteration!, worker_set[w], centres)
  end
  for pref in prefs
    p_sum, p_count = fetch(pref)
    add_in_place(sums,p_sum);add_in_place(counts,p_count)
  end
  sums_to_centres!(centres, sums, counts, 1:d, 1:k)
  return centres
end

function fetch_assignments{N<:Integer}(worker_set::Vector{N},
                                       assignments::Array{N, 1},
                                       dist::Vector{UnitRange{N}})
  for (i,range) = enumerate(dist)
    #copy!(assignments, range[1], remotecall_fetch(worker_set[i], get_assignments), 1)
    copy!(assignments, range[1], remotecall_fetch(get_assignments, worker_set[i]), 1)
  end
  return assignments
end

function dist_data{T<:AbstractFloat,N<:Integer}(m::AbstractArray{T,2},
                                                worker_set::AbstractArray{N,1},
                                                k::N)
  d, n = size(m)
  dist = Base.splitrange(n, length(worker_set))
  nw = length(dist)
  @sync for i = 1:nw
    p_data = view(m, :, dist[i])
    p_n = length(dist[i])
    #remotecall_wait(worker_set[i], init_dist_worker, p_data, k, d, p_n)
    @async remotecall_wait(init_dist_worker, worker_set[i], p_data, k, d, p_n)
  end
  return nw, dist
end
