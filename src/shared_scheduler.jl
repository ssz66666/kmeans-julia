# k-means julia implementation
# Sizhe Sun ssz66666@gmail.com

importall Shared_Kmeans_Worker

function kmeans{T<:AbstractFloat, N<:Integer}(m::Matrix{T},
                                              k::N,
                                              iter_count::N=100,
                                              test_convergence::Bool=true,
                                              init_centres::AbstractMatrix{T}=randomSelectCentroid(m, k),
                                              worker_set::Vector{N}=workers())

  data::SharedArray{T, 2} = SharedArray(T, size(m), init=false, pids=worker_set)
  copy!(data, m)
  kmeans(data,k,iter_count,test_convergence,init_centres,worker_set)
end


function kmeans{T<:AbstractFloat, N<:Integer}(data::SharedArray{T, 2},
                                              k::N,
                                              iter_count::N=100,
                                              test_convergence::Bool=true,
                                              init_centres::AbstractMatrix{T}=randomSelectCentroid(m, k),
                                              worker_set::Vector{N}=workers())

  
  const d::N, n::N = size(data)
  const centres::SharedArray{T, 2} = SharedArray(T,(d,k),init=false, pids=worker_set)
  const dmat::SharedArray{T, 2} = SharedArray(T, (k,n), init=false, pids=worker_set)
  const assignments::SharedArray{N, 1} = SharedArray(N, (n...), init=false, pids=worker_set)
  const dist::SharedArray{UnitRange{N}, 1} = convert(SharedArray, Base.splitrange(n, length(worker_set)))
  const nw = length(dist)
  const sums::SharedArray{T, 2} = SharedArray(T,(d,k),init=false, pids=worker_set)
  const counts::SharedArray{N, 1} = SharedArray(N,(k...),init=false, pids=worker_set)
  const prev_cents::SharedArray{T, 2} = SharedArray(T,(d,k),init=false)
  copy!(centres, init_centres)
  init_worker(length(dist),worker_set,T,N,d,k)

  iter::N = 0
  while iter < iter_count
    get_next_centres!(worker_set, centres, data, dmat,
                                assignments, dist, sums, counts, k, d, n)
    if test_convergence && ==(centres,prev_cents)
      break
    end
    test_convergence && copy!(prev_cents, centres)
    iter += 1
  end
  ret = copy(sdata(assignments))
  finalize(centres);finalize(dmat);finalize(assignments);finalize(dist)
  finalize(sums);finalize(counts);finalize(prev_cents)
  ret
end


function get_next_centres!{T<:AbstractFloat, N<:Integer}(worker_set::Vector{N},
                                       centres::SharedArray{T, 2},
                                       data::SharedArray{T, 2},
                                       dmat::SharedArray{T, 2},
                                       assignments::SharedArray{N, 1},
                                       dist::SharedArray{UnitRange{N}, 1},
                                       sums::SharedArray{T, 2},
                                       counts::SharedArray{N, 1},
                                       k::N, d::N, n::N)
  nw = length(dist)
  fill!(sdata(sums), zero(eltype(sums)))
  fill!(sdata(counts), zero(eltype(counts)))
  prefs = Array{Future}(nw)
  for w = 1 : nw
    prefs[w] = remotecall(next_iteration, worker_set[w], centres,
                          data,dmat,assignments,dist,k,d,n,w)
  end
  for pref in prefs
    p_sum, p_count = fetch(pref)
    add_in_place(sums,p_sum);add_in_place(counts,p_count)
  end
  
  sums_to_centres!(centres, sums, counts, 1:d, 1:k)
  return centres
end

function init_worker{N<:Integer}(nw::N, worker_set::Vector{N}, ttype::Type, ntype::Type, d::N, k::N)

  @sync for i = 1:nw
    @async remotecall_wait(init_shared_worker,worker_set[i],ttype,ntype,d,k)
  end
end
