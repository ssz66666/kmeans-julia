abstract ClusteringAlg
abstract KMeansClusteringAlg<:ClusteringAlg
abstract GMMClusteringAlg<:ClusteringAlg

abstract AbstractClusteringTestCase

type TestCase{N<:Integer}<:AbstractClusteringTestCase
  data_filename::AbstractString
  estimated_k::N #estimated value of k(number of clusters)
  args
end

type TestDataSet{N<:Integer}<:AbstractClusteringTestCase
  num::N # number of test cases
  tcases::Vector{TestCase}
end
