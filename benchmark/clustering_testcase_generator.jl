DEFAULT_NUMBER = 200
DEFAULT_DIMENSION = 2
DEFAULT_NUM_CLUSTERS = 4
DEFAULT_DISTANCE = 800.0
DEFAULT_RADIUS = 100.0
DEFAULT_L_DISTANCE = 0.0

function genTestCase(rec::Int64=DEFAULT_NUMBER, d::Int64=DEFAULT_DIMENSION,
                      k::Int64=DEFAULT_NUM_CLUSTERS,
                      u_dist::Float64=DEFAULT_DISTANCE,
                      u_radius::Float64=DEFAULT_RADIUS,
                      l_dist::Float64=DEFAULT_L_DISTANCE)
  # generate test case for clustering
  # rec - number of records to be generated
  # d - number of dimensions (properties of each record)
  # k - number of pre-determined cluster centres
  # u_dist - upper bound of distance between cluster centre and origin
  # u_radius - upper bound of radius of cluster, the definition of
  #            radius is Not accurate in this implementation
  # l_dist - lower bound of distance between cluster centre and origin,
  #          default value is 0.0

  mat = Array(Float64,rec,d)
  # generate cluster centres
  centre = rand(l_dist:u_dist, k, d)
  for i in 1:rec
    # randomly select a centroid
    ctrrow = rand(1:k)
    for j in range(1,d)
      mat[i,j] = centre[ctrrow,j] + rand(-u_radius:u_radius)
    end
  end
  return mat
end

function genTestFile(filename::AbstractString, rec::Int64=DEFAULT_NUMBER,
                      d::Int64=DEFAULT_DIMENSION, k::Int64=DEFAULT_NUM_CLUSTERS,
                      u_dist::Float64=DEFAULT_DISTANCE,
                      u_radius::Float64=DEFAULT_RADIUS,
                      l_dist::Float64=DEFAULT_L_DISTANCE)
  # filename - filename of output data file
  # rec - number of records to be generated
  # d - number of dimensions (properties of each record)
  # k - number of pre-determined cluster centres
  # u_dist - upper bound of distance between cluster centre and origin
  # u_radius - upper bound of radius of cluster, the definition of
  #            radius is Not accurate in this implementation
  # l_dist - lower bound of distance between cluster centre and origin,
  #          default value is 0.0
  writedlm(filename, genTestCase(rec, d, k, u_dist, u_radius, l_dist))
end

function genNormalTestCase(rec::Int64=DEFAULT_NUMBER, d::Int64=DEFAULT_DIMENSION,
                      k::Int64=DEFAULT_NUM_CLUSTERS,
                      u_dist::Float64=DEFAULT_DISTANCE,
                      u_radius::Float64=DEFAULT_RADIUS,
                      l_dist::Float64=DEFAULT_L_DISTANCE)
  # generate test case for clustering
  # rec - number of records to be generated
  # d - number of dimensions (properties of each record)
  # k - number of pre-determined cluster centres
  # u_dist - upper bound of distance between cluster centre and origin
  # u_radius - upper bound of radius of cluster, the definition of
  #            radius is Not accurate in this implementation
  # l_dist - lower bound of distance between cluster centre and origin,
  #          default value is 0.0

  mat = Array(Float64,rec,d)
  # generate cluster centres
  centre = rand(l_dist:u_dist, k, d)
  for i in 1:rec
    # randomly select a centroid
    ctrrow = rand(1:k)
    for j in range(1,d)
      mat[i,j] = centre[ctrrow,j] + u_radius * randn()
    end
  end
  return mat
end

function genNormalTestFile(filename::AbstractString, rec::Int64=DEFAULT_NUMBER,
                      d::Int64=DEFAULT_DIMENSION, k::Int64=DEFAULT_NUM_CLUSTERS,
                      u_dist::Float64=DEFAULT_DISTANCE,
                      u_radius::Float64=DEFAULT_RADIUS,
                      l_dist::Float64=DEFAULT_L_DISTANCE)
  # filename - filename of output data file
  # rec - number of records to be generated
  # d - number of dimensions (properties of each record)
  # k - number of pre-determined cluster centres
  # u_dist - upper bound of distance between cluster centre and origin
  # u_radius - upper bound of radius of cluster, the definition of
  #            radius is Not accurate in this implementation
  # l_dist - lower bound of distance between cluster centre and origin,
  #          default value is 0.0
  writedlm(filename, genNormalTestCase(rec, d, k, u_dist, u_radius, l_dist))
end
#=
recs = [100, 100, 500, 500, 1000, 2000, 2000, 2000, 5000, 5000, 10000, 25000, 100000]
ds = [2, 10, 10, 15, 10, 20, 25, 40, 40, 50, 50, 100, 100]

for i=1:13
  genNormalTestFile("testcase$i.txt", recs[i], ds[i], 5, 1000.0, 150.0)
end
=#
