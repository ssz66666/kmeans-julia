#include("kmeans_scheduler_shared.jl")
#include("kmeans_scheduler_dist.jl")
include("./../src/serial_kmeans.jl")
#import KMeans_Parallel_Shared
#import KMeans_Parallel_Dist.kmeans
import Serial_Kmeans.kmeans
#KMeans_Parallel_Shared.load_libraries()
#KMeans_Parallel_Dist.load_libraries()
