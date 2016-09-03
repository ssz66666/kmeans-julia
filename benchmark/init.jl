#include("kmeans_scheduler_shared.jl")
include("./../src/distributed_kmeans.jl")
#include("./../src/serial_kmeans.jl")
#include("./../src/threaded_kmeans.jl")
#import KMeans_Parallel_Shared
import Distributed_Kmeans.kmeans
#import Serial_Kmeans.kmeans
#import Threaded_Kmeans.kmeans
#KMeans_Parallel_Shared.load_libraries()
Distributed_Kmeans.load_libraries()
