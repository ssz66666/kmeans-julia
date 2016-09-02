#include("benchmarktypes.jl")

global data
global k

function get_test_vector(t::AbstractClusteringTestCase)
  if typeof(t) == TestCase return [t]
  else return t.tcases end
end

function clustering_benchmark{N<:Integer}(np::N,
                                          data_init_script::AbstractString,
                                          all_data_init_script::AbstractString,
                                          name_of_function::AbstractString,
                                          type_of_alg::ClusteringAlg,
                                          test_case::AbstractClusteringTestCase)
  global data, k
  tests::Vector{TestCase} = get_test_vector(test_case)

  for test in tests
    filename = test.data_filename
    println("running testcase:$filename")
    data = readdlm(filename)
    k = test.estimated_k
    if data_init_script != "" include(data_init_script) end
    if all_data_init_script != "" include_all(all_data_init_script) end
    for l = 1:5
      @time result = eval(Expr(:call,parse(name_of_function),
      map(eval,get(test.args,type_of_alg,()))...))
      finalize(data);finalize(result)
    end
  end
end

function build_tests{T<:AbstractString,N<:Integer}(files::Vector{T},ks::Vector{N},config)

  if !(length(files)==length(ks))  error("invalid args") end
  tests = Array(TestCase, (0...))
  for i in eachindex(files, ks)
    newtest = TestCase(files[i],ks[i],config)
    push!(tests, newtest)
  end
  return TestDataSet(length(tests),tests)
end

function include_all{N<:Integer}(file::AbstractString, procs::Vector{N}=procs())
  if file != ""
    f = """include("$file")"""
    @sync for w in procs
      expr = Expr(:call, :eval, :Main, Expr(:call, :remotecall_fetch, :eval,
                                        w, Expr(:call, :parse, f)))
      @async eval(Main, expr)
    end
  end
end

function load_init()
  if init_script != "" include(init_script) end
  if init_all != "" include_all(init_all) end
end

type MyClusteringAlg<:KMeansClusteringAlg end

np = 1
if nprocs() < np addprocs(np) end
init_script="init.jl"
init_all="init_all.jl"
init_data="init_data.jl"
init_all_data="init_all_data.jl"
func_name = "kmeans"

tests = ["iris.txt",
         "testcase1.txt",
         "testcase2.txt",
         "testcase3.txt",
         "testcase4.txt",
         "testcase5.txt",
         "testcase6.txt",
         "testcase7.txt",
         "testcase8.txt",
         "testcase9.txt",
         "testcase10.txt",
         "testcase11.txt",
         "testcase12.txt"]

ks = [3,5,5,5,5,5,5,5,5,5,5,5,5]

args = (:data,:k,5000,false)

config = Dict((MyClusteringAlg()=>args))

load_init()
testpackages = build_tests(tests,ks,config)
run() = clustering_benchmark(np, init_data, init_all_data, func_name, MyClusteringAlg(), testpackages)
