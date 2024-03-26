cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)



make_benchmark (srad-benchmark-module srad-bench "srad.cu;")

make_benchmark_choice (srad-benchmark srad-benchmark-smartio srad-benchmark-module)
