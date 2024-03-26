cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)



make_benchmark (multivectoradd-benchmark-module multivectoradd-bench "main.cu;")

make_benchmark_choice (multivectoradd-benchmark multivectoradd-benchmark-smartio multivectoradd-benchmark-module)
