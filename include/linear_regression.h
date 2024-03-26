#ifndef __LINEAR_REGRESSION_H__
#define __LINEAR_REGRESSION_H__

#include <iostream>
#include <functional>
#include <numeric>
#include <chrono>
#include <omp.h>
#include <vector>

//#include <python3.8/Python.h>
#define NO_REUSE ULONG_MAX

class LinearRegression {
public:
    // virt_dist_sum, reuse_dist_sum, reuse_dist_square_sum, virt_reuse_dist_product_sum
    std::vector<uint64_t> prev_results;

    LinearRegression() {
        prev_results.resize(5, 0);
    }

    /*
      data: even index, virt_timestamp_dist
            odd index, actual reuse dist
    */
    void get_ls_solution (uint64_t* data, uint32_t num_samples, float& offset, float& slope) 
    {
        uint64_t virt_dist_sum = prev_results[0];
        uint64_t reuse_dist_sum = prev_results[1];
        uint64_t reuse_dist_square_sum = prev_results[2];
        uint64_t virt_reuse_dist_product_sum = prev_results[3];
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for reduction(+:virt_dist_sum, reuse_dist_sum, reuse_dist_square_sum, virt_reuse_dist_product_sum)
        for (uint32_t i = 0; i < num_samples; i++) {
            if (data[i*2+1] != NO_REUSE) {
                //printf("%u: virt %lu - reuse %lu\n", i, data[i*2], data[i*2+1]);
                virt_dist_sum += data[i*2];
                reuse_dist_sum += data[i*2+1];
                reuse_dist_square_sum += (data[i*2+1]*data[i*2+1]);
                virt_reuse_dist_product_sum += (data[i*2]*data[i*2+1]);
            }
        }
        
        //prev_results[0] = virt_dist_sum;
        //prev_results[1] = reuse_dist_sum;
        //prev_results[2] = reuse_dist_square_sum;
        //prev_results[3] = virt_reuse_dist_product_sum;
        //prev_results[4] += num_samples;       

        // Update the total number of samples
        //num_samples = prev_results[4];
        if ((num_samples*reuse_dist_sum - reuse_dist_sum*reuse_dist_sum) == 0) {
            slope = 1;
            offset = virt_dist_sum - reuse_dist_sum;
            return;
        }
        
        std::cout << "num_samples: " << num_samples << "\t";
        std::cout << "virt_reuse_dist_product_sum: " <<  virt_reuse_dist_product_sum << "\t";
        std::cout << "reuse_dist_sum: " <<  reuse_dist_sum << "\t";
        std::cout << "virt_dist_sum: " <<  virt_dist_sum << "\t";
        std::cout << "reuse_dist_sqaure_sum: " <<  reuse_dist_square_sum << "\t\n";
        std::cout << "denom: " << ((double)num_samples*(double)virt_reuse_dist_product_sum - (double)reuse_dist_sum*(double)virt_dist_sum) << std::endl;
        std::cout << "nom: " << ((double)num_samples*(double)reuse_dist_square_sum - (double)reuse_dist_sum*(double)reuse_dist_sum) << std::endl;

        slope = (((double)num_samples*(double)virt_reuse_dist_product_sum - (double)reuse_dist_sum*(double)virt_dist_sum) 
                / ((double)num_samples*(double)reuse_dist_square_sum - (double)reuse_dist_sum*(double)reuse_dist_sum));
        offset = ((double)virt_dist_sum - slope*(double)reuse_dist_sum) / (double)num_samples;
        
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1);


        if (slope < 0.0) {
            slope = 1.0;
            offset = 0.0;
        }
        //slope = (slope < 0) ? -slope : slope;
        //std::cout << "Time taken for OLS " << std::dec << time_span.count() << " ns." << std::endl;
    }

    //#include <python3.7m/Python.h>
    /*
    #include <boost/math/statistics/linear_regression.hpp>

    using boost::math::statistics::simple_ordinary_least_squares_with_R_squared;
    void get_ls_by_boost(uint64_t* data_x, uint64_t* data_y, uint32_t num_samples, float& offset, float& slope) 
    {
        std::vector<double> x(data_x, data_x+num_samples);        
        std::vector<double> y(data_y, data_y+num_samples);        
        auto [c0, c1] = simple_ordinary_least_squares_with_R_squared(x, y);
        
        offset = (float)c0;
        slope = (float)c1;
    }
    */
};

#endif
