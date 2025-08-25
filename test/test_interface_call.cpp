//
// Created by holynova on 25-4-24.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <omp.h>
#include <cmath>

#include "sdk/bi_sdk_api.h"

namespace test_interface_call
{
    template <typename T>
    void find_and_print_top_n(std::vector< std::vector<T> > &vec, const int n) {
        // std::cout << std::fixed << std::setprecision(2);
        for (size_t i = 0; i < vec.size(); ++i) {
            std::vector<int> tmp_vec(vec[i].size());
            std::iota(tmp_vec.begin(), tmp_vec.end(), 0);
            std::sort(tmp_vec.begin(), tmp_vec.end(), [&](int a, int b) {
                return vec[i][a] > vec[i][b];
            });

            std::cout << "item " << i << ": " << std::endl;
            for (int k = 0; k < n; ++k) {
                std::cout << "(" << tmp_vec[k] << ", " << vec[i][tmp_vec[k]] << ")";
                if (k != n - 1) {
                    std::cout << " | ";
                }
            }
            std::cout << std::endl;
        }
    }

    void find_and_print_top_n_softmax_log(std::vector< std::vector<float> >& vec, const int n) {
        for (size_t i = 0; i < vec.size(); ++i) {
            std::vector<float> softmax_log_vec;
            softmax_log_vec.reserve(vec[i].size());
            double sum_exp = 0;
            for (float f_num : vec[i]) {
                sum_exp += exp(f_num);
            }
            double softmax_scale = 1.0f / sum_exp;
            for (float f_num : vec[i]) {
                auto after_softmax = static_cast<float>(exp(f_num) * softmax_scale);
                softmax_log_vec.emplace_back(-1 * logf(after_softmax));
            }

            std::vector<int> tmp_vec(vec[i].size());
            std::iota(tmp_vec.begin(), tmp_vec.end(), 0);
            std::sort(tmp_vec.begin(), tmp_vec.end(), [&](const int a, const int b) {
                return softmax_log_vec[a] < softmax_log_vec[b];
            });

            std::cout << "item " << i << ": " << std::endl;
            for (int k = 0; k < n; ++k) {
                std::cout << "(" << tmp_vec[k] << ", " << softmax_log_vec[tmp_vec[k]] << ")";
                if (k != n - 1) {
                    std::cout << " | ";
                }
            }
            std::cout << std::endl;
        }
    }

} // namespace test_interface_call

TEST(InterfaceCall, Gpt2Model)
{
    bool ret = true;
    std::string gpt2_res_path = "./gpt2_res_all_float/gpt2_final.bin";

    std::ifstream fin(gpt2_res_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cout << "Failed to open file: " << gpt2_res_path << std::endl;
        ret = false;
    }
    ASSERT_TRUE(ret);

    fin.seekg(0, std::ios::end);
    size_t data_size = static_cast<size_t>(fin.tellg());
    fin.seekg(0, std::ios::beg);
    auto data_in = new char[data_size + 10];
    fin.read(data_in, data_size);
    fin.close();

    BIModelInterfaceBase *model_interface = CreateBIModelInterface(BIModelTypes::BIGpt2);

    model_interface->set_threads_num(1);

    model_interface->bi_init(data_in, data_size);
    delete[] data_in;

    std::vector< std::vector<unsigned int> > input_vec = { { 0, 1, 2 } };

    model_interface->bi_set_input(input_vec);

    std::vector< std::vector<float> > output_vec = {};

    model_interface->bi_run(output_vec);

    // test_interface_call::find_and_print_top_n(output_vec, 20);
    // test_interface_call::find_and_print_top_n_softmax_log(output_vec, 20);

    std::vector< std::vector<unsigned int> > input_vec2 = { { 0 }, { 0, 1 }, { 0, 1, 2 } };

    model_interface->bi_set_input(input_vec2);

    model_interface->bi_run(output_vec);

    test_interface_call::find_and_print_top_n(output_vec, 20);
    test_interface_call::find_and_print_top_n_softmax_log(output_vec, 20);

    DeleteBIModelInterface(model_interface);

}

TEST(InterfaceCall, Gpt2ModelSixLayer) {
    bool ret = true;
    std::string gpt2_res_path = "./six_layer/gpt2_final.bin";

    std::ifstream fin(gpt2_res_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cout << "Failed to open file: " << gpt2_res_path << std::endl;
        ret = false;
    }
    ASSERT_TRUE(ret);

    fin.seekg(0, std::ios::end);
    size_t data_size = static_cast<size_t>(fin.tellg());
    fin.seekg(0, std::ios::beg);
    auto data_in = new char[data_size + 10];
    fin.read(data_in, data_size);
    fin.close();

    BIModelInterfaceBase *model_interface = CreateBIModelInterface(BIModelTypes::BIGpt2);

    model_interface->set_threads_num(1);

    model_interface->bi_init(data_in, data_size);
    delete[] data_in;

    std::vector< std::vector<unsigned int> > input_vec = { { 5, 7, 9, 30, 50 } };

    model_interface->bi_set_input(input_vec);
    std::vector< std::vector<float> > output_vec = {};
    model_interface->bi_run(output_vec);

    test_interface_call::find_and_print_top_n(output_vec, 5);
    test_interface_call::find_and_print_top_n_softmax_log(output_vec, 5);



    model_interface->bi_set_input(input_vec);
    output_vec.clear();
    model_interface->bi_run(output_vec);

    test_interface_call::find_and_print_top_n(output_vec, 5);
    test_interface_call::find_and_print_top_n_softmax_log(output_vec, 5);



    input_vec.clear();
    output_vec.clear();
    std::vector<unsigned int> tmp_data = { 90, 50, 60, 80};
    input_vec.emplace_back(std::move(tmp_data));
    model_interface->bi_set_input(input_vec);
    model_interface->bi_run(output_vec);

    test_interface_call::find_and_print_top_n(output_vec, 5);
    test_interface_call::find_and_print_top_n_softmax_log(output_vec, 5);


}

TEST(InterfaceCall, OMPTest) {
#pragma omp parallel default(none) shared(std::cout)
    {
        // 获取当前并行区域的线程数
        int num_threads = omp_get_num_threads();

        // 获取当前线程的 ID
        int thread_id = omp_get_thread_num();

        // 只在第一个线程打印
        if (thread_id == 0) {
            std::cout << "Number of threads: " << num_threads << std::endl;
        }
    }
}



