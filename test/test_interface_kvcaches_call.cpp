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

namespace test_interface_kvcaches_call
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

} // namespace test_interface_kvcaches_call

TEST(InterfaceKvcachesCall, Gpt2Model)
{
    bool ret = true;
    std::string gpt2_res_path = "./fix-fake/gpt2_final.bin";

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

    // // 先进行推理引擎初始化，同时进行一次 </s> 推理，获取其对应的 kv_cache_id，并将推理结果保留下来
    // std::vector< std::vector<float> > output_vec = {};
    // unsigned int kv_cache_id;
    // model_interface->bi_init(data_in, data_size, output_vec, kv_cache_id);
    // delete[] data_in;
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    // // 模拟进行第二次推理，需要获取到上一步推理(</s>)时生成的 kv_cache_id作为参数传递
    // std::vector< std::vector<unsigned int> > input_vec = { { 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},{ 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},
    //     { 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},{ 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},
    //     { 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},{ 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},
    //     { 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},{ 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},
    //     { 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},{ 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3}};
    // // 根据 input数据的 batch_size来进行 kv_cache_id_map构建
    // std::vector< std::vector<unsigned int> > kv_cache_id_map{};
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_cache_id, 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // std::vector<size_t> avail_lens;
    // std::vector<unsigned int> kv_block_ids;
    // output_vec.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // std::vector<unsigned int> decode_ids;
    // decode_ids = {2046, 2045, 2044, 2043, 2042};
    // bool vaild_decode_ids = model_interface->bi_valid_decode_ids(decode_ids);
    // decode_ids.clear();
    // decode_ids = {2047, 2046, 2045, 2044, 2043};
    // vaild_decode_ids = model_interface->bi_valid_decode_ids(decode_ids);
    // decode_ids.clear();
    // decode_ids = {1999, 1998, 1997, 1996, 1995};
    // vaild_decode_ids = model_interface->bi_valid_decode_ids(decode_ids);
    // decode_ids.clear();
    // decode_ids = {1, 2, 3, 4, 5};
    // vaild_decode_ids = model_interface->bi_valid_decode_ids(decode_ids);
    //
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    // // 模拟进行第三次推理，需要获取上一步推理 tokenid为 {0,3} 时生成的 kv_cache_id 作为参数进行传递
    // input_vec = { {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33},
    //     {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33},
    //     {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33},
    //     {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33},
    //     {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}};
    // kv_cache_id_map.clear();
    // for (int i = 0; i < kv_block_ids.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_block_ids[i], 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    //
    // // 模拟进行第四次推理，需要获取上一步推理 tokenid为 {0, 3, 33} 时生成的 kv_cache_id 作为参数进行传递
    // input_vec = {{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},
    //     {0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},
    //     {0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},
    //     {0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},
    //     {0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4},{0, 3, 33, 4}};
    // kv_cache_id_map.clear();
    // for (int i = 0; i < kv_block_ids.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_block_ids[i], 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    //
    // // 模拟进行第五次推理，需要获取上一步推理 tokenid为 {0, 3, 33, 4} 时生成的 kv_cache_id 作为参数进行传递
    // input_vec = {{0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59},
    //     {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59},
    //     {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59},
    //     {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59},
    //     {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59},
    //     {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59},
    //     {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59},
    //     {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59},
    //     {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59},
    //     {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}, {0, 3, 33, 4, 59}};
    // kv_cache_id_map.clear();
    // for (int i = 0; i < kv_block_ids.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_block_ids[i], 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    //
    // // 模拟进行第六次推理，需要获取上一步推理 tokenid为 {0, 3, 33, 4, 59} 时生成的 kv_cache_id 作为参数进行传递
    // input_vec = {{0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58},
    // {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58},
    // {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58},
    // {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58},
    // {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58},
    // {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58},
    // {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58},
    // {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58},
    // {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58},
    // {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}, {0, 3, 33, 4, 59, 58}};
    // kv_cache_id_map.clear();
    // for (int i = 0; i < kv_block_ids.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_block_ids[i], 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    //
    // // 模拟进行第七次推理，需要获取上一步推理 tokenid为 {0, 3, 33, 4, 59， 35} 时生成的 kv_cache_id 作为参数进行传递
    // input_vec = {{0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35},
    // {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35},
    // {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35},
    // {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35},
    // {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35},
    // {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35},
    // {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35},
    // {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35},
    // {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35},
    // {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}, {0, 3, 33, 4, 59, 58, 35}};
    // kv_cache_id_map.clear();
    // for (int i = 0; i < kv_block_ids.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_block_ids[i], 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    //
    // // 模拟进行第八次推理，需要获取上一步推理 tokenid为 {0, 3, 33, 4, 59， 35} 时生成的 kv_cache_id 作为参数进行传递
    // input_vec = {{0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}, {0, 3, 33, 4, 59, 58, 35, 11}};
    // kv_cache_id_map.clear();
    // for (int i = 0; i < kv_block_ids.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_block_ids[i], 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    //
    // // 模拟进行第九次推理，需要获取上一步推理 tokenid为 {0, 3, 33, 4, 59， 35, 11} 时生成的 kv_cache_id 作为参数进行传递
    // input_vec = {{0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11}};
    // kv_cache_id_map.clear();
    // for (int i = 0; i < kv_block_ids.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_block_ids[i], 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    //
    // // 模拟进行第十次推理，需要获取上一步推理 tokenid 为{0, 3, 33, 4, 59， 35, 11， 11} 时生成的 kv_cache_id 作为参数进行传递
    // input_vec = {{0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11}};
    // kv_cache_id_map.clear();
    // for (int i = 0; i < kv_block_ids.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_block_ids[i], 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    //
    // // 模拟进行第十一次推理，需要获取上一步推理 tokenid 为{0, 3, 33, 4, 59， 35, 11， 11， 11} 时生成的 kv_cache_id 作为参数进行传递
    // input_vec = {{0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11}};
    // kv_cache_id_map.clear();
    // for (int i = 0; i < kv_block_ids.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_block_ids[i], 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    // // 统计当前有多少 kvblock可用，进行部分清理并进行下一步推理测试
    // unsigned int  avaliable_kvblock_count = 0;
    // model_interface->bi_get_avaliable_kvblock_count(avaliable_kvblock_count);
    // std::vector<unsigned int> remove_kvblockid{11, 12, 13};
    // auto sub_ret = model_interface->bi_release_kvcache_block(remove_kvblockid);
    // model_interface->bi_get_avaliable_kvblock_count(avaliable_kvblock_count);
    //
    //
    // // 模拟进行第十二次推理，需要获取上一步推理 tokenid 为{0, 3, 33, 4, 59， 35, 11， 11， 11， 11} 时生成的 kv_cache_id 作为参数进行传递
    // input_vec = {{0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}, {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11},
    // {0, 3, 33, 4, 59, 58, 35, 11, 11, 11, 11, 11}};
    // kv_cache_id_map.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_block_ids[i], 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    //
    //
    //
    //
    // // 将以上第二次、第三次推理数据进行清空，重新进行与上面输入相同的第二次、第三次推理
    // model_interface->bi_reset(kv_cache_id);
    // input_vec = { { 0, 3}, {0, 3}, {0, 3} };
    // kv_cache_id_map.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_cache_id, 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    // input_vec = { {0, 3, 33}, {0, 3, 33}, {0, 3, 33} };
    // kv_cache_id_map.clear();
    // for (int i = 0; i < kv_block_ids.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_block_ids[i], 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);

    DeleteBIModelInterface(model_interface);

}

TEST(InterfaceKvcachesCall, AWQGpt2Model) {
bool ret = true;
    std::string gpt2_res_path = "./awq/gpt2_final.bin";

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

    // // 先进行推理引擎初始化，同时进行一次 </s> 推理，获取其对应的 kv_cache_id，并将推理结果保留下来
    // std::vector< std::vector<float> > output_vec = {};
    // unsigned int kv_cache_id;
    // model_interface->bi_init(data_in, data_size, output_vec, kv_cache_id);
    // delete[] data_in;
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    // // 模拟进行第二次推理，需要获取到上一步推理(</s>)时生成的 kv_cache_id作为参数传递
    // std::vector< std::vector<unsigned int> > input_vec = { { 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},{ 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},
    //     { 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},{ 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},
    //     { 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},{ 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},
    //     { 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},{ 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},
    //     { 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3},{ 0, 3}, {0, 3}, {0, 3},{ 0, 3}, {0, 3}};
    // // 根据 input数据的 batch_size来进行 kv_cache_id_map构建
    // std::vector< std::vector<unsigned int> > kv_cache_id_map{};
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_cache_id, 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // std::vector<size_t> avail_lens;
    // std::vector<unsigned int> kv_block_ids;
    // output_vec.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // std::vector<unsigned int> decode_ids;
    // decode_ids = {2046, 2045, 2044, 2043, 2042};
    // bool vaild_decode_ids = model_interface->bi_valid_decode_ids(decode_ids);
    // decode_ids.clear();
    // decode_ids = {2047, 2046, 2045, 2044, 2043};
    // vaild_decode_ids = model_interface->bi_valid_decode_ids(decode_ids);
    // decode_ids.clear();
    // decode_ids = {1999, 1998, 1997, 1996, 1995};
    // vaild_decode_ids = model_interface->bi_valid_decode_ids(decode_ids);
    // decode_ids.clear();
    // decode_ids = {1, 2, 3, 4, 5};
    // vaild_decode_ids = model_interface->bi_valid_decode_ids(decode_ids);
    //
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    // // 模拟进行第三次推理，需要获取上一步推理 tokenid为 {0,3} 时生成的 kv_cache_id 作为参数进行传递
    // input_vec = { {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33},
    //     {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33},
    //     {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33},
    //     {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33},
    //     {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}, {0, 3, 33}};
    // kv_cache_id_map.clear();
    // for (int i = 0; i < kv_block_ids.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_block_ids[i], 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    //
    // // 将以上第二次、第三次推理数据进行清空，重新进行与上面输入相同的第二次、第三次推理
    // model_interface->bi_reset(kv_cache_id);
    // input_vec = { { 0, 3}, {0, 3}, {0, 3} };
    // kv_cache_id_map.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_cache_id, 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);
    //
    // input_vec = { {0, 3, 33}, {0, 3, 33}, {0, 3, 33} };
    // kv_cache_id_map.clear();
    // for (int i = 0; i < kv_block_ids.size(); ++i) {
    //     kv_cache_id_map.push_back({kv_block_ids[i], 1});
    // }
    // model_interface->bi_set_input(input_vec, kv_cache_id_map);
    //
    // output_vec.clear();
    // kv_block_ids.clear();
    // avail_lens.clear();
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     avail_lens.emplace_back(input_vec[i].size());
    // }
    // model_interface->bi_run(avail_lens, output_vec, kv_block_ids, false);
    //
    // test_interface_kvcaches_call::find_and_print_top_n(output_vec, 20);
    // test_interface_kvcaches_call::find_and_print_top_n_softmax_log(output_vec, 20);

    DeleteBIModelInterface(model_interface);

}