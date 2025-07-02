//
// Created by holynova on 25-4-23.
//

#include <thread>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/neon/bi_ne_functions.h>
#include <runtime/bi_tensor.hpp>

#include "runtime/bi_scheduler.hpp"
#include "utils/utils.hpp"
#include "nlohmann/json.hpp"

#include "model_interface/gpt2_model.h"

namespace res_pack {

    bool load_res_paths(const std::string &path_prefix, const std::string &path_storage_file,
        std::map<GPT2ResOrder, std::string> &res_paths) {

        int res_order_int;
        constexpr int res_order_count = static_cast<int>(GPT2ResOrder::all_res_count);
        std::string res_path_str;

        std::ifstream paths_storage(path_prefix + path_storage_file);
        if (!paths_storage.is_open()) {
            std::cout << "Cannot open file: " << path_prefix << path_storage_file << "!" << std::endl;
            return false;
        }

        res_paths.clear();

        while (paths_storage >> res_order_int >> res_path_str) {
            if (res_order_int >= res_order_count) {
                std::cout << "Res order out of range! " << res_order_int << std::endl;
                return false;
            }
            res_paths[static_cast<GPT2ResOrder>(res_order_int)] = res_path_str;
        }

        paths_storage.close();
        return true;
    }

    /*bool read_and_write_scales(int res_order, const std::string &path_prefix, const std::string &res_path,
        std::fstream &dst_file) {
        std::ifstream in_file(path_prefix + res_path);
        if (!in_file.is_open()) {
            std::cout << "Cannot open file: " << path_prefix << res_path << "!" << std::endl;
            return false;
        }

        std::vector<float> scales;
        float one_scale;
        while (in_file >> one_scale) {
            scales.push_back(one_scale);
        }
        in_file.close();

        GPT2ResHeader res_header;
        res_header.res_order = res_order;
        res_header.data_length = static_cast<int>(scales.size() * sizeof(float));
        res_header.shape[0] = static_cast<int>(scales.size());
        for (int i = 1; i < tensor_max_dim; ++i) {
            res_header.shape[i] = 1;
        }

        // 写入头信息
        dst_file.write(reinterpret_cast<char*>(&res_header), sizeof(res_header));
        // 写入具体数据
        dst_file.write(reinterpret_cast<char*>(scales.data()), sizeof(float) * scales.size());

        return true;
    }*/

    // 本函数直接将 F32（未量化版本层） 转为 F16，若当前为量化版本，其中两 weight层为 int8类型，两 bias层为 int32类型，需要进行特殊处理
    bool read_and_write_npy(int res_order, const std::string &path_prefix, const std::string &res_path,
        std::fstream &dst_file) {
        std::ifstream in_file(path_prefix + res_path, std::ios::in | std::ios::binary);
        if (!in_file.is_open()) {
            std::cout << "Cannot open file: " << path_prefix << res_path << "!" << std::endl;
            return false;
        }
        in_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        // 读取 numpy 文件头信息
        npy::header_t header = BatmanInfer::utils::parse_npy_header(in_file);

        // 模型层的 shape 不能大于 6 维
        if (header.shape.size() > tensor_max_dim) {
            std::cout << "Wrong shape!" << std::endl;
            return false;
        }

        size_t element_count = 1, element_size = header.dtype.itemsize;
        for (auto i : header.shape) {
            element_count *= i;
        }

        // 验证文件完整性
        const size_t current_position = in_file.tellg();
        in_file.seekg(0, std::ios_base::end);
        const size_t end_position = in_file.tellg();
        in_file.seekg(current_position, std::ios_base::beg);

        if ((end_position - current_position) != (element_count * element_size)) {
            std::cout << "File size mismatch! " << path_prefix << res_path << std::endl;
            return false;
        }

        // 检查是否需要进行 f32 到 f16 的转换
        const std::string f32_type_str = "<f4";
        const std::string f16_type_str = "<f2";
        bool enable_f32_to_f16_cast = false;
        if (header.dtype.str() == f32_type_str) {
            enable_f32_to_f16_cast = true;
        }

        // 进行 f32 到 f16 的转换的情景
        if (enable_f32_to_f16_cast) {
            auto buffer = new half[element_count + 10];

            for (auto j = 0; j < element_count; ++j) {
                float f32_val;
                in_file.read(reinterpret_cast<char*>(&f32_val), sizeof(f32_val));
                half f16_val = half_float::half_cast<half, std::round_to_nearest>(f32_val);
                buffer[j] = f16_val;
            }

            GPT2ResHeader res_header;
            res_header.res_order = res_order;
            res_header.data_length = static_cast<int>(element_count * sizeof(half));
            for (auto k = 0; k < header.shape.size(); ++k) {
                res_header.shape[k] = static_cast<int>(header.shape[k]);
            }
            for (auto k = header.shape.size(); k < tensor_max_dim; ++k) {
                res_header.shape[k] = 1;
            }
            memcpy(res_header.data_type, f16_type_str.c_str(), f16_type_str.length());

            // 写入头信息
            dst_file.write(reinterpret_cast<char*>(&res_header), sizeof(res_header));
            // 写入具体数据
            dst_file.write(reinterpret_cast<char*>(buffer), sizeof(half) * element_count);

            delete[] buffer;
        }
        // 通常情况下的数据拷贝（包括 weight层量化[int8] 和 bias层量化[int32]）
        else {
            auto buffer = new char[element_count * element_size + 10];

            in_file.read(buffer, element_count * element_size);

            GPT2ResHeader res_header;
            res_header.res_order = res_order;
            res_header.data_length = static_cast<int>(element_count * element_size);
            for (auto k = 0; k < header.shape.size(); ++k) {
                res_header.shape[k] = static_cast<int>(header.shape[k]);
            }
            for (auto k = header.shape.size(); k < tensor_max_dim; ++k) {
                res_header.shape[k] = 1;
            }
            memcpy(res_header.data_type, header.dtype.str().c_str(), header.dtype.str().length());

            // 写入头信息
            dst_file.write(reinterpret_cast<char*>(&res_header), sizeof(res_header));
            // 写入具体数据
            dst_file.write(buffer, element_count * element_size);

            delete[] buffer;
        }

        return true;
    }

#ifdef FIX_VER
    bool read_and_write_json(int res_order, const std::string &path_prefix, const std::string &res_path,
        std::fstream &dst_file) {

        std::ifstream json_fin(path_prefix + res_path);
        if (!json_fin.is_open()) {
            std::cout << "Cannot open file: " << path_prefix << res_path << "!" << std::endl;
            return false;
        }

        nlohmann::json json_data;
        json_fin >> json_data;

        HyperParameters params;

        params.attn_input_scale = json_data["attn_input_scale"];
        params.attn_input_zp = json_data["attn_input_zp"];

        params.attn_output_scale = json_data["attn_output_scale"];
        params.attn_output_zp = json_data["attn_output_zp"];

        params.q_output_scale = json_data["q_output_scale"];
        params.q_output_zp = json_data["q_output_zp"];

        params.k_output_scale = json_data["k_output_scale"];
        params.k_output_zp = json_data["k_output_zp"];

        params.v_output_scale = json_data["v_output_scale"];
        params.v_output_zp = json_data["v_output_zp"];

        params.out_input_scale = json_data["out_input_scale"];
        params.out_input_zp = json_data["out_input_zp"];

        params.fc1_input_scale = json_data["fc1_input_scale"];
        params.fc1_input_zp = json_data["fc1_input_zp"];

        params.fc1_output_scale = json_data["fc1_output_scale"];
        params.fc1_output_zp = json_data["fc1_output_zp"];

        params.fc2_input_scale = json_data["fc2_input_scale"];
        params.fc2_input_zp = json_data["fc2_input_zp"];

        // write data
        GPT2ResHeader res_header;
        res_header.res_order = res_order;
        res_header.data_length = sizeof(HyperParameters);
        memset(&res_header.data_type, 0, 8);
        memset(&res_header.shape, 0, 6 * sizeof(int));

        // 写入头信息
        dst_file.write(reinterpret_cast<char*>(&res_header), sizeof(res_header));
        // 写入参数信息
        dst_file.write(reinterpret_cast<char*>(&params), sizeof(params));

        return true;
    }
#endif
} // namespace res_pack

TEST(ResPack, PackGPT) {
    bool ret = true;


    std::map<GPT2ResOrder, std::string> res_paths;
    std::string res_path_prefix = "../input_res-fix/", path_storage_file = "_all_res_paths.txt";
    ret = res_pack::load_res_paths(res_path_prefix, path_storage_file, res_paths);
    ASSERT_TRUE(ret);

    std::fstream dst_file(res_path_prefix + "gpt2_final.bin", std::fstream::out | std::ios::binary);
    if (!dst_file.is_open()) {
        std::cout << "Cannot open file: " << res_path_prefix << "gpt2_final.bin!" << std::endl;
        ret = false;
    }
    ASSERT_TRUE(ret);


    std::cout << "Start pack all resource files..." << std::endl;


    for (int i = 0; i < static_cast<int>(GPT2ResOrder::all_res_count); ++i) {
        std::cout << "Packing resource " << i << std::endl;

        switch (auto cur_order = static_cast<GPT2ResOrder>(i)) {
            case GPT2ResOrder::gather_weight:
            case GPT2ResOrder::add_weight:
            case GPT2ResOrder::attn_gamma_weight:
#ifdef FLOAT_VER
            case GPT2ResOrder::attn_qkv_weight:
#elifdef FIX_VER
            case GPT2ResOrder::attn_qkv_weight_scale:
#endif
            case GPT2ResOrder::attn_qkv_bias:
            case GPT2ResOrder::attn_c_proj_weight:
            case GPT2ResOrder::attn_c_proj_bias:
            case GPT2ResOrder::mlp_gamma_weight:
#ifdef FLOAT_VER
            case GPT2ResOrder::c_fc_weight:
#elifdef FIX_VER
            case GPT2ResOrder::c_fc_weight_scale:
#endif
            case GPT2ResOrder::c_fc_bias:
            case GPT2ResOrder::c_proj_weight:
            case GPT2ResOrder::c_proj_bias:
            case GPT2ResOrder::rms_gamma_weight:
            case GPT2ResOrder::lm_head_weight: {
                ret = res_pack::read_and_write_npy(static_cast<int>(cur_order), res_path_prefix,
                    res_paths[cur_order], dst_file);
                break;
            }
 /*
            case GPT2ResOrder::attn_qkv_weight_scale:
            case GPT2ResOrder::c_fc_weight_scale: {
                ret = res_pack::read_and_write_scales(static_cast<int>(cur_order), res_path_prefix,
                    res_paths[cur_order], dst_file);
                break;
            }*/
#ifdef FIX_VER
            case GPT2ResOrder::decode_layer_scales: {
                ret = res_pack::read_and_write_json(static_cast<int>(cur_order), res_path_prefix,
                    res_paths[cur_order], dst_file);
                break;
            }
#endif
            default: ;
        }

        ASSERT_TRUE(ret);
    }


    dst_file.close();
    std::cout << "ResPack test passed!" << std::endl;

}




