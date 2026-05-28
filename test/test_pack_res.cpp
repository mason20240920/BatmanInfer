//
// Created by holynova on 25-4-23.
//

#include <cstdint>
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

    // 需要将 txt 明文数据进行读取并存储
    bool read_and_write_scales(int res_order, const std::string &path_prefix, const std::string &res_path,
        std::fstream &dst_file) {
        std::ifstream scales_file(path_prefix + res_path);
        float value;
        std::vector<float> all_scales;
        while (scales_file >> value) {
            all_scales.push_back(value);
        }

        // 所有 scales都已经读取完成，需要进行数据构建并存储
        size_t element_count = all_scales.size();
        auto buffer = new float[element_count + 10];

        for (auto j = 0; j < element_count; j++) {
            buffer[j] = all_scales[j];
        }

        GPT2ResHeader res_header;
        res_header.res_order = res_order;
        res_header.data_length = static_cast<int>(element_count * sizeof(float));
        res_header.shape[0] = element_count;
        for (auto j = 1; j < 6; j++) {
            res_header.shape[j] = 1;
        }
        const std::string f32_type_str = "<f4";
        memcpy(res_header.data_type, f32_type_str.c_str(), f32_type_str.length());

        // 写入头信息
        dst_file.write(reinterpret_cast<char*>(&res_header), sizeof(res_header));
        // 写入具体数据
        dst_file.write(reinterpret_cast<char*>(buffer), sizeof(float) * element_count);

        delete[] buffer;

        return true;
    }

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

    // 本函数读取 JSON 文件并进行存储
    bool read_and_write_json(int res_order, const std::string &path_prefix, const std::string &res_path,
        std::fstream &dst_file) {
        std::ifstream json_file(path_prefix + res_path);
        if (!json_file.is_open()) {
            std::cout << "Cannot open file: " << path_prefix << res_path << "!" << std::endl;
            return false;
        }

        nlohmann::json json_data;
        json_file >> json_data;
        json_file.close();

        AllLayerHyperParameters all_params;
        memset(&all_params, 0, sizeof(AllLayerHyperParameters));

        auto layers = json_data["layers"];
        all_params.layer_count = static_cast<int>(layers.size());

        for (int i = 0; i < all_params.layer_count && i < 6; ++i) {
            auto &layer = layers[i];
            all_params.layers[i].attn_input_scale = layer.value("attn_input_scale", 0.f);
            all_params.layers[i].attn_input_zp = layer.value("attn_input_zp", 0);
            all_params.layers[i].attn_output_scale = layer.value("attn_output_scale", 0.f);
            all_params.layers[i].attn_output_zp = layer.value("attn_output_zp", 0);
            all_params.layers[i].q_output_scale = layer.value("q_output_scale", 0.f);
            all_params.layers[i].q_output_zp = layer.value("q_output_zp", 0);
            all_params.layers[i].k_output_scale = layer.value("k_output_scale", 0.f);
            all_params.layers[i].k_output_zp = layer.value("k_output_zp", 0);
            all_params.layers[i].v_output_scale = layer.value("v_output_scale", 0.f);
            all_params.layers[i].v_output_zp = layer.value("v_output_zp", 0);
            all_params.layers[i].softmax_out_scale = layer.value("softmax_out_scale", 0.f);
            all_params.layers[i].softmax_out_zp = layer.value("softmax_out_zp", 0);
            all_params.layers[i].pv_bmm_out_scale = layer.value("pv_bmm_out_scale", 0.f);
            all_params.layers[i].pv_bmm_out_zp = layer.value("pv_bmm_out_zp", 0);
            all_params.layers[i].fc1_input_scale = layer.value("fc1_input_scale", 0.f);
            all_params.layers[i].fc1_input_zp = layer.value("fc1_input_zp", 0);
            all_params.layers[i].fc1_output_scale = layer.value("fc1_output_scale", 0.f);
            all_params.layers[i].fc1_output_zp = layer.value("fc1_output_zp", 0);
            all_params.layers[i].gelu_output_scale = layer.value("gelu_output_scale", 0.f);
            all_params.layers[i].gelu_output_zp = layer.value("gelu_output_zp", 0);
        }

        // write header + binary struct
        GPT2ResHeader res_header;
        res_header.res_order = res_order;
        res_header.data_length = sizeof(AllLayerHyperParameters);
        memset(&res_header.data_type, 0, 8);
        memset(&res_header.shape, 0, 6 * sizeof(int));

        dst_file.write(reinterpret_cast<char*>(&res_header), sizeof(res_header));
        dst_file.write(reinterpret_cast<char*>(&all_params), sizeof(AllLayerHyperParameters));

        return true;
    }

    // 本函数将 int8(有符号) 转为 int4(有符号) 进行存储使用
    int read_and_write_npy_int8toint4(int res_order, const std::string &path_prefix, const std::string &res_path,
        std::fstream &dst_file) {
        std::ifstream in_file(path_prefix + res_path, std::ios::in | std::ios::binary);
        if (!in_file.is_open()) {
            std::cout << "Cannot open file: " << path_prefix << res_path << "!" << std::endl;
            return false;
        }
        in_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        // 读取 numpy 文件头信息
        npy::header_t header = utils::parse_npy_header(in_file);

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

        // 被打包文件数据个数是偶数，安全起见进行相关判断
        if (element_count % 2 != 0) {
            std::cout << "element_count % 2 != 0" << std::endl;
            return false;
        }

        auto buffer = new uint8_t[element_count];

        for (int j = 0; j < element_count/2; ++j) {
            uint8_t uint8_val1;
            uint8_t uint8_val2;
            in_file.read(reinterpret_cast<char*>(&uint8_val1), sizeof(uint8_val1));
            in_file.read(reinterpret_cast<char*>(&uint8_val2), sizeof(uint8_val2));
            buffer[j] = (((uint8_val1 & 0x0F) << 4) | (uint8_val2 & 0x0F));
        }

        GPT2ResHeader res_header;
        res_header.res_order = res_order;
        res_header.data_length = static_cast<int>(element_count/2 * sizeof(uint8_t));
        for (auto k = 0; k < header.shape.size(); ++k) {
            res_header.shape[k] = static_cast<int>(header.shape[k]);
        }
        for (auto k = header.shape.size(); k < tensor_max_dim; ++k) {
            res_header.shape[k] = 1;
        }

        // npy不支持存储为 int4，这里类型为自定义类型
        const std::string int4_type_str = "int4";
        memcpy(res_header.data_type, int4_type_str.c_str(), int4_type_str.length());

        // 写入头信息
        dst_file.write(reinterpret_cast<char*>(&res_header), sizeof(res_header));
        // 写入具体数据
        dst_file.write(reinterpret_cast<char*>(buffer), sizeof(uint8_t) * element_count/2);

        delete[] buffer;

        return true;
    }
} // namespace res_pack

TEST(ResPack, PackGPT) {
    bool ret = true;


    std::map<GPT2ResOrder, std::string> res_paths;
    std::string res_path_prefix = "./three_layer/", path_storage_file = "_all_res_paths.txt";
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
        // 按顺序进行打包，如果遇到没有的数据直接跳过
        if (res_paths.find(static_cast<GPT2ResOrder>(i)) == res_paths.end()) {
            continue;
        }

        std::cout << "Packing resource " << i << std::endl;

        switch (auto cur_order = static_cast<GPT2ResOrder>(i)) {
            case GPT2ResOrder::transformer_wte_weight:
            case GPT2ResOrder::add_wte_weight:
            case GPT2ResOrder::attn_gamma_weights_0:
            case GPT2ResOrder::c_attn_bias_0:
            case GPT2ResOrder::p_attn_weights_0:
            case GPT2ResOrder::p_attn_bias_0:
            case GPT2ResOrder::mlp_rms_gamma_0:
            case GPT2ResOrder::c_fc_bias_0:
            case GPT2ResOrder::c_proj_weights_0:
            case GPT2ResOrder::c_proj_bias_0:
            case GPT2ResOrder::eos_k_o_0:
            case GPT2ResOrder::eos_q_o_0:
            case GPT2ResOrder::eos_v_o_0:
            case GPT2ResOrder::attn_gamma_weights_1:
            case GPT2ResOrder::c_attn_bias_1:
            case GPT2ResOrder::p_attn_weights_1:
            case GPT2ResOrder::p_attn_bias_1:
            case GPT2ResOrder::mlp_rms_gamma_1:
            case GPT2ResOrder::c_fc_bias_1:
            case GPT2ResOrder::c_proj_weights_1:
            case GPT2ResOrder::c_proj_bias_1:
            case GPT2ResOrder::eos_k_o_1:
            case GPT2ResOrder::eos_q_o_1:
            case GPT2ResOrder::eos_v_o_1:
            case GPT2ResOrder::attn_gamma_weights_2:
            case GPT2ResOrder::c_attn_bias_2:
            case GPT2ResOrder::p_attn_weights_2:
            case GPT2ResOrder::p_attn_bias_2:
            case GPT2ResOrder::mlp_rms_gamma_2:
            case GPT2ResOrder::c_fc_bias_2:
            case GPT2ResOrder::c_proj_weights_2:
            case GPT2ResOrder::c_proj_bias_2:
            case GPT2ResOrder::eos_k_o_2:
            case GPT2ResOrder::eos_q_o_2:
            case GPT2ResOrder::eos_v_o_2:
            case GPT2ResOrder::attn_gamma_weights_3:
            case GPT2ResOrder::c_attn_bias_3:
            case GPT2ResOrder::p_attn_weights_3:
            case GPT2ResOrder::p_attn_bias_3:
            case GPT2ResOrder::mlp_rms_gamma_3:
            case GPT2ResOrder::c_fc_bias_3:
            case GPT2ResOrder::c_proj_weights_3:
            case GPT2ResOrder::c_proj_bias_3:
            case GPT2ResOrder::eos_k_o_3:
            case GPT2ResOrder::eos_q_o_3:
            case GPT2ResOrder::eos_v_o_3:
            case GPT2ResOrder::attn_gamma_weights_4:
            case GPT2ResOrder::c_attn_bias_4:
            case GPT2ResOrder::p_attn_weights_4:
            case GPT2ResOrder::p_attn_bias_4:
            case GPT2ResOrder::mlp_rms_gamma_4:
            case GPT2ResOrder::c_fc_bias_4:
            case GPT2ResOrder::c_proj_weights_4:
            case GPT2ResOrder::c_proj_bias_4:
            case GPT2ResOrder::eos_k_o_4:
            case GPT2ResOrder::eos_q_o_4:
            case GPT2ResOrder::eos_v_o_4:
            case GPT2ResOrder::attn_gamma_weights_5:
            case GPT2ResOrder::c_attn_bias_5:
            case GPT2ResOrder::p_attn_weights_5:
            case GPT2ResOrder::p_attn_bias_5:
            case GPT2ResOrder::mlp_rms_gamma_5:
            case GPT2ResOrder::c_fc_bias_5:
            case GPT2ResOrder::c_proj_weights_5:
            case GPT2ResOrder::c_proj_bias_5:
            case GPT2ResOrder::eos_k_o_5:
            case GPT2ResOrder::eos_q_o_5:
            case GPT2ResOrder::eos_v_o_5:
            case GPT2ResOrder::mlp_after_rms_gamma: {
                ret = res_pack::read_and_write_npy(static_cast<int>(static_cast<GPT2ResOrder>(i)), res_path_prefix,
                    res_paths[static_cast<GPT2ResOrder>(i)], dst_file);
                break;
            }
#if defined(FIX_VER) || defined(AWQ_VER)
            case GPT2ResOrder::c_attn_weights_0:
            case GPT2ResOrder::reordered_c_fc_weights_0:
            case GPT2ResOrder::c_attn_weights_1:
            case GPT2ResOrder::reordered_c_fc_weights_1:
            case GPT2ResOrder::c_attn_weights_2:
            case GPT2ResOrder::reordered_c_fc_weights_2:
            case GPT2ResOrder::c_attn_weights_3:
            case GPT2ResOrder::reordered_c_fc_weights_3:
            case GPT2ResOrder::c_attn_weights_4:
            case GPT2ResOrder::reordered_c_fc_weights_4:
            case GPT2ResOrder::c_attn_weights_5:
            case GPT2ResOrder::reordered_c_fc_weights_5: {
                ret = res_pack::read_and_write_npy_int8toint4(static_cast<int>(static_cast<GPT2ResOrder>(i)), res_path_prefix,
                    res_paths[static_cast<GPT2ResOrder>(i)], dst_file);
                break;
            }
            case GPT2ResOrder::c_attn_scales_0:
            case GPT2ResOrder::c_fc_scales_0:
            case GPT2ResOrder::c_attn_scales_1:
            case GPT2ResOrder::c_fc_scales_1:
            case GPT2ResOrder::c_attn_scales_2:
            case GPT2ResOrder::c_fc_scales_2:
            case GPT2ResOrder::c_attn_scales_3:
            case GPT2ResOrder::c_fc_scales_3:
            case GPT2ResOrder::c_attn_scales_4:
            case GPT2ResOrder::c_fc_scales_4:
            case GPT2ResOrder::c_attn_scales_5:
            case GPT2ResOrder::c_fc_scales_5: {
                ret = res_pack::read_and_write_scales(static_cast<int>(static_cast<GPT2ResOrder>(i)), res_path_prefix,
                    res_paths[static_cast<GPT2ResOrder>(i)],dst_file);
                break;
            }
#endif
#ifdef FIX_VER
            case GPT2ResOrder::decode_layer_scales: {
                ret = res_pack::read_and_write_json(static_cast<int>(cur_order), res_path_prefix,
                    res_paths[cur_order], dst_file);
                break;
            }
#endif
            default:;
        }

        ASSERT_TRUE(ret);
    }


    dst_file.close();
    std::cout << "ResPack test passed!" << std::endl;

}