//
// Created by holynova on 25-4-18.
//

#pragma once

#include "sdk/bi_sdk_api.h"

#include "runtime/neon/bi_ne_functions.h"
#include "runtime/bi_tensor.hpp"
#include "runtime/bi_memory_manager_on_demand.hpp"
#include "runtime/bi_memory_group.hpp"
#include "runtime/bi_scheduler.hpp"
#include "runtime/neon/functions/BINEIntentMGPTBlock.hpp"
#include "utils/utils.hpp"

using namespace BatmanInfer;

constexpr int max_seq_len    = 64;
constexpr int max_batch_size = 1;
constexpr int dict_size      = 21128;
constexpr int hidden_size    = 768;
constexpr int tensor_max_dim = 6;
constexpr int layer_num      = 6;
constexpr int class_num      = 5;

const PermutationVector q_perm{0, 2, 1, 3};
const PermutationVector k_perm{2, 0, 1, 3};
const PermutationVector qkv_o_perm{0, 2, 1, 3};

// 为资源的打包设定一个顺序 6层 GPT
enum class GPT2ResOrder {
    transformer_wte_weight = 0,
    add_wte_weight,

    attn_layernorm_weight_0,
    attn_layernorm_bias_0,
    c_attn_weights_0,
    c_attn_bias_0,
    p_attn_weights_0,
    p_attn_bias_0,
    mlp_layernorm_weights_0,
    mlp_layernorm_bias_0,
    reordered_c_fc_weights_0,
    c_fc_bias_0,
    c_proj_weights_0,
    c_proj_bias_0,

    attn_layernorm_weight_1,
    attn_layernorm_bias_1,
    c_attn_weights_1,
    c_attn_bias_1,
    p_attn_weights_1,
    p_attn_bias_1,
    mlp_layernorm_weights_1,
    mlp_layernorm_bias_1,
    reordered_c_fc_weights_1,
    c_fc_bias_1,
    c_proj_weights_1,
    c_proj_bias_1,

    attn_layernorm_weight_2,
    attn_layernorm_bias_2,
    c_attn_weights_2,
    c_attn_bias_2,
    p_attn_weights_2,
    p_attn_bias_2,
    mlp_layernorm_weights_2,
    mlp_layernorm_bias_2,
    reordered_c_fc_weights_2,
    c_fc_bias_2,
    c_proj_weights_2,
    c_proj_bias_2,

    attn_layernorm_weight_3,
    attn_layernorm_bias_3,
    c_attn_weights_3,
    c_attn_bias_3,
    p_attn_weights_3,
    p_attn_bias_3,
    mlp_layernorm_weights_3,
    mlp_layernorm_bias_3,
    reordered_c_fc_weights_3,
    c_fc_bias_3,
    c_proj_weights_3,
    c_proj_bias_3,

    attn_layernorm_weight_4,
    attn_layernorm_bias_4,
    c_attn_weights_4,
    c_attn_bias_4,
    p_attn_weights_4,
    p_attn_bias_4,
    mlp_layernorm_weights_4,
    mlp_layernorm_bias_4,
    reordered_c_fc_weights_4,
    c_fc_bias_4,
    c_proj_weights_4,
    c_proj_bias_4,

    attn_layernorm_weight_5,
    attn_layernorm_bias_5,
    c_attn_weights_5,
    c_attn_bias_5,
    p_attn_weights_5,
    p_attn_bias_5,
    mlp_layernorm_weights_5,
    mlp_layernorm_bias_5,
    reordered_c_fc_weights_5,
    c_fc_bias_5,
    c_proj_weights_5,
    c_proj_bias_5,

    final_layernorm_weights,
    final_layernorm_bias,

    lm_score_weights,

    all_res_count,
};

// 资源中每一块的头信息
typedef struct GPT2ResHeader_ {
    char         data_type[8];
    int          shape[6];
    unsigned int data_length;
    unsigned int res_order;

    GPT2ResHeader_() {
        memset(this, 0, sizeof(GPT2ResHeader_));
    }
} GPT2ResHeader;

using OrderPtrMap = std::map<GPT2ResOrder, char *>;

class BIGPT2Model final : public BIModelInterfaceBase {
public:
    explicit BIGPT2Model(std::shared_ptr<BIIMemoryManager> memory_manager);
    BIGPT2Model();

    BIErrCode bi_init(const char *data_in, size_t data_size) override;
    BIErrCode bi_set_input(std::vector< std::vector<unsigned int> > &input_vec) override;
    BIErrCode bi_run(std::vector< std::vector<float> > &output_vec) override;
    void set_threads_num(unsigned int num_threads) override;

private:
    /**
     * 根据传入数据，填充一个一维 tensor 的内容
     * @tparam T 传入数据类型
     * @param tensor 要填充的 tensor
     * @param data_in 传入数据的具体值
     * @return 返回码
     */
    template<typename T>
    BIErrCode fill_tensor_data_1D(BITensor &tensor, std::vector<T> &data_in);

    /**
     * 根据传入数据，填充一个二维 tensor 的内容
     * @tparam T 传入数据类型
     * @param tensor 要填充的 tensor
     * @param data_in 传入数据的具体值
     * @return 返回码
     */
    template<typename T>
    BIErrCode fill_tensor_data_2D(BITensor &tensor, std::vector< std::vector<T> > &data_in);

    /**
     * 根据传入数据，填充一个二维 tensor 的内容
     * @tparam T 传入数据类型
     * @param tensor 要填充的 tensor
     * @param data_in 传入数据的具体值
     * @param max_item_len 最长 item 的长度
     * @return
     */
    template<typename T>
    BIErrCode fill_tensor_data_2D(BITensor &tensor, std::vector< std::vector<T> > &data_in, const size_t max_item_len);

    BIErrCode parse_model_data(const char *data_in, size_t data_size, OrderPtrMap &order2ptr);

    BIErrCode load_weight_tensor(BITensor &tensor, GPT2ResOrder res_order, OrderPtrMap &order2ptr);

    BIErrCode load_weight_tensors(std::array<BITensor, layer_num> &tensors, GPT2ResOrder res_order, OrderPtrMap &order2ptr, int step);

    BIErrCode load_hyper_params(OrderPtrMap &order2ptr);

    BIErrCode load_all_non_dynamic_tensors(OrderPtrMap &order2ptr);

    /**
     * 根据传入输入的形状，重新设置所有动态算子的形状
     * @param tensor_shape 传入的输入的形状
     * @return 返回码
     */
    BIErrCode set_all_intermediate_tensors(const std::vector<int> &tensor_shape);

    /**
     * 初始时设置对所有 layer 执行 configure 操作
     * @param tensor_shape 传入的输入的形状
     * @return 返回码
     */
    BIErrCode init_configure_all_layers(const std::vector<int> &tensor_shape);

    /**
     * 中间执行过程中，对所有 layer 执行动态 configure
     * @param tensor_shape 传入的输入的形状
     * @return 返回码
     */
    BIErrCode dynamic_configure_all_layers(const std::vector<int> &tensor_shape);

    void print_tensor(const BatmanInfer::BITensor &tensor, const std::string &name = "temp", const BatmanInfer::BIIOFormatInfo::PrintRegion region = BatmanInfer::BIIOFormatInfo::PrintRegion::Full);

    std::pair<int8_t, int8_t> unpack_int8_to_int4(int8_t packed);
private:
    BIMemoryGroup                               _memory_group;
    std::unique_ptr<BIMemoryGroupResourceScope> _scope_manager;

private:
    BITensor _ori_input_tensor;
    BITensor _ori_gather_output_tensor;
    BITensor _ori_add_output_tensor;
    BITensor _ori_multi_gpt_o_tensor;
    BITensor _ori_final_ln_o_tensor;
    BITensor _ori_lm_head_output_tensor;

    BITensor _gather_weight_tensor;
    BITensor _add_weight_tensor;
    std::array<BITensor, layer_num> _attn_gamma_weight_tensors;
    std::array<BITensor, layer_num> _attn_gamma_bias_tensors;
    std::array<BITensor, layer_num> _c_attn_weight_tensors;
    std::array<BITensor, layer_num> _c_attn_bias_tensors;
    std::array<BITensor, layer_num> _p_attn_weight_tensors;
    std::array<BITensor, layer_num> _p_attn_bias_tensors;
    std::array<BITensor, layer_num> _mlp_weight_tensors;
    std::array<BITensor, layer_num> _mlp_bias_tensors;
    std::array<BITensor, layer_num> _c_fc_weight_tensors;
    std::array<BITensor, layer_num> _c_fc_bias_tensors;
    std::array<BITensor, layer_num> _c_proj_weight_tensors;
    std::array<BITensor, layer_num> _c_proj_bias_tensors;
    BITensor _final_layernorm_weight_tensor;
    BITensor _final_layernorm_bias_tensor;
    BITensor _lm_score_weight_tensor;

    BITensor _sub_input_tensor;
    BITensor _sub_gather_output_tensor;
    BITensor _sub_add_weight_tensor;
    BITensor _sub_add_output_tensor;
    BITensor _sub_multi_gpt_o_tensor;
    BITensor _sub_final_ln_o_tensor;
    BITensor _sub_lm_head_output_tensor;

    BINEGather             _gather_layer;
    BINEArithmeticAddition _add_layer;
    BINEIntentMGPTBlock _gpt_multi_block_layer;
    BINELayerNormLayer _final_layernorm_layer;
    BINEGEMM _lm_head_layer;

    BIIntentGPTGlobalConfig gpt_block_config;
    std::vector<BIIntentGPTLayerConfig> gpt_layer_configs;
};
