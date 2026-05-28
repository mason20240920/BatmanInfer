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
#include "runtime/neon/functions/BINEMultiGPTBlock.hpp"
#include "utils/utils.hpp"

using namespace BatmanInfer;

extern int max_seq_len;
extern int max_batch_size;
extern int dict_size;
extern int hidden_size;
extern int tensor_max_dim;
extern int layer_num;
extern int fc_out_size;
extern int head_bs;


const PermutationVector q_perm{0, 2, 1, 3};
const PermutationVector k_perm{2, 0, 1, 3};
const PermutationVector qkv_o_perm{0, 2, 1, 3};

// 为资源的打包设定一个顺序 6层 GPT[单层和 3层需要复用该逻辑]
enum class GPT2ResOrder {
    transformer_wte_weight = 0,
    add_wte_weight,

    attn_gamma_weights_0,
    c_attn_weights_0,
    c_attn_scales_0,
    c_attn_bias_0,
    p_attn_weights_0,
    p_attn_bias_0,
    mlp_rms_gamma_0,
    reordered_c_fc_weights_0,
    c_fc_scales_0,
    c_fc_bias_0,
    c_proj_weights_0,
    c_proj_bias_0,
    eos_k_o_0,
    eos_q_o_0,
    eos_v_o_0,

    attn_gamma_weights_1,
    c_attn_weights_1,
    c_attn_scales_1,
    c_attn_bias_1,
    p_attn_weights_1,
    p_attn_bias_1,
    mlp_rms_gamma_1,
    reordered_c_fc_weights_1,
    c_fc_scales_1,
    c_fc_bias_1,
    c_proj_weights_1,
    c_proj_bias_1,
    eos_k_o_1,
    eos_q_o_1,
    eos_v_o_1,

    attn_gamma_weights_2,
    c_attn_weights_2,
    c_attn_scales_2,
    c_attn_bias_2,
    p_attn_weights_2,
    p_attn_bias_2,
    mlp_rms_gamma_2,
    reordered_c_fc_weights_2,
    c_fc_scales_2,
    c_fc_bias_2,
    c_proj_weights_2,
    c_proj_bias_2,
    eos_k_o_2,
    eos_q_o_2,
    eos_v_o_2,

    attn_gamma_weights_3,
    c_attn_weights_3,
    c_attn_scales_3,
    c_attn_bias_3,
    p_attn_weights_3,
    p_attn_bias_3,
    mlp_rms_gamma_3,
    reordered_c_fc_weights_3,
    c_fc_scales_3,
    c_fc_bias_3,
    c_proj_weights_3,
    c_proj_bias_3,
    eos_k_o_3,
    eos_q_o_3,
    eos_v_o_3,

    attn_gamma_weights_4,
    c_attn_weights_4,
    c_attn_scales_4,
    c_attn_bias_4,
    p_attn_weights_4,
    p_attn_bias_4,
    mlp_rms_gamma_4,
    reordered_c_fc_weights_4,
    c_fc_scales_4,
    c_fc_bias_4,
    c_proj_weights_4,
    c_proj_bias_4,
    eos_k_o_4,
    eos_q_o_4,
    eos_v_o_4,

    attn_gamma_weights_5,
    c_attn_weights_5,
    c_attn_scales_5,
    c_attn_bias_5,
    p_attn_weights_5,
    p_attn_bias_5,
    mlp_rms_gamma_5,
    reordered_c_fc_weights_5,
    c_fc_scales_5,
    c_fc_bias_5,
    c_proj_weights_5,
    c_proj_bias_5,
    eos_k_o_5,
    eos_q_o_5,
    eos_v_o_5,

    mlp_after_rms_gamma,

#ifdef FIX_VER
    decode_layer_scales,  // JSON格式，包含所有层的量化参数
#endif

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
    BIGPT2Model(int max_seq_len, int max_batch_size, int dict_size, int hidden_size, int tensor_max_dim, int layer_num, int head_bs);

    BIErrCode bi_init(const char *data_in, size_t data_size, std::vector< std::vector<float> > &output_vec, unsigned int &kv_cache_id) override;
    BIErrCode bi_set_input(std::vector< std::vector<unsigned int> > &input_vec, std::vector< std::vector<unsigned int> > &kv_cache_id_map) override;
    BIErrCode bi_run(std::vector<size_t> &avail_lens, std::vector< std::vector<float> > &output_vec, std::vector<unsigned int> &kv_block_ids, bool is_init) override;
    bool bi_valid_decode_ids(std::vector<unsigned int> &kv_block_ids) override;
    BIErrCode bi_release_kvcache_block(std::vector<unsigned int> &kv_block_ids) override;
    BIErrCode bi_release_kvcache_leaf_block(std::vector<unsigned int> &kv_block_ids) override;
    void bi_get_avaliable_kvblock_count(unsigned int &avaliable_kvblock_count) override;
    BIErrCode bi_reset(unsigned int &kv_cache_id) override;
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

    BIErrCode load_weight_tensor(BITensor &tensor, GPT2ResOrder res_order, OrderPtrMap &order2ptr, bool need_transpose);

    BIErrCode load_weight_tensors(std::array<BITensor, 6> &tensors, GPT2ResOrder res_order, OrderPtrMap &order2ptr, int step);

    BIErrCode load_weight_tensor_and_dequantization(BITensor &tensor, BITensor &tensor_output, GPT2ResOrder res_order, OrderPtrMap &order2ptr, std::vector<float> &scales);

    BIErrCode load_scale_vector(std::vector<float> &scales, GPT2ResOrder res_order, OrderPtrMap &order2ptr);

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
    BIErrCode dynamic_configure_all_layers(const std::vector<int> &tensor_shape,  std::vector< std::vector<unsigned int> > &kv_cache_id_map);

    void print_tensor(const BatmanInfer::BITensor &tensor, const std::string &name = "temp", const BatmanInfer::BIIOFormatInfo::PrintRegion region = BatmanInfer::BIIOFormatInfo::PrintRegion::Full);

    std::pair<int8_t, int8_t> unpack_int8_to_int4(int8_t packed);
private:
    BIMemoryGroup                               _memory_group;
    std::unique_ptr<BIMemoryGroupResourceScope> _scope_manager;

private:
    BITensor _ori_input_tensor;
    BITensor _ori_gather_output_tensor;
    BITensor _ori_add_output_tensor;
    BITensor _ori_split_add_output_tensor;
    BITensor _ori_multi_gpt_o_tensor;
    BITensor _ori_mlp_rms_output_tensor;
    BITensor _ori_lm_head_output_tensor;

    BITensor _gather_weight_tensor;
    BITensor _add_weight_tensor;
    std::array<BITensor, 6> _attn_gamma_weight_tensors;
    std::array<BITensor, 6> _c_attn_weight_tensors;        //awq反量化结果
    std::array<BITensor, 6> _c_attn_unpacked_weight_tensors; //解包后的数据（fix和awq共用）
    std::array<BITensor, 6> _c_attn_bias_tensors;
    std::array<BITensor, 6> _p_attn_weight_tensors;
    std::array<BITensor, 6> _p_attn_bias_tensors;
    std::array<BITensor, 6> _mlp_weight_tensors;
    std::array<BITensor, 6> _c_fc_weight_tensors;       //awq反量化结果
    std::array<BITensor, 6> _c_fc_unpacked_weight_tensors;  //解包后的数据（fix和awq共用）
    std::array<BITensor, 6> _c_fc_bias_tensors;
    std::array<BITensor, 6> _c_proj_weight_tensors;
    std::array<BITensor, 6> _c_proj_bias_tensors;
    std::array<BITensor, 6> _eos_k_smooth_o_tensor;
    std::array<BITensor, 6> _eos_q_smooth_o_tensor;
    std::array<BITensor, 6> _eos_v_smooth_o_tensor;
    BITensor _rms_gamma_weight_tensor;
    BITensor _lm_head_weight_tensor;

    BITensor _sub_input_tensor;
    BITensor _sub_gather_output_tensor;
    BITensor _sub_add_weight_tensor;
    BITensor _sub_add_output_tensor;
    BITensor _sub_split_add_output_tensor;
    BITensor _sub_multi_gpt_o_tensor;
    BITensor _sub_mlp_rms_output_tensor;
    BITensor _sub_lm_head_output_tensor;

    BINEGather             _gather_layer;
    BINEArithmeticAddition _add_layer;
    BINEMultiGPTBlock      _gpt_multi_block_layer;
    BINERMSNormLayer       _rms_norm_layer;
    BINEGEMM               _lm_head_layer;

    BIGPTGlobalConfig gpt_block_config;
    std::vector<BIGPTLayerConfig> gpt_layer_configs;

    BIITensorPack _pack;

    unsigned int kv_root_id;
};
