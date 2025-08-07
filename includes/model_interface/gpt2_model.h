//
// Created by holynova on 25-4-18.
//

#pragma once

// #define QYW_PRINT

#include "sdk/bi_sdk_api.h"

#include "runtime/neon/bi_ne_functions.h"
#include "runtime/bi_tensor.hpp"
#include "runtime/bi_memory_manager_on_demand.hpp"
#include "runtime/bi_memory_group.hpp"
#include "runtime/bi_scheduler.hpp"
#include "utils/utils.hpp"

using namespace BatmanInfer;

constexpr int max_seq_len    = 16;
constexpr int max_batch_size = 50;
constexpr int dict_size      = 6003;
constexpr int hidden_size    = 768;
constexpr int tensor_max_dim = 6;

#ifdef FIX_VER
typedef struct HyperParameters_ {
    float attn_input_scale;
    float attn_output_scale;
    float q_output_scale;
    float k_output_scale;
    float v_output_scale;
    float out_input_scale;
    float fc1_input_scale;
    float fc1_output_scale;
    float fc2_input_scale;
    int   attn_input_zp;
    int   attn_output_zp;
    int   q_output_zp;
    int   k_output_zp;
    int   v_output_zp;
    int   out_input_zp;
    int   fc1_input_zp;
    int   fc1_output_zp;
    int   fc2_input_zp;
}HyperParameters;

typedef struct AttnHyperParams_ {
    static constexpr float softmax_q_scale   = 0.00392156862745098f;
    static constexpr int   softmax_zp        = -128;
    float attn_gemm_i_scale;
    int   attn_gemm_i_zero;
    float attn_gemm_o_scale;
    int   attn_gemm_o_zero;
    float query_scale;
    int   query_zp;
    float value_scale;
    int   value_zp;
    float key_scale;
    int   key_zp;
    float proj_in_scale;
    int   proj_in_zp;

    AttnHyperParams_() :
        attn_gemm_i_scale(0),
        attn_gemm_i_zero(0),
        attn_gemm_o_scale(0),
        attn_gemm_o_zero(0),
        query_scale(0),
        query_zp(0),
        value_scale(0),
        value_zp(0),
        key_scale(0),
        key_zp(0),
        proj_in_scale(0),
        proj_in_zp(0)
    {}

}AttnHyperParams;
#endif

const PermutationVector q_perm{0, 2, 1, 3};
const PermutationVector k_perm{2, 0, 1, 3};
const PermutationVector qkv_o_perm{0, 2, 1, 3};

#ifdef FIX_VER
typedef struct MLPHyperParams_ {
    float fc1_input_scale;
    int   fc1_input_zero_point;
    float fc1_output_scale;
    int   fc1_output_zero_point;
    float gelu_output_scale;
    int   gelu_output_zero_point;

    MLPHyperParams_() :
        fc1_input_scale(0),
        fc1_input_zero_point(0),
        fc1_output_scale(0),
        fc1_output_zero_point(0),
        gelu_output_scale(0),
        gelu_output_zero_point(0)
    {}

}MLPHyperParams;
#endif

// 为资源的打包设定一个顺序
enum class GPT2ResOrder {
    gather_weight = 0,
    add_weight,
    attn_gamma_weight,
    attn_qkv_weight,
#ifdef FIX_VER
    attn_qkv_weight_scale,
#endif
    attn_qkv_bias,
    attn_c_proj_weight,
    attn_c_proj_bias,
    mlp_gamma_weight,
    c_fc_weight,
#ifdef FIX_VER
    c_fc_weight_scale,
#endif
    c_fc_bias,
    c_proj_weight,
    c_proj_bias,
    rms_gamma_weight,
    lm_head_weight,
#ifdef FIX_VER
    decode_layer_scales,
#endif
    eos_k_smooth_o,
    eos_q_smooth_o,
    eos_v_smooth_o,
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

    BIErrCode load_weight_tensor(BITensor &tensor, GPT2ResOrder res_order, OrderPtrMap &order2ptr);

    BIErrCode load_scale_vector(std::vector<float> &scales, GPT2ResOrder res_order, OrderPtrMap &order2ptr);

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
     * @return 返回码
     */
    BIErrCode init_configure_all_layers();

    /**
     * 中间执行过程中，对所有 layer 执行动态 configure
     * @param tensor_shape 传入的输入的形状
     * @param kv_cache_id_map 上一步推理生成的 kv_cache_id 数据
     * @return 返回码
     */
    BIErrCode dynamic_configure_all_layers(const std::vector<int> &tensor_shape, std::vector< std::vector<unsigned int> > &kv_cache_id_map);

    void print_tensor(const BatmanInfer::BITensor &tensor, const std::string &name = "temp", const BatmanInfer::BIIOFormatInfo::PrintRegion region = BatmanInfer::BIIOFormatInfo::PrintRegion::Full);

private:
    BIMemoryGroup                               _memory_group;
    std::unique_ptr<BIMemoryGroupResourceScope> _scope_manager;

private:
    BITensor _ori_input_tensor;
    BITensor _ori_gather_output_tensor;
    BITensor _ori_attn_rms_output_tensor;
    BITensor _ori_add_output_tensor;
    BITensor _ori_split_add_output_tensor;
    BITensor _ori_attn_output_tensor;
    BITensor _ori_mlp_output_tensor;
    BITensor _ori_add_mlp_output_tensor;
    BITensor _ori_mlp_rms_output_tensor;
    BITensor _ori_lm_head_output_tensor;

    BITensor _gather_weight_tensor;
    BITensor _add_weight_tensor;
    BITensor _attn_gamma_weight_tensor;
    BITensor _attn_qkv_weight_tensor;
    BITensor _attn_qkv_bias_tensor;
    BITensor _attn_c_proj_weight_tensor;
    BITensor _attn_c_proj_bias_tensor;
    BITensor _mlp_gamma_weight_tensor;
    BITensor _c_fc_weight_tensor;
    BITensor _c_fc_bias_tensor;
    BITensor _c_proj_weight_tensor;
    BITensor _c_proj_bias_tensor;
    BITensor _rms_gamma_weight_tensor;
    BITensor _lm_head_weight_tensor;
    BITensor _eos_k_smooth_o_tensor;
    BITensor _eos_q_smooth_o_tensor;
    BITensor _eos_v_smooth_o_tensor;

    BITensor _sub_input_tensor;
    BITensor _sub_gather_output_tensor;
    BITensor _sub_add_weight_tensor;
    BITensor _sub_add_output_tensor;
    BITensor _sub_split_add_output_tensor;
    BITensor _sub_attn_output_tensor;
    BITensor _sub_mlp_input_tensor;
    BITensor _sub_mlp_output_tensor;
    BITensor _sub_add_mlp_output_tensor;
    BITensor _sub_mlp_rms_output_tensor;
    BITensor _sub_lm_head_output_tensor;

    BINEGather             _gather_layer;
    BINEArithmeticAddition _add_layer;
#ifdef FIX_VER
    BINEAttentionLowpLayer _attn_lowp_layer;
#elifdef FLOAT_VER
    BINEAttentionLayer     _attn_layer;
#endif
    BINEArithmeticAddition _attn_rms_add_layer;
#ifdef FIX_VER
    BINEMLPLayer           _mlp_layer;
#elifdef FLOAT_VER
    BINEFeedForwardLayer   _mlp_layer;
#endif
    BINEArithmeticAddition _add_mlp_layer;
    BINERMSNormLayer       _rms_norm_layer;
    BINEGEMM               _lm_head_layer;

#ifdef FIX_VER
    BIQuantizationInfo _c_fc_weight_q_info;
#endif
    // std::vector<int> _output_positions;

    BIITensorPack _pack;
#ifdef FIX_VER
    AttnHyperParams _attn_hyper_params;
    MLPHyperParams  _mlp_hyper_params;
#endif

    unsigned int kv_root_id;
};
