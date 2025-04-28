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
#include "utils/utils.hpp"

using namespace BatmanInfer;

constexpr int max_seq_len    = 16;
constexpr int max_batch_size = 20;
constexpr int dict_size      = 6003;
constexpr int hidden_size    = 768;
constexpr int tensor_max_dim = 6;

struct AttnHyperParams {
    static constexpr float attn_gemm_i_scale = 0.006409900328692268f;
    static constexpr int   attn_gemm_i_zero  = -6;
    static constexpr float attn_gemm_o_scale = 0.08648063435274012f;
    static constexpr int   attn_gemm_o_zero  = -9;
    static constexpr float query_scale       = 0.04602363623824774f;
    static constexpr int   query_zp          = -11;
    static constexpr float value_scale       = 0.08648063435274012f;
    static constexpr int   value_zp          = -9;
    static constexpr float key_scale         = 0.0459319413877001f;
    static constexpr int   key_zp            = -18;
    static constexpr float softmax_q_scale   = 0.00392156862745098f;
    static constexpr int   softmax_zp        = -128;
    static constexpr float proj_in_scale     = 0.0865f;
    static constexpr int   proj_in_zp        = -9;
};

const PermutationVector q_perm{0, 2, 1, 3};
const PermutationVector k_perm{2, 0, 1, 3};
const PermutationVector qkv_o_perm{0, 2, 1, 3};

struct MLPHyperParams {
    static constexpr float fc1_input_scale        = 0.006902442025203331f;
    static constexpr int   fc1_input_zero_point   = -9;
    static constexpr float fc1_output_scale       = 0.1969725440530216f;
    static constexpr int   fc1_output_zero_point  = -19;
    static constexpr float gelu_output_scale      = 0.11368115240452337f;
    static constexpr int   gelu_output_zero_point = -127;
};

// 为资源的打包设定一个顺序
enum class GPT2ResOrder {
    gather_weight = 0,
    add_weight,
    attn_gamma_weight,
    attn_qkv_weight,
    attn_qkv_weight_scale,
    attn_qkv_bias,
    attn_c_proj_weight,
    attn_c_proj_bias,
    mlp_gamma_weight,
    c_fc_weight,
    c_fc_weight_scale,
    c_fc_bias,
    c_proj_weight,
    c_proj_bias,
    rms_gamma_weight,
    lm_head_weight,
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
     * @return 返回码
     */
    BIErrCode dynamic_configure_all_layers(const std::vector<int> &tensor_shape);

private:
    BIMemoryGroup                               _memory_group;
    std::unique_ptr<BIMemoryGroupResourceScope> _scope_manager;

private:
    BITensor _ori_input_tensor;
    BITensor _ori_gather_output_tensor;
    BITensor _ori_attn_rms_output_tensor;
    BITensor _ori_add_output_tensor;
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

    BITensor _sub_input_tensor;
    BITensor _sub_gather_output_tensor;
    BITensor _sub_add_weight_tensor;
    BITensor _sub_add_output_tensor;
    BITensor _sub_attn_output_tensor;
    BITensor _sub_mlp_input_tensor;
    BITensor _sub_mlp_output_tensor;
    BITensor _sub_add_mlp_output_tensor;
    BITensor _sub_mlp_rms_output_tensor;
    BITensor _sub_lm_head_output_tensor;

    BINEGather             _gather_layer;
    BINEArithmeticAddition _add_layer;
    BINEAttentionLowpLayer _attn_lowp_layer;
    BINEArithmeticAddition _attn_rms_add_layer;
    BINEMLPLayer           _mlp_layer;
    BINEArithmeticAddition _add_mlp_layer;
    BINERMSNormLayer       _rms_norm_layer;
    BINEGEMM               _lm_head_layer;

    BIQuantizationInfo _c_fc_weight_q_info;
    std::vector<int> _output_positions;

};
