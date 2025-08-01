//
// Created by Mason on 2025/2/10.
//

#pragma once

#include <runtime/bi_i_function.hpp>
#include <runtime/bi_i_memory_manager.hpp>
#include <runtime/bi_i_weights_manager.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>
#include <function_info/bi_GEMMInfo.h>
#include <runtime/neon/bi_ne_functions.h>

#include <data/core/bi_types.hpp>
#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_tensor.hpp>
#include "bi_ne_copy.hpp"


namespace BatmanInfer {
    /**
     * FFN神经网络
     * 1. 先进行LayerNorm
     * 2. 再进行扩展维度(NEGemm)
     * 3. 进行激活函数操作（合并Gemm）
     * 4. 再进行恢复维度（恢复维度）
     */
    class BINEFeedForwardLayer : public BIIFunction {
    public:
        explicit BINEFeedForwardLayer(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINEFeedForwardLayer() : BINEFeedForwardLayer(BIMemoryManagerOnDemand::make_default()) {
        }

        BINEFeedForwardLayer(const BINEFeedForwardLayer &) = delete;

        BINEFeedForwardLayer(BINEFeedForwardLayer &&) = delete;

        BINEFeedForwardLayer &operator=(const BINEFeedForwardLayer &) = delete;

        BINEFeedForwardLayer &operator=(BINEFeedForwardLayer &&) = delete;

        ~BINEFeedForwardLayer() override;

        void dynamic_configure(const BIITensor *input,
                               const size_t &batch_size);

        /**
         *
         * @param input 输入张量
         * @param fc_weights
         * @param fc_bias
         * @param proj_weights
         * @param proj_bias
         * @param gamma
         * @param output
         * @param max_batch_size
         * @param max_seq_len
         */
        void configure(const BIITensor *input,
                       const BIITensor *fc_weights,
                       const BIITensor *fc_bias,
                       const BIITensor *proj_weights,
                       const BIITensor *proj_bias,
                       const BIITensor *gamma,
                       const BIActivationLayerInfo &act_info,
                       BIITensor *output,
                       const size_t &max_batch_size,
                       const size_t &max_seq_len);

        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *fc_weights,
                                 const BIITensorInfo *fc_bias,
                                 const BIITensorInfo *proj_weights,
                                 const BIITensorInfo *proj_bias,
                                 const BIITensorInfo *gamma,
                                 const BIITensorInfo *output);

        // 继承函数
        void run();

        void prepare() override;

    private:
        // 内存管理
        BIMemoryGroup _memory_group;
        std::unique_ptr<BIMemoryGroupResourceScope> _scope_mg;

        // 算子操作
        BINERMSNormLayer _rms_layer; // 用于执行归一操作的层

        BINEGEMM _c_fc_fuse_act; // 用于进行扩展维度 (融合激活函数)

        BINEGEMM _c_proj; // 用于进行恢复维度

        BINECopy _copy_f; // 拷贝张量算子

        // 中间张量处理
        BITensor _norm_output; // 归一化输出
        BITensor _fuse_output; // 扩展激活函数输出
        BITensor _proj_output; // 恢复维度输出

        BITensor _sub_norm_output, _sub_fuse_output, _sub_proj_output;

        BITensorInfo _sub_norm_output_info, _sub_fuse_output_info, _sub_proj_output_info;

        // 参数长度
        size_t _max_batch;
        size_t _max_seq;

        size_t _batch_size = 1;
        // size_t _seq_len = 1;

        // 其他参数 (是否准备就绪)
        bool _is_prepared;
    };
}


