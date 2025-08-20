/**
 * 创建多层GPT Block, 实现多层调度
 * Created by Mason on 2025/7/24.
 */

#pragma once

#include <runtime/neon/functions/BINEGPT2Block.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>

namespace BatmanInfer {
    /**
     * @brief 单层权重张量
     */
    struct BIGPTLayerConfig {
        // 每层的权重张量
        const BIITensor *ln_1_weight;
        const BIITensor *c_attn_weights;
        const BIITensor *c_attn_bias;
        const BIITensor *o_attn_weights;
        const BIITensor *o_attn_bias;
        const BIITensor *fc_weights;
        const BIITensor *fc_bias;
        const BIITensor *proj_weights;
        const BIITensor *proj_bias;
        const BIITensor *ln_2_weight;

        // 每层有不同的配置
        BIActivationLayerInfo act_info;
        int layer_idx;

        // 构造函数
        BIGPTLayerConfig(): ln_1_weight(nullptr), c_attn_weights(nullptr),
                            c_attn_bias(nullptr), o_attn_weights(nullptr),
                            o_attn_bias(nullptr), fc_weights(nullptr),
                            fc_bias(nullptr), proj_weights(nullptr),
                            proj_bias(nullptr), ln_2_weight(nullptr), layer_idx(0) {
        }

        // 验证配置
        bool is_valid() const {
            return ln_1_weight && c_attn_weights && c_attn_bias &&
                   o_attn_weights && o_attn_bias && fc_weights &&
                   fc_bias && proj_weights && proj_bias && ln_2_weight;
        }
    };

    struct BIGPTGlobalConfig {
        PermutationVector q_perm;
        PermutationVector k_perm;
        PermutationVector qkv_perm;
        size_t hidden_size;
        size_t max_seq_len;
        size_t max_batch_size;

        BIGPTGlobalConfig() : hidden_size(0), max_seq_len(0), max_batch_size(0) {
        }
    };

    // Forward declaration
    class BIITensor;

    class BINEMultiGPTBlock final : public BIIFunction {
    public:
        explicit BINEMultiGPTBlock(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINEMultiGPTBlock(): BINEMultiGPTBlock(BIMemoryManagerOnDemand::make_default()) {
        }

        BINEMultiGPTBlock(const BINEMultiGPTBlock &other) = delete;

        BINEMultiGPTBlock &operator=(const BINEMultiGPTBlock &other) = delete;

        BINEMultiGPTBlock(BINEMultiGPTBlock &&other) = delete;

        BINEMultiGPTBlock &operator=(BINEMultiGPTBlock &&other) = delete;

        ~BINEMultiGPTBlock() override;

        /**
         * @brief 多层配置接口 - 使用vector存储每层配置
         * @param input 输入张量
         * @param layer_configs
         * @param global_config
         * @param output
         */
        void configure(BIITensor *input,
                       const std::vector<BIGPTLayerConfig> &layer_configs,
                       const BIGPTGlobalConfig &global_config,
                       BIITensor *eos_weights,
                       BIITensor *output);

        /**
         * 高性能数组版本 - 编译确定层数
         */
        template<size_t NumLayers>
        void configure_fixed(BIITensor *input,
                             const std::array<BIGPTLayerConfig, NumLayers> &layer_configs,
                             const BIGPTGlobalConfig &global_config,
                             BIITensor *eos_weights,
                             BIITensor *output);

        void dynamic_configure(const BIITensor *input,
                               const size_t &seq_len,
                               const size_t &batch_size,
                               const std::vector<std::vector<unsigned int> > &kv_caches_vec);

        /**
         * @brief 更新KV Blocks
         * @param kv_block_ids
         */
        void get_kv_block_ids(std::vector<unsigned int> &kv_block_ids);

        void set_avail_lens(std::vector<size_t> *avail_lens) const;

        void run() override;

        void prepare() override;

    private:
        BIMemoryGroup _memory_group;

        std::vector<std::unique_ptr<BINEGPT2Block> > _layer_blocks;
        std::vector<BITensor> _intermediate_tensors; // 中间张量

        std::vector<BITensor> _sub_intermediate_tensors; // 中间张量的子张量

        BITensorInfo _sub_intermediate_tensor_info; // 中间张量的sub info信息

    private:
        size_t _hidden_size{}; // 隐藏层
        size_t _max_seq_len{}; // 最大长度输入
        size_t _max_batch_size{}; // 一块的大小
        size_t _batch_size = 1;
        size_t _seq_len = 1;
        size_t _layer_num = 1; // GPT Block的层数
        bool _is_prepared; // 是否已经完全初始化(预先把内存加载完)
        std::unique_ptr<BIMemoryGroupResourceScope> _scope_mg;
        bool _is_first_kv_cache = true; // 是否第一次KV Cache
        std::vector<std::vector<unsigned int> > _kv_decode_ids; // 进行kv cache的传递
        std::vector<std::vector<unsigned int> > _kv_history_ids; // 历史ids
        std::vector<unsigned int> _block_ids{};
        std::vector<PhysicalBlock *> _physic_blocks;


    private:
        void store_kv_cache();

        void concat_kv_cache();
    };
}



