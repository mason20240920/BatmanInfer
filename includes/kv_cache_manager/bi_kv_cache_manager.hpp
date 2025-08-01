//
// Created by Mason on 2025/5/26.
//

#pragma once
#include <mutex>

#include "bi_kv_cahe_i_manager.hpp"
#include "memory_tree.hpp"
#include "physical_block_manager.hpp"
#include "data/core/common/bi_core_common_macros.hpp"
#include "sdk/bi_sdk_api.h"

namespace BatmanInfer {
    /**
     * @brief 解码的KV Cache管理器
     * @desc: 使用单例模式, 每次运行仅初始化一次, 后续每次拷贝都是直接实例调用
     */
    class KVCacheManager final : IKVCacheManager {
    public:
        /**
         * @brief 禁止拷贝与赋值
         */
        BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(KVCacheManager);

        static void initialize(size_t num_blocks, size_t block_size, size_t max_seq_len);

        static KVCacheManager &getInstance();

        /**
         * @brief 获取当前的解码节点
         * @return 返回当前叶子的解码节点
         */
        [[nodiscard]] std::unordered_map<unsigned int, unsigned int> get_decode_ids() const;

        [[nodiscard]] unsigned int root_id() const;

        /**
         * 根据需求动态请求新的KV Cache内存
         * @param parent_id
         * @param require_block_count
         * @param inp_ids: 输入的解码ids
         */
        BIErrCode alloc_decode_next(const unsigned int parent_id,
            const int require_block_count,
            const std::vector<unsigned int> &inp_ids,
            std::vector<unsigned int> &block_ids) const;

        /**
         * 根据输出的buffer值存储到对应的block_id里
         * @param source_buffer
         * @param block_id
         * @param is_k: 是不是key_states
         * @param is_smooth_quant: 是不是SmoothQuant量化
          */
        void memcpy_decode_buffer(void *source_buffer,
                                  int block_id,
                                  int batch_idx,
                                  bool is_k = false,
                                  bool is_smooth_quant = false) const;

        /**
         * 释放空的解码序列
         * @param leaf_id: 叶子节点id
         */
        void release_useless_decode_id(const unsigned int leaf_id) const;

        /**
         * @brief 释放空的解码序列
         * @param leaf_ids
         */
        BIErrCode release_useless_decodes_id(const std::vector<unsigned int>& leaf_ids) const;

        /**
         * @brief 停止截止符
         * @param eos_ids
         */
        BIErrCode release_end_symbol(const std::vector<unsigned int>& eos_ids) const;

        /**
         * 根据序列节点号，获取解码出来的解码K, V Cache结果
         * @param leaf_id
         * @param block_ids
         */
        void decode_sequence_lst(const unsigned int leaf_id, std::vector<unsigned int> &block_ids) const;

        /**
         * 解码序列的内存Block数组
         * @param block_ids
         * @param block_mem_lst
         */
        void decode_sequence_blocks(const std::vector<unsigned int> &block_ids,
                                    std::vector<PhysicalBlock *> &block_mem_lst) const;

        void *decode_buffer_ptr(unsigned int block_id) const;

        /**
         * @brief 每次解码之后对KV Cache的blocks进行释放
         */
        void reset_decode_lst() const;

        size_t get_avaliable_block_count() const;

    private:
        KVCacheManager();

        static std::unique_ptr<KVCacheManager> instance_;
        static std::once_flag onceFlag_;

        // 初始化函数，由 call_once 调用
        static void initInstance() {
            instance_.reset(new KVCacheManager()); // 使用 new 创建，并由 unique_ptr 管理
        }

        std::unique_ptr<MemoryTree> m_tree_; // 序列管理树
        PhysicalBlockManager *manager_; // 物理内存块管理器
        static size_t NUM_BLOCKS;
        static size_t BLOCK_SIZE; // 4KB per block
    };
}
