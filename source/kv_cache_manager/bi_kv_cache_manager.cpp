//
// Created by Mason on 2025/5/26.
//
#include <kv_cache_manager/bi_kv_cache_manager.hpp>

namespace BatmanInfer {
    KVCacheManager &KVCacheManager::getInstance() {
        // 静态局部变量, 线程安全
        static KVCacheManager instance;
        return instance;
    }

    KVCacheManager::KVCacheManager() {
        constexpr size_t NUM_BLOCKS = 1024;
        constexpr size_t BLOCK_SIZE = 4096; // 4KB per block
        // 先申请manager_
        manager_ = create_block_manager(NUM_BLOCKS, BLOCK_SIZE);
        // root的block_id
        const auto root_alloc = acquire_block(manager_, 1);
        if (!root_alloc.block_ids)
            throw std::runtime_error("Failed to acquire root block");
        // 再申请内存树
        m_tree_ = std::make_unique<MemoryTree>(root_alloc.block_ids[0]);
    }

    /**
             * @brief 获取当前的解码节点
             * @return 返回当前叶子的解码节点
             */
    std::vector<unsigned int> KVCacheManager::get_decode_ids() const {
        return m_tree_->get_leaf_ids();
    }

    unsigned int KVCacheManager::root_id() const {
        return m_tree_->get_root()->block_id;
    }

    /**
     * 根据需求动态请求新的KV Cache内存
     * @param parent_id
     * @param require_block_count
     */
    void KVCacheManager::alloc_decode_next(const unsigned int parent_id, const int require_block_count) const {
        const auto block_alloc = acquire_block(manager_, require_block_count);
        std::vector<unsigned int> decode_ids;
        decode_ids.reserve(require_block_count);
        // 构建新的内存树
        for (int i = 0; i < require_block_count; i++) {
            decode_ids.emplace_back(block_alloc.block_ids[i]);
        }
        m_tree_->add_children(parent_id, decode_ids);
    }

    /**
     * 根据输出的buffer值存储到对应的block_id里
     * @param source_buffer
     * @param block_id
     * @param block_size
     */
    void KVCacheManager::memcpy_decode_buffer(const void *source_buffer, const int block_id,
                                              const int block_size) const {
        auto physic_block = manager_->blocks[block_id];
        memcpy(physic_block.buffer, source_buffer, block_size);
    }

    /**
     * 释放空的解码序列
     * @param leaf_id: 叶子节点id
     */
    void KVCacheManager::release_useless_decode_id(const unsigned int leaf_id) const {
        std::vector<int> release_ids;
        m_tree_->remove_decode_lst(leaf_id, release_ids);
        release_blocks(manager_, release_ids.data(), release_ids.size());
    }

    /**
     * 根据序列节点号，获取解码出来的解码K, V Cache结果
     * @param leaf_id
     * @param block_ids
     */
    void KVCacheManager::decode_sequence_lst(const unsigned int leaf_id, std::vector<unsigned int> &block_ids) const {
        m_tree_->get_block_ids_seq(leaf_id, block_ids);
        std::reverse(block_ids.begin(), block_ids.end());
    }

    /**
     * 解码序列的内存Block数组
     * @param block_ids
     * @param block_mem_lst
     */
    void KVCacheManager::decode_sequence_blocks(const std::vector<unsigned int> &block_ids,
                                                std::vector<PhysicalBlock *> &block_mem_lst) const {
        for (const unsigned int block_id: block_ids) {
            block_mem_lst.emplace_back(&manager_->blocks[block_id]);
        }
    }
}
