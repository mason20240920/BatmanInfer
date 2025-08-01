//
// Created by Mason on 2025/5/26.
//
#include <kv_cache_manager/bi_kv_cache_manager.hpp>

#include "cpu/bi_cpu_types.hpp"

namespace BatmanInfer {
    size_t KVCacheManager::NUM_BLOCKS;
    size_t KVCacheManager::BLOCK_SIZE;

    KVCacheManager &KVCacheManager::getInstance() {
        // 静态局部变量, 线程安全
        static KVCacheManager instance;
        return instance;
    }

    void KVCacheManager::initialize(const size_t num_blocks, const size_t block_size, size_t max_seq_len) {
        NUM_BLOCKS = num_blocks;
        BLOCK_SIZE = block_size;
        MemoryTree::initialize(max_seq_len, block_size);
    }


    KVCacheManager::KVCacheManager() {
        // 先申请manager_
        manager_ = create_block_manager(NUM_BLOCKS, BLOCK_SIZE);
        // root的block_id
        const auto root_alloc = acquire_block(manager_, 1);
        if (!root_alloc.block_ids)
            throw std::runtime_error("Failed to acquire root block");
        // 再申请内存树(默认开启的是起始符<s>
        m_tree_ = std::make_unique<MemoryTree>(root_alloc.block_ids[0], 0);
    }

    /**
             * @brief 获取当前的解码节点
             * @return 返回当前叶子的解码节点
             */
    std::unordered_map<unsigned int, unsigned int> KVCacheManager::get_decode_ids() const {
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
     BIErrCode KVCacheManager::alloc_decode_next(const unsigned int parent_id,
                                                const int require_block_count,
                                                const std::vector<unsigned int> &inp_ids,
                                                std::vector<unsigned int> &block_ids) const {
        const auto block_alloc = acquire_block(manager_, require_block_count);
        // std::vector<unsigned int> block_ids;
        block_ids.reserve(require_block_count);
        // 构建新的内存树
        for (int i = 1; i <= require_block_count; i++) {
            block_ids.emplace_back(block_alloc.block_ids[i - 1]);
        }
        bool ret = m_tree_->add_children(parent_id, block_ids, inp_ids);
        if (!ret) {
            return BIErrCode::BIGeneralError;
        }
        return BIErrCode::BISuccess;
    }

    /**
        * 根据输出的buffer值存储到对应的block_id里
        * @param source_buffer
        * @param block_id
        * @param is_k_cond
        * @param is_smooth_quant
        */
    void KVCacheManager::memcpy_decode_buffer(void *source_buffer,
                                              int block_id,
                                              int batch_idx,
                                              bool is_k_cond,
                                              bool is_smooth_quant) const {
        auto [id, buffer] = manager_->blocks[block_id];
        unsigned long k_block_size, v_block_size;
        char *k_src_buffer, *v_src_buffer;
        if (is_smooth_quant) {
            k_block_size = BLOCK_SIZE / 3 * 2;
            v_block_size = BLOCK_SIZE / 3;
            k_src_buffer = batch_idx * k_block_size + static_cast<char *>(source_buffer);
            v_src_buffer = batch_idx * v_block_size + static_cast<char *>(source_buffer);
        } else {
            k_block_size = v_block_size = BLOCK_SIZE / 2;
            k_src_buffer = batch_idx * k_block_size + static_cast<char *>(source_buffer);
            v_src_buffer = batch_idx * k_block_size + static_cast<char *>(source_buffer);
        }

        // char *src_buffer = batch_idx * BLOCK_SIZE / 2 + static_cast<char *>(source_buffer);
        if (is_k_cond) {
            memcpy(buffer, k_src_buffer, k_block_size);
            return;
        }
        memcpy(static_cast<char *>(buffer) + k_block_size, v_src_buffer, v_block_size);
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

    BIErrCode KVCacheManager::release_useless_decodes_id(const std::vector<unsigned int> &leaf_ids) const {
        std::vector<int> release_ids;
        const bool is_success = m_tree_->remove_decodes_lst(leaf_ids, release_ids);
        // BI_COMPUTE_ERROR_ON_MSG(!is_success, "Input is not leaf nodes"); // 判断是不是叶子节点
        if (!is_success) { return BIErrCode::BIKVCacheDelFailed; }    // 不进行错误码返回，不是叶子节点不进行后续删除操作
        release_blocks(manager_, release_ids.data(), release_ids.size());
        return BIErrCode::BISuccess;
    }

    BIErrCode KVCacheManager::release_end_symbol(const std::vector<unsigned int> &eos_ids) const {
        std::vector<int>release_ids;
        const bool is_success = m_tree_->remove_eos_lst(eos_ids, release_ids);
        // BI_COMPUTE_ERROR_ON_MSG(!is_success, "Input is not leaf nodes"); // 判断是不是叶子节点
        if (!is_success) { return BIErrCode::BIKVCacheDelFailed; }    // 不进行错误码返回，不是叶子节点不进行后续删除操作
        release_blocks(manager_, release_ids.data(), release_ids.size());
        return BIErrCode::BISuccess;
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

    void *KVCacheManager::decode_buffer_ptr(unsigned int block_id) const {
        if (!m_tree_->find_node(block_id))
            throw std::runtime_error("Failed to get decode buffer");
        return manager_->blocks[block_id].buffer;
    }

    void KVCacheManager::reset_decode_lst() const {
        m_tree_->reset();
        reset_block_manager(manager_);
    }

    size_t KVCacheManager::get_avaliable_block_count() const {
        return manager_->free_blocks;
    }

}
