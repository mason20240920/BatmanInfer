//
// Created by Mason on 2025/5/26.
//
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <kv_cache_manager/memory_tree.hpp>

#include "data/core/bi_error.h"
/**
 * @brief 内存树的管理模块的block_id与申请物理内存的block_id一一映射，所以可以直接申请Block_ids数量的memory_pool
 * 进行动态调整，同时不进行删除，只进行更新
 */
namespace BatmanInfer {
    constexpr size_t MAX_SEQ_LEN = 16;
    /**
     * 内存树结构
     */
    MemoryTree::MemoryTree(const unsigned int root_id, const unsigned int decode_id) {
        // 对内存池，全量进行初始化
        for (int i = 0; i < DEFAULT_CAPACITY; i++) {
            memory_pool.emplace_back(new MemoryNode(i, 0));
        }
        node_map.reserve(DEFAULT_CAPACITY);
        root_node = create_node(root_id, decode_id);
    }

    /**
     * 获取根节点
     * @return 获取根节点
     */
    MemoryTree::MemoryNode *MemoryTree::get_root() const {
        return root_node;
    }

    /**
     * @brief 创建叶子节点
     * @param id 创建节点的id
     * @param decode_id
     * @return
     */
    MemoryTree::MemoryNode *MemoryTree::create_node(unsigned int id, unsigned int decode_id) {
        // 先查找节点是否已经存在
        auto it = node_map.find(id);
        if (it != node_map.end())
            return it->second;

        // 创建新节点, 加入节点内存池
        MemoryNode *new_node = memory_pool.at(id).get();
        new_node->decode_id = decode_id;
        node_map.emplace(id, new_node);

        // 新节点一开始是叶子节点
        leaf_nodes.insert({id, decode_id});

        return new_node;
    }

    /**
     * @brief 增加子节点
     * @param parent_id 父节点的创建
     * @param child_ids 子节点的id数组
     * @param decode_ids
     * @return 返回子节点
     */
    void MemoryTree::add_children(const unsigned int parent_id,
                                  const std::vector<unsigned int> &child_ids,
                                  const std::vector<unsigned int> &decode_ids) {
        MemoryNode *parent = find_node(parent_id);
        if (!parent)
            BI_COMPUTE_ERROR("Find Parent Node failure");

        // 预留空间，避免Rehash
        parent->children.reserve(parent->children.size() + child_ids.size());

        // 父节点添加子节点后不再是叶子节点
        if (parent->children.empty()) {
            leaf_nodes.erase(parent_id);
        }

        for (int i = 0; i < child_ids.size(); i++) {
            unsigned int child_id = child_ids[i];
            unsigned int decode_id = decode_ids[i + 1];
            MemoryNode *child = find_node(child_id);
            if (!child) {
                child = create_node(child_id, decode_id);
                child->parent = parent;
            }

            parent->children[child_id] = child;
        }
    }

    /**
     * 优化的子节点添加，避免重复查找
     */
    MemoryTree::MemoryNode *MemoryTree::add_child(const unsigned int parent_id,
                                                  const unsigned int child_id,
                                                  const unsigned int decode_id) {
        MemoryNode *parent = find_node(parent_id);
        if (!parent) {
            parent = create_node(parent_id, 0);
        }

        MemoryNode *child = find_node(child_id);
        if (!child) {
            child = create_node(child_id, decode_id);
            child->parent = parent;
        }

        // 父节点添加子节点后不再是叶子节点
        if (parent->children.empty()) {
            leaf_nodes.erase(parent_id);
        }

        // 使用哈希表存储子节点引用，查找时间为O(1)
        parent->children[child_id] = child;
        return child;
    }

    /**
     * @brief 根据id查找节点, Q(1)的时间复杂度
     * @param id
     * @return
     */
    MemoryTree::MemoryNode *MemoryTree::find_node(const unsigned int id) const {
        const auto it = node_map.find(id);
        return (it != node_map.end()) ? it->second : nullptr;
    }

    /**
     * @brief 获取目前节点的数量
     * @return
     */
    size_t MemoryTree::size() const {
        return node_map.size();
    }

    /**
     * 清空树
     */
    void MemoryTree::clear() {
        node_map.clear();
        root_node = nullptr;
    }

    /**
     * 获取节点深度
     */
    int MemoryTree::get_depth(const unsigned int id) const {
        if (const MemoryNode *node = find_node(id); !node) return -1;

        int depth = 0;
        for (const auto &[_, parent_node]: node_map) {
            for (const auto &[child_id, _]: parent_node->children) {
                if (child_id == id) {
                    int parent_depth = get_depth(parent_node->block_id);
                    if (parent_depth >= 0) {
                        depth = std::max(depth, parent_depth + 1);
                    }
                }
            }
        }
        return depth;
    }


    std::unordered_map<unsigned int, unsigned int> MemoryTree::get_leaf_ids() const {
        return {leaf_nodes.begin(), leaf_nodes.end()};
    }

    /**
     * @brief 根据叶子节点获取一个序列的内存block_ids
     * @param leaf_id
     * @param block_ids: 容器的ids
     * @return
     */
    void MemoryTree::get_block_ids_seq(const unsigned int leaf_id, std::vector<unsigned int> &block_ids) const {
        if (block_ids.empty())
            block_ids.reserve(MAX_SEQ_LEN); // 最大序列长度

        // 验证节点存在
        MemoryNode *node = find_node(leaf_id);
        if (!node)
            return;

        // 添加当前节点
        block_ids.emplace_back(leaf_id);

        // 查看是否是根节点
        if (root_node->block_id == leaf_id)
            return;

        get_block_ids_seq(node->parent->block_id, block_ids);
    }

    /**
     * @brief 根据叶子节点对树结构进行更新
     * @param leaf_id
     * @param block_ids 需要进行内存管理器释放的block_ids
     */
    void MemoryTree::remove_decode_lst(const unsigned int leaf_id, std::vector<int> &block_ids) {
        // 验证节点存在
        const MemoryNode *node = find_node(leaf_id);
        if (!node)
            return;

        // 查找父节点
        const auto parent_node = node->parent;
        if (!parent_node)
            return;

        parent_node->children.erase(leaf_id);
        block_ids.emplace_back(leaf_id);
        leaf_nodes.erase(leaf_id);
        node_map.erase(leaf_id);
        // 如果父节点也没有出度，则直接对父节点进行更新
        if (parent_node->children.empty())
            remove_decode_lst(parent_node->block_id, block_ids);
    }
};
