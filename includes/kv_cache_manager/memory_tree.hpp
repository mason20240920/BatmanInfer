//
// Created by Mason on 2025/5/26.
//

#pragma once
#include <unordered_map>
#include <unordered_set>
#include <vector>
/**
 * @brief 内存树的管理模块的block_id与申请物理内存的block_id一一映射，所以可以直接申请Block_ids数量的memory_pool
 * 进行动态调整，同时不进行删除，只进行更新
 */
namespace BatmanInfer {
    /**
     * 内存树结构
     */
    class MemoryTree {
    public:
        struct MemoryNode {
            unsigned int block_id; // 内存块的id

            unsigned int decode_id; // 解码序列的id

            std::unordered_map<unsigned int, MemoryNode *> children; // 使用哈希表存储子节点，提高查询效率

            MemoryNode *parent; // 获取父节点

            explicit MemoryNode(const unsigned int block_id, const unsigned int decode_id) : block_id(block_id),
                decode_id(decode_id), parent(nullptr) {
            }
        };

    private:
        // 预留节点容量, 减少refresh开销
        static size_t DEFAULT_CAPACITY;
        static size_t MAX_SEQ_LEN;

        // 使用节点池管理内存, 提高内存局部性
        std::vector<std::unique_ptr<MemoryNode> > memory_pool;

        // ID到节点的快速映射
        std::unordered_map<unsigned int, MemoryNode *> node_map;

        // 叶子节点两个key: 一个是block_id, 一个是decode_id
        std::unordered_map<unsigned int, unsigned int> leaf_nodes; // 维护叶子节点表

        MemoryNode *root_node;

    public:
        // 构造函数, 预分配内存
        explicit MemoryTree(const unsigned int root_id, unsigned int decode_id);

        static void initialize(size_t max_seq, size_t max_capacity);

        /**
         * 获取根节点
         * @return 获取根节点
         */
        [[nodiscard]] MemoryNode *get_root() const;

        /**
         * @brief 创建叶子节点
         * @param id 创建节点的id
         * @return
         */
        MemoryNode *create_node(unsigned int id, unsigned int decode_id);

        /**
         * @brief 增加子节点
         * @param parent_id 父节点的创建
         * @param child_ids 子节点的id数组
         * @return 返回子节点
         */
        bool add_children(unsigned int parent_id, const std::vector<unsigned int> &child_ids,
                          const std::vector<unsigned int> &decode_ids);

        /**
         * 优化的子节点添加，避免重复查找
         */
        MemoryNode *add_child(unsigned int parent_id, unsigned int child_id, unsigned int decode_id);

        /**
         * @brief 根据id查找节点, Q(1)的时间复杂度
         * @param id
         * @return
         */
        [[nodiscard]] MemoryNode *find_node(const unsigned int id) const;

        /**
         * @brief 获取目前节点的数量
         * @return
         */
        [[nodiscard]] size_t size() const;

        /**
         * 清空树
         */
        void clear();

        /**
         * @brief 重置Reset树节点
         */
        void reset();


        /**
         * 获取节点深度
         */
        [[nodiscard]] int get_depth(const unsigned int id) const;

        // 获取叶子节点现在变得非常高效
        [[nodiscard]] std::unordered_map<unsigned int, unsigned int> get_leaf_ids() const;

        /**
         * @brief 根据叶子节点获取一个序列的内存block_ids
         * @param leaf_id
         * @param block_ids: 容器的ids
         * @return
         */
        void get_block_ids_seq(const unsigned int leaf_id, std::vector<unsigned int> &block_ids) const;

        /**
         * @brief 根据叶子节点对树结构进行更新
         * @param leaf_id
         * @param block_ids 需要进行内存管理器释放的block_ids
         */
        void remove_decode_lst(const unsigned int leaf_id, std::vector<int> &block_ids);

        /**
         * @brief 批量删除叶子节点
         * @param leaf_ids
         * @param block_ids
         */
        bool remove_decodes_lst(const std::vector<unsigned int>& leaf_ids, std::vector<int> &block_ids);

        /**
         * @brief 移除截止符
         * @param leaf_ids
         * @param block_ids
         * @return
         */
        bool remove_eos_lst(const std::vector<unsigned int>& leaf_ids, std::vector<int> &block_ids);

        /**
         * @desc 判断是不是叶子节点
         * @param node_ids
         * @return
         */
        [[nodiscard]] bool is_leaf_nodes(const std::vector<unsigned int> &node_ids) const;

    private:
        /**
         * @brief 对节点删除的拓扑排序
         * @param nodes_to_remove
         * @param sorted_result
         */
        void topological_sort_for_removal(const std::unordered_set<unsigned int>& nodes_to_remove,
                                          std::vector<unsigned int>& sorted_result) const;

        int calculate_depth(const MemoryNode* node) const;

    };
}
