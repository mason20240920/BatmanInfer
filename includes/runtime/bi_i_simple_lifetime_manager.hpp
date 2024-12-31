//
// Created by Mason on 2024/12/27.
//

#ifndef BATMANINFER_BI_I_SIMPLE_LIFETIME_MANAGER_HPP
#define BATMANINFER_BI_I_SIMPLE_LIFETIME_MANAGER_HPP

#include <runtime/bi_i_lifetime_manager.hpp>
#include <runtime/bi_i_memory_pool.hpp>
#include <runtime/bi_types.hpp>

#include <set>
#include <list>

namespace BatmanInfer {
    /**
     * @brief 简单生命周期管理器的抽象类接口
     */
    class BIISimpleLifetimeManager: public BIILifetimeManager {
    public:
        BIISimpleLifetimeManager();
        /** Prevent instances of this class to be copy constructed */
        BIISimpleLifetimeManager(const BIISimpleLifetimeManager &) = delete;
        /** Prevent instances of this class to be copied */
        BIISimpleLifetimeManager &operator=(const BIISimpleLifetimeManager &) = delete;

        BIISimpleLifetimeManager(BIISimpleLifetimeManager &&) = default;

        BIISimpleLifetimeManager &operator=(BIISimpleLifetimeManager &&) = default;

        // 继承方法
        void register_group(BatmanInfer::BIIMemoryGroup *group) override;
        bool release_group(BatmanInfer::BIIMemoryGroup *group) override;
        void start_lifetime(void *obj) override;
        void end_life_time(void *obj, BatmanInfer::BIIMemory &obj_memory, size_t size, size_t alignment) override;
        [[nodiscard]] bool are_all_finalized() const override;

    protected:
        /**
         * @brief 更新内存块和映射
         */
        virtual void update_blobs_and_mappings() = 0;

    protected:
        /**
         * @brief 元素结构体
         */
        struct Element {
            explicit Element(void *id_ = nullptr,
                    BIIMemory *handle_ = nullptr,
                    size_t size_ = 0,
                    size_t alignment_ = 0,
                    bool status_ = false) :
                    id(id_),
                    handle(handle_),
                    size(size_),
                    alignment(alignment_),
                    status(status_) {

            }

            /**
             * @brief 元素id
             */
            void *id;
            /**
             * @brief 元素内存管理
             */
            BIIMemory *handle;
            /**
             * @brief 元素的大小
             */
            size_t size;
            /**
             * @brief 内存对齐需求
             */
            size_t alignment;
            /**
             * @brief 生命周期状态
             */
            bool status;
        };

        /**
         * @brief 内存块结构
         */
        struct Blob {
            void *id;
            size_t max_size;
            size_t max_alignment;
            std::set<void *> bound_elements;
        };

        /**
         * @brief 激活的内存组
         */
        BIIMemoryGroup *_active_group;

        /**
         * @brief 包含激活元素的字典，
         * 存储对象的地址
         */
        std::map<void *, Element> _active_elements;

        /**
         * @brief 释放的内存块
         */
        std::list<Blob> _free_blobs;

        /**
         * @brief 占用的内存块
         */
        std::list<Blob> _occupied_blobs;

        /**
         * @brief 一张包含最终内存组的字典。
         */
        std::map<BIIMemoryGroup *,
                 std::map<void *, Element>> _finalized_groups;
    };
}

#endif //BATMANINFER_BI_I_SIMPLE_LIFETIME_MANAGER_HPP
