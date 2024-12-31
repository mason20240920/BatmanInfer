//
// Created by Mason on 2024/12/27.
//

#ifndef BATMANINFER_BI_MEMORY_GROUP_HPP
#define BATMANINFER_BI_MEMORY_GROUP_HPP

#include <data/core/bi_error.h>
#include <data/core/utils/misc/macros.hpp>
#include <runtime/bi_allocator.hpp>
#include <runtime/bi_i_memory_group.hpp>
#include <runtime/bi_i_memory_manager.hpp>
#include <runtime/bi_i_memory_pool.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>

namespace BatmanInfer {
    // 前向声明
    class BIIMemory;

    class BIMemoryGroup final : public BIIMemoryGroup {
    public:
        BIMemoryGroup(std::shared_ptr<BIIMemoryManager> = nullptr) noexcept;

        ~BIMemoryGroup() = default;

        BIMemoryGroup(const BIMemoryGroup &) = delete;

        BIMemoryGroup &operator=(const BIMemoryGroup &) = delete;

        BIMemoryGroup(BIMemoryGroup &&) = default;

        BIMemoryGroup &operator=(BIMemoryGroup &&) = default;

        void manage(BatmanInfer::BIIMemoryManageable *obj) override;
        void finalize_memory(BatmanInfer::BIIMemoryManageable *obj, BatmanInfer::BIIMemory &obj_memory, size_t size, size_t alignment) override;
        void acquire() override;
        void release() override;
        BIMemoryMappings & mappings() override;


    private:
        /**
         * @brief 内存组使用的内存管理器
         */
        std::shared_ptr<BIIMemoryManager> _memory_manager;

        /**
         * @brief 与该组调度相关的内存池
         */
        BIIMemoryPool *_pool;

        /**
         * @brief 内存组的内存映射
         */
        BIMemoryMappings  _mappings;

        /**
         * @brief 是否内存管理器会在释放时候自动清理
         */
        bool _auto_clear;
    };

    inline BIMemoryGroup::BIMemoryGroup(std::shared_ptr<BIIMemoryManager> memory_manager) noexcept :
    _memory_manager(memory_manager), _pool(nullptr), _mappings(), _auto_clear(false){

    }

    inline void BIMemoryGroup::manage(BatmanInfer::BIIMemoryManageable *obj) {
        if (_memory_manager && (obj != nullptr)) {
            BI_COMPUTE_ERROR_ON(!_memory_manager->lifetime_manager());


        }
    }

}

#endif //BATMANINFER_BI_MEMORY_GROUP_HPP
