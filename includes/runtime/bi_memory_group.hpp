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
#include <runtime/bi_types.hpp>

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

            // 将注册延迟到第一个被管理的对象。
            _memory_manager->lifetime_manager()->register_group(this);

            // 将内存组与张量联系在一起
            obj->associate_memory_group(this);

            // 开始对象的生命周期
            _memory_manager->lifetime_manager()->start_lifetime(obj);
        }
    }

    inline void BIMemoryGroup::finalize_memory(BatmanInfer::BIIMemoryManageable *obj,
                                               BatmanInfer::BIIMemory &obj_memory, size_t size, size_t alignment) {
        if (_memory_manager) {
            BI_COMPUTE_ERROR_ON(!_memory_manager->lifetime_manager());
            _memory_manager->lifetime_manager()->end_life_time(obj, obj_memory, size, alignment);
        }
    }

    inline void BIMemoryGroup::acquire() {
        if (!_mappings.empty()) {
            BI_COMPUTE_ERROR_ON(!_memory_manager->pool_manager());
            // 如果调用者尚未填充底层内存管理器，
            // 在此进行填充。同时设置标志以在释放时自动清除内存管理器。
            // 这是在使用用户未设置的默认内存管理器时所需的。
            if (_memory_manager->pool_manager()->num_pools() == 0) {
                BIAllocator allocator{};
                _memory_manager->populate(allocator, 1);
                _auto_clear = true;
            }

            _pool = _memory_manager->pool_manager()->lock_pool();
            _pool->acquire(_mappings);
        }
    }

    inline void BIMemoryGroup::release() {
        if (_pool != nullptr) {
            BI_COMPUTE_ERROR_ON(!_memory_manager->pool_manager());
            BI_COMPUTE_ERROR_ON(_mappings.empty());
            _pool->release(_mappings);
            _memory_manager->pool_manager()->unlock_pool(_pool);
            _pool = nullptr;

            if (_auto_clear) {
                _memory_manager->clear();
                _auto_clear = false;
            }
        }
    }

    inline BIMemoryMappings &BIMemoryGroup::mappings() {
        return _mappings;
    }

}

#endif //BATMANINFER_BI_MEMORY_GROUP_HPP
