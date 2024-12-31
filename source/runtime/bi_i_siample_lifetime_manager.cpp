//
// Created by Mason on 2024/12/27.
//

#include <runtime/bi_i_simple_lifetime_manager.hpp>
#include <runtime/bi_i_memory_group.hpp>
#include <data/core/bi_error.h>

namespace BatmanInfer {
    BIISimpleLifetimeManager::BIISimpleLifetimeManager() :
    _active_group(nullptr),_active_elements(),_free_blobs(),_occupied_blobs(),_finalized_groups(){

    }

    void BIISimpleLifetimeManager::register_group(BatmanInfer::BIIMemoryGroup *group) {
        // 如果激活的内存组是空
        if (_active_group == nullptr) {
            BI_COMPUTE_ERROR_ON(group == nullptr);
            // 就把这个注册的内存组放到激活的组里面
            _active_group = group;
        }
    }

    bool BIISimpleLifetimeManager::release_group(BatmanInfer::BIIMemoryGroup *group) {
        if (group == nullptr)
            return false;
        const bool status = bool(_finalized_groups.erase(group));
        if (status)
            group->mappings().clear();
        return status;
    }

    void BIISimpleLifetimeManager::start_lifetime(void *obj) {
        BI_COMPUTE_ERROR_ON(obj == nullptr);
        BI_COMPUTE_ERROR_ON_MSG(_active_elements.find(obj) != std::end(_active_elements),
                                 "Memory object is already registered!");

        // Check if there is a free blob
        if (_free_blobs.empty())
            _occupied_blobs.emplace_front(Blob{obj, 0, 0, {obj}});
        else {
            _occupied_blobs.splice(std::begin(_occupied_blobs), _free_blobs, std::begin(_free_blobs));
            _occupied_blobs.front().id = obj;
        }

        // 将对象插入组中，并将其最终状态标记为假。
        _active_elements.insert(std::make_pair(obj, obj));
    }

    void BIISimpleLifetimeManager::end_life_time(void *obj,
                                                 BatmanInfer::BIIMemory &obj_memory,
                                                 size_t size,
                                                 size_t alignment) {
        BI_COMPUTE_ERROR_ON(obj == nullptr);

        // 查找到元素
        auto active_object_it = _active_elements.find(obj);
        BI_COMPUTE_ERROR_ON(active_object_it == std::end(_active_elements));

        // Update object fields and mark object as complete
        Element &el  = active_object_it->second;
        el.handle    = &obj_memory;
        el.size      = size;
        el.alignment = alignment;
        el.status    = true;

        // 在被占用的列表中查找对象
        auto occupied_blob_it = std::find_if(std::begin(_occupied_blobs), std::end(_occupied_blobs),
                                             [&obj](const Blob &b) { return obj == b.id; });
        BI_COMPUTE_ERROR_ON(occupied_blob_it == std::end(_occupied_blobs));


        // 更新占用返回free
        occupied_blob_it->bound_elements.insert(obj);
        occupied_blob_it->max_size      = std::max(occupied_blob_it->max_size, size);
        occupied_blob_it->max_alignment = std::max(occupied_blob_it->max_alignment, alignment);
        occupied_blob_it->id            = nullptr;
        _free_blobs.splice(std::begin(_free_blobs), _occupied_blobs, occupied_blob_it);

        // 检查是否所有对象都已完成并重置活动组
        if (are_all_finalized()) {
            BI_COMPUTE_ERROR_ON(!_occupied_blobs.empty());

            // 更新内存块和组映射
            update_blobs_and_mappings();

            // 更新已完成的内存组
            _finalized_groups[_active_group].insert(std::begin(_active_elements),
                                                    std::end(_active_elements));

            // 重置状态
            _active_elements.clear();
            _active_group = nullptr;
            _free_blobs.clear();
        }
    }

    bool BIISimpleLifetimeManager::are_all_finalized() const {
        return !std::any_of(std::begin(_active_elements), std::end(_active_elements),
                            [](const std::pair<void *, Element> &e) { return !e.second.status; });
    }
}