//
// Created by Mason on 2025/1/12.
//

#ifndef BATMANINFER_BI_MEMORY_HELPERS_HPP
#define BATMANINFER_BI_MEMORY_HELPERS_HPP

#include <data/core/experimental/types.hpp>
#include <data/core/bi_i_tensor_pack.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <runtime/bi_memory_group.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace BatmanInfer {
    inline int offset_int_vec(int offset) {
        return ACL_INT_VEC + offset;
    }

    template<typename TensorType>
    struct WorkspaceDataElement {
        int slot{-1};
        experimental::MemoryLifetime lifetime{experimental::MemoryLifetime::Temporary};
        std::unique_ptr<TensorType> tensor{nullptr};
    };

    template<typename TensorType>
    using WorkspaceData = std::vector<WorkspaceDataElement<TensorType> >;

    template<typename TensorType>
    WorkspaceData<TensorType>
    manage_workspace(const experimental::BIMemoryRequirements &mem_reqs, BIMemoryGroup &mgroup,
                     BIITensorPack &run_pack) {
        BIITensorPack dummy_pack = BIITensorPack();
        return manage_workspace<TensorType>(mem_reqs, mgroup, run_pack, dummy_pack);
    }

    /**
     * 配置转置之类的张量内存信息
     * @tparam TensorType
     * @param mem_reqs
     * @param mgroup
     * @param run_pack
     * @param prep_pack
     * @param allocate_now
     * @return
     */
    template<typename TensorType>
    WorkspaceData<TensorType> manage_workspace(const experimental::BIMemoryRequirements &mem_reqs,
                                               BIMemoryGroup &mgroup,
                                               BIITensorPack &run_pack,
                                               BIITensorPack &prep_pack,
                                               bool allocate_now = true) {
        WorkspaceData<TensorType> workspace_memory;
        for (const auto &req: mem_reqs) {
            if (req.size == 0) {
                continue;
            }

            const auto aux_info = BITensorInfo{BITensorShape(req.size), 1, BIDataType::U8}; // 创建一个张量信息
            workspace_memory.emplace_back(
                WorkspaceDataElement<TensorType>{req.slot, req.lifetime, std::make_unique<TensorType>()});

            auto aux_tensor = workspace_memory.back().tensor.get(); // 获取最后的内存的tensor
            BI_COMPUTE_ERROR_ON_NULLPTR(aux_tensor);
            aux_tensor->allocator()->init(aux_info, req.alignment); // 根据aux_info进行初始化, req.alignment是初始化信息

            if (req.lifetime == experimental::MemoryLifetime::Temporary) {
                mgroup.manage(aux_tensor);
            } else {
                prep_pack.add_tensor(req.slot, aux_tensor);
            }
            run_pack.add_tensor(req.slot, aux_tensor);
        }

        for (auto &mem: workspace_memory) {
            if (allocate_now || mem.lifetime == experimental::MemoryLifetime::Temporary) {
                auto tensor = mem.tensor.get();
                tensor->allocator()->allocate();
            }
        }

        return workspace_memory;
    }

    template<typename TensorType>
    void release_prepare_tensors(WorkspaceData<TensorType> &workspace, BIITensorPack &prep_pack) {
        workspace.erase(std::remove_if(workspace.begin(), workspace.end(),
                                       [&prep_pack](auto &wk) {
                                           const bool to_erase = wk.lifetime == experimental::MemoryLifetime::Prepare;
                                           if (to_erase) {
                                               prep_pack.remove_tensor(wk.slot);
                                           }
                                           return to_erase;
                                       }),
                        workspace.end());
    }

    /** Allocate all tensors with Persistent or Prepare lifetime if not already allocated */
    template<typename TensorType>
    void allocate_tensors(const experimental::BIMemoryRequirements &mem_reqs, WorkspaceData<TensorType> &workspace) {
        for (auto &ws: workspace) {
            const int slot = ws.slot;
            for (auto &m: mem_reqs) {
                if (m.slot == slot && m.lifetime != experimental::MemoryLifetime::Temporary) {
                    auto tensor = ws.tensor.get();
                    if (!tensor->allocator()->is_allocated()) {
                        tensor->allocator()->allocate();
                    }
                    break;
                }
            }
        }
    }

    /** Utility function to release tensors with lifetime marked as Prepare */
    template<typename TensorType>
    void release_temporaries(const experimental::BIMemoryRequirements &mem_reqs, WorkspaceData<TensorType> &workspace) {
        for (auto &ws: workspace) {
            const int slot = ws.slot;
            for (auto &m: mem_reqs) {
                if (m.slot == slot && m.lifetime == experimental::MemoryLifetime::Prepare) {
                    auto tensor = ws.tensor.get();
                    tensor->allocator()->free();
                    break;
                }
            }
        }
    }

    /**
     * @brief 根据新的内存需求来重新分配张量的工作内存
     * @tparam TensorType
     * @param mem_reqs 根据slot的id适配
     * @param workspace 生命周期要求前后一致; 只有size和alignment受影响
     */
    template<typename TensorType>
    void reallocate_tensors(const experimental::BIMemoryRequirements &mem_reqs, WorkspaceData<TensorType> &workspace) {
        for (auto &ws: workspace) {
            auto tensor = ws.tensor.get();
            const int slot = ws.slot; // 获取内存识别id
            for (auto &m: mem_reqs) {
                if (m.slot == slot) {
                    BI_COMPUTE_ERROR_ON(ws.lifetime != m.lifetime);
                    size_t current_size = tensor->info()->total_size();
                    if (!tensor->allocator()->is_allocated() || current_size < m.size || tensor->allocator()->
                        alignment() != m.alignment) {
                        if (tensor->allocator()->is_allocated()) // 如果张量已经初始化就释放掉
                            tensor->allocator()->free();
                        const BIITensorInfo &info = tensor->info()->set_tensor_shape(BITensorShape(m.size));
                        tensor->allocator()->init(info, m.alignment);
                        tensor->allocator()->allocate();
                    }
                    break;
                }
            }
        }
    }
}

#endif //BATMANINFER_BI_MEMORY_HELPERS_HPP
