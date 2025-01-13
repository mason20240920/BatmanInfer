//
// Created by Mason on 2025/1/12.
//

#ifndef BATMANINFER_BI_CPU_AUX_TENSOR_HANDLER_HPP
#define BATMANINFER_BI_CPU_AUX_TENSOR_HANDLER_HPP

#include <data/core/bi_i_tensor_pack.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <runtime/bi_tensor.hpp>

#include <common/utils/bi_log.hpp>
#include <support/bi_cast.hpp>
#include <data/core/experimental/types.hpp>


namespace BatmanInfer::cpu {
    /** 用于管理张量（Tensor）内存的工具类
     *
     * @note 重要：这个类不仅仅是一个指针，它真正“拥有”张量的内存（会负责分配和释放内存）。
     *
     * @note 关于 bypass_alloc 和 bypass_import 标志
     * - `bypass_alloc`：跳过新张量的内存分配，用于避免不必要的内存操作。
     * - `bypass_import`：跳过从已存在的张量导入内存，用于节省运行时的开销。
     *
     * 这两个标志的作用是避免在某些情况下（比如运行时不需要用到张量）浪费资源。
     * 但如果用错了，可能会导致性能问题（没跳过却不需要）或内存错误（跳过了但需要）。
     *
     * 使用时注意：
     * 1. **作用域问题**：这个类的对象在销毁时会释放它管理的内存。如果对象超出了作用域（比如函数结束了），
     *    那么它管理的张量内存也会被释放。所以要确保在使用张量时，这个对象一直存在。
     *
     * 2. **标志位的设置**：`bypass_alloc` 和 `bypass_import` 的值需要根据实际情况设置。
     *    如果你确定张量不会被用到，就设置为 `true`（跳过）；如果会用到，就设置为 `false`（不跳过）。
     *
     * **典型用法示例**：
     *
     *      bool use_aux_tensor = 判断是否需要辅助张量的条件;
     *
     *      // 创建处理器对象，设置标志位
     *      CpuAuxTensorHandler aux_handler(..., !use_aux_tensor || bypass_alloc || bypass_import);
     *
     *      if (use_aux_tensor)
     *      {
     *          tensor_pack.add_tensor(aux_handler.get()); // 把张量添加到张量包
     *      }
     *      op.run(tensor_pack); // 运行操作
     */

    class CpuAuxTensorHandler {
    public:
        /** Create a temporary tensor handle, by either important an existing tensor from a tensor pack, or allocating a
         *  new one.
         *
         * @param[in]     slot_id       Slot id of the tensor to be retrieved in the tensor pack
         *                              If no such tensor exists in the tensor pack, a new tensor will be allocated.
         * @param[in]     info          Tensor info containing requested size of the new tensor.
         *                              If requested size is larger than the tensor retrieved from the tensor pack,
         *                              a new tensor will be allocated.
         * @param[in,out] pack          Tensor pack to retrieve the old tensor. When @p pack_inject is true, the new
         *                              tensor will also be added here.
         * @param[in]     pack_inject   In case of a newly allocated tensor, whether to add this tensor back to the
         *                              @p pack
         * @param[in]     bypass_alloc  Bypass allocation in case of a new tensor
         *                              This is to prevent unnecessary memory operations when the handler object is not
         *                              used
         * @param[in]     bypass_import Bypass importation in case of a retrieved tensor
         *                                  This is to prevent unnecessary memory operations when the handler object is not
         *                                  used
         */
        CpuAuxTensorHandler(int slot_id,
                            BITensorInfo &info,
                            BIITensorPack &pack,
                            bool pack_inject = false,
                            bool bypass_alloc = false,
                            bool bypass_import = false)
                : _tensor() {
            if (info.total_size() == 0) {
                return;
            }
            _tensor.allocator()->soft_init(info);

            auto *packed_tensor = utils::cast::polymorphic_downcast<BIITensor *>(pack.get_tensor(slot_id));
            if ((packed_tensor == nullptr) || (info.total_size() > packed_tensor->info()->total_size())) {
                if (!bypass_alloc) {
                    _tensor.allocator()->allocate();
                    BI_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL("Allocating auxiliary tensor");
                }

                if (pack_inject) {
                    pack.add_tensor(slot_id, &_tensor);
                    _injected_tensor_pack = &pack;
                    _injected_slot_id = slot_id;
                }
            } else {
                if (!bypass_import) {
                    _tensor.allocator()->import_memory(packed_tensor->buffer());
                }
            }
        }

        /** Create a temporary handle to the original tensor with a new @ref TensorInfo
         * This is useful if we want to change a tensor's tensor info at run time without modifying the original tensor
         *
         * @param[in] info          New tensor info to "assign" to @p tensor
         * @param[in] tensor        Tensor to be assigned a new @ref TensorInfo
         * @param[in] bypass_import Bypass importing @p tensor's memory into the handler.
         *                          This is to prevent unnecessary memory operations when the handler object is not used
         */
        CpuAuxTensorHandler(BITensorInfo &info, const BIITensor &tensor, bool bypass_import = false) : _tensor() {
            _tensor.allocator()->soft_init(info);
            if (!bypass_import) {
                BI_COMPUTE_ERROR_ON(tensor.info() == nullptr);
                if (info.total_size() <= tensor.info()->total_size()) {
                    _tensor.allocator()->import_memory(tensor.buffer());
                }
            }
        }

        CpuAuxTensorHandler(const CpuAuxTensorHandler &) = delete;

        CpuAuxTensorHandler &operator=(const CpuAuxTensorHandler) = delete;

        ~CpuAuxTensorHandler() {
            if (_injected_tensor_pack) {
                _injected_tensor_pack->remove_tensor(_injected_slot_id);
            }
        }

        BIITensor *get() {
            return &_tensor;
        }

        BIITensor *operator()() {
            return &_tensor;
        }

    private:
        BITensor _tensor{};
        BIITensorPack *_injected_tensor_pack{nullptr};
        int _injected_slot_id{BITensorType::ACL_UNKNOWN};
    };
}


#endif //BATMANINFER_BI_CPU_AUX_TENSOR_HANDLER_HPP
