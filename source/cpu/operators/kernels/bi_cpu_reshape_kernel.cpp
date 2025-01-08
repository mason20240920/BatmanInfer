//
// Created by Mason on 2025/1/8.
//

#include "cpu/kernels/bi_cpu_reshape_kernel.hpp"

#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/bi_vlidate.hpp>

#include <data/core/utils/helpers/bi_utils.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                BIStatus validate_arguments(const BIITensorInfo *src,
                                            const BIITensorInfo *dst) {
                    BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
                    // Note: BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src) 在这里不需要，因为这个内核不使用 CPU FP16 指令。
                    BI_COMPUTE_RETURN_ERROR_ON(src->data_type() == BIDataType::UNKNOWN);

                    if (dst->tensor_shape().total_size() != 0) {
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
                        BI_COMPUTE_RETURN_ERROR_ON(
                                src->tensor_shape().total_size() != dst->tensor_shape().total_size());
                    }

                    return BIStatus{};
                }

                template<typename T>
                void reshape_tensor_per_element(const BIWindow &window,
                                                const BIITensor *src,
                                                BIITensor *dst) {
                    const BITensorShape &src_shape = src->info()->tensor_shape();
                    const BITensorShape &dst_shape = dst->info()->tensor_shape();

                    BIIterator dst_it(dst, window);

                    execute_window_loop(window, [&](const BICoordinates &dst_coord) {
                        BICoordinates src_coord = index2coords(src_shape, coords2index(dst_shape, dst_coord));
                        const auto output_str = dst->ptr_to_element(dst_coord);
                        const auto input_ptr = src->ptr_to_element(src_coord);

                        *reinterpret_cast<T *>(output_str) = *reinterpret_cast<T *>(input_ptr);
                    }, dst_it);
                }

                void reshape_tensor_per_element_selector(const BIWindow &window,
                                                         const BIITensor *src,
                                                         BIITensor *dst) {
                    switch (src->info()->data_type()) {
                        case BIDataType::U8:
                        case BIDataType::S8:
                        case BIDataType::QSYMM8:
                        case BIDataType::QASYMM8:
                        case BIDataType::QASYMM8_SIGNED:
                        case BIDataType::QSYMM8_PER_CHANNEL:
                            reshape_tensor_per_element<uint8_t>(window, src, dst);
                            break;
                        case BIDataType::U16:
                        case BIDataType::S16:
                        case BIDataType::F16:
                            reshape_tensor_per_element<uint16_t>(window, src, dst);
                            break;
                        case BIDataType::U32:
                        case BIDataType::S32:
                        case BIDataType::F32:
                            reshape_tensor_per_element<uint32_t>(window, src, dst);
                            break;
                        case BIDataType::U64:
                        case BIDataType::S64:
                        case BIDataType::F64:
                            reshape_tensor_per_element<uint64_t>(window, src, dst);
                            break;
                        default:
                            BI_COMPUTE_ERROR("Unsupported data type!");
                    }
                }

                void reshape_tensor_per_row(const BIWindow &window,
                                            const BIITensor *src,
                                            BIITensor *dst) {
                    const BITensorShape &src_shape = src->info()->tensor_shape();
                    const BITensorShape &dst_shape = dst->info()->tensor_shape();
                    BICoordinates src_coord{};
                    BICoordinates dst_coord{};

                    const auto element_size = dst->info()->element_size();
                    const auto window_start_x = static_cast<int>(window.x().start());
                    const auto window_end_x = static_cast<int>(window.x().end());
                    const auto src_row_size = static_cast<int>(src_shape[0]);
                    const auto row_size_in_bytes = src_row_size * element_size;

                    auto output_ptr = dst->ptr_to_element(dst_coord);
                    auto input_ptr = src->ptr_to_element(src_coord);

                    BIWindow win = window;
                    win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                    BIIterator dst_it(dst, win);
                    execute_window_loop(
                            win,
                            [&](BICoordinates &id) {
                                dst_coord = id;

                                for (int x = window_start_x; x < window_end_x; x += src_row_size) {
                                    src_coord = index2coords(src_shape, coords2index(dst_shape, dst_coord));
                                    output_ptr = dst->ptr_to_element(dst_coord);
                                    input_ptr = src->ptr_to_element(src_coord);

                                    std::memcpy(output_ptr, input_ptr, row_size_in_bytes);

                                    dst_coord.increment(BIWindow::DimX, src_row_size);
                                }
                            },
                            dst_it);
                }

                void reshape_tensor_per_window(const BIWindow &window, const BIITensor *src, BIITensor *dst) {
                    BIIterator src_it(src, window);
                    BIIterator dst_it(dst, window);

                    const size_t element_size = dst->info()->element_size();
                    const auto window_size = window.x().end() - window.x().start();
                    const auto window_size_in_bytes = window_size * element_size;

                    const auto input_ptr = src_it.ptr();
                    const auto output_ptr = dst_it.ptr();

                    std::memcpy(output_ptr, input_ptr, window_size_in_bytes);
                }
            } // namespace

            void BICpuReshapeKernel::configure(const BIITensorInfo *src, BIITensorInfo *dst) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));
                BI_COMPUTE_UNUSED(src);

                _reshape_tensor_fn = reshape_tensor_per_element_selector;
                // 配置内核窗口
                BIWindow win = calculate_max_window(*dst);

                BIICPPKernel::configure(win);
            }

            BIStatus
            BICpuReshapeKernel::validate(const BIITensorInfo *src, const BIITensorInfo *dst) {
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst));
                return BIStatus{};
            }

            void BICpuReshapeKernel::run_op(BIITensorPack &tensors,
                                            const BIWindow &window,
                                            const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICpuKernel::window(), window);

                const auto src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);
                _reshape_tensor_fn(window, src, dst);
            }

            const char *BICpuReshapeKernel::name() const {
                return "BICpuReshapeKernel";
            }

            size_t BICpuReshapeKernel::get_mws(const CPUInfo &platform, size_t thread_count) const {
                BI_COMPUTE_UNUSED(thread_count);
                BI_COMPUTE_UNUSED(platform);

                return BIICPPKernel::default_mws;
            }

            void BICpuReshapeKernel::prepare(BatmanInfer::BIITensorPack &tensors) {
                const auto src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);

                const BIITensorInfo *src_info = src->info();
                const BIITensorInfo *dst_info = dst->info();

                // Calculate kernel window based on the padding info
                BIWindow win;

                const bool src_has_holes = has_holes(*src_info, src_info->num_dimensions() - 1);
                const bool dst_has_holes = has_holes(*dst_info, dst_info->num_dimensions() - 1);
                const bool src_has_holes_in_x = has_holes(*src_info, BIWindow::DimX);
                const bool dst_has_holes_in_x = has_holes(*dst_info, BIWindow::DimX);
                const auto src_row_size = static_cast<int>(src_info->tensor_shape()[0]);
                const auto dst_row_size = static_cast<int>(dst_info->tensor_shape()[0]);

                if (!src_has_holes && !dst_has_holes) {
                    size_t split_dimension;

                    std::tie(win, split_dimension) = calculate_squashed_or_max_window(*dst_info);

                    /**
                     * 按窗口复制张量。如果源张量（src）和目标张量（dst）
                     * 是连续的内存分配且没有任何空洞或填充，则张量会被压缩为
                     * 一维窗口，并且我们可以在函数 reshape_tensor_per_window 中
                     * 使用单次内存复制调用来复制整个窗口。
                     */
                    if (split_dimension != BIWindow::DimY)
                        _reshape_tensor_fn = reshape_tensor_per_row;
                    else
                        _reshape_tensor_fn = reshape_tensor_per_window;
                } else {
                    win = calculate_max_window(*dst_info);

                    // 如果源和目标在X维度上没有空洞，并且它们的行中元素数量相同，则逐行复制张量。
                    if (!src_has_holes_in_x && !dst_has_holes_in_x && (src_row_size == dst_row_size))
                        _reshape_tensor_fn = reshape_tensor_per_row;
                    else
                        // 回退到逐元素复制
                        _reshape_tensor_fn = reshape_tensor_per_element_selector;
                }

                BIICPPKernel::configure(win);
            }
        }
    }
}