//
// Created by Mason on 2025/1/18.
//
#include "cpu/kernels/bi_cpu_copy_kernel.hpp"

#include "data/core/bi_error.h"
#include "data/core/bi_helpers.hpp"
#include "data/core/bi_i_tensor.hpp"
#include "data/core/bi_tensor_info.hpp"
#include "data/core/bi_types.hpp"
#include "data/core/utils/misc/bi_shape_calculator.hpp"
#include "data/core/bi_vlidate.hpp"
#include "data/core/bi_utils.hpp"

#include "data/core/helpers/bi_auto_configuration.hpp"
#include "data/core/helpers/bi_window_helpers.hpp"

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                BIStatus validate_arguments(const BIITensorInfo *src, const BIITensorInfo *dst,
                                            const PaddingList &padding = PaddingList()) {
                    BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
                    BI_COMPUTE_RETURN_ERROR_ON(src->data_type() == BIDataType::UNKNOWN);
                    BI_COMPUTE_RETURN_ERROR_ON(padding.size() > 4);

                    // Validate destination if initialized
                    if (dst->total_size() != 0) {
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(
                            misc::shape_calculator::compute_padded_shape(src->tensor_shape(), padding),
                            dst->tensor_shape());
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
                    }

                    return BIStatus{};
                }

                std::pair<BIStatus, BIWindow>
                validate_and_configure_window(const BIITensorInfo *src, BIITensorInfo *dst) {
                    // Destination auto inizialitation if not yet initialized
                    auto_init_if_empty(*dst, *src);
                    return std::make_pair(BIStatus{}, calculate_max_window(*dst));
                }

                std::pair<BIStatus, BIWindow>
                validate_and_configure_window_with_padding(const BIITensorInfo *src, BIITensorInfo *dst,
                                                           const PaddingList &padding) {
                    const BITensorShape src_shape = src->tensor_shape();
                    const BITensorShape padded_shape = misc::shape_calculator::compute_padded_shape(src_shape, padding);
                    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(padded_shape));
                    // Configure window
                    const BIWindow win = calculate_max_window(*dst, dst->dimension(0));
                    return std::make_pair(BIStatus{}, win);
                }
            } // namespace

            void BICpuCopyKernel::configure(const BIITensorInfo *src, BIITensorInfo *dst, const PaddingList &padding) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, padding));

                _padding = padding;

                std::pair<BIStatus, BIWindow> win_config;
                if (padding.empty()) {
                    win_config = validate_and_configure_window(src, dst);
                } else {
                    win_config = validate_and_configure_window_with_padding(src, dst, padding);
                }

                BI_COMPUTE_ERROR_THROW_ON(win_config.first);
                BIICpuKernel::configure(win_config.second);
            }

            void BICpuCopyKernel::dynamic_configure(const BIITensorInfo *dst) {
                auto win = BIICpuKernel::window();
                dynamic_origin_max_window(*dst, win, dst->dimension(0));
                BIICpuKernel::dynamic_configure(win);
            }


            BIStatus BICpuCopyKernel::validate(const BatmanInfer::BIITensorInfo *src,
                                               const BatmanInfer::BIITensorInfo *dst,
                                               const PaddingList &padding) {
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, padding));

                if (padding.empty()) {
                    BI_COMPUTE_RETURN_ON_ERROR(
                        validate_and_configure_window(src->clone().get(), dst->clone().get()).first);
                } else {
                    BI_COMPUTE_RETURN_ON_ERROR(
                        validate_and_configure_window_with_padding(src->clone().get(), dst->clone().get(),
                            padding).first);
                }

                return BIStatus{};
            }

            void BICpuCopyKernel::run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICpuKernel::window(), window);

                const auto src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);

                if (_padding.empty()) {
                    BIWindow dst_window{window};
                    dst_window.set(BIWindow::DimX,
                                   BIWindow::BIDimension(dst_window.x().start(), dst_window.x().end(),
                                                         src->info()->dimension(0)));
                    BIWindow out_slice = dst_window.first_slice_window_1D();
                    do {
                        BIIterator src_it(src, out_slice);
                        BIIterator dst_it(dst, out_slice);

                        execute_window_loop(
                            out_slice,
                            [&](const BICoordinates &) {
                                memcpy(dst_it.ptr(), src_it.ptr(),
                                       dst->info()->dimension(0) *
                                       dst->info()->element_size());
                            },
                            src_it, dst_it);
                    } while (dst_window.slide_window_slice_1D(out_slice));
                } else {
                    BIWindow src_window{window};
                    src_window.set(BIWindow::DimX,
                                   BIWindow::BIDimension(0, window.x().end() - _padding[0].first,
                                                         src->info()->dimension(0)));

                    BIIterator src_it(src, src_window);
                    BIIterator dst_it(dst, window);
                    const size_t row_size_in_bytes = src->info()->dimension(0) * src->info()->element_size();
                    execute_window_loop(
                        window,
                        [&](const BICoordinates &) {
                            auto dst_ptr = dst_it.ptr() + _padding[0].first * dst->info()->element_size();
                            std::memcpy(dst_ptr, src_it.ptr(), row_size_in_bytes);
                        },
                        src_it, dst_it);
                }
            }

            const char *BICpuCopyKernel::name() const {
                return "BICpuCopyKernel";
            }
        } // namespace kernels
    } // namespace cpu
}
