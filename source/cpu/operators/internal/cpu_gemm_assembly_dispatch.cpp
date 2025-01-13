//
// Created by Mason on 2025/1/6.
//

#include <cpu/operators/internal/cpu_gemm_assembly_dispatch.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <data/core/cpp/bi_cpp_validate.hpp>
#include <data/core/helpers/bi_memory_helpers.hpp>
#include <data/core/utils/bi_assembly_utils.hpp>
#include <cpu/kernels/assembly/bi_arm_gemm.hpp>
#include <cpu/kernels/assembly/bi_cpu_gemm_assmbly_wrapper.hpp>
#include <cpu/operators/cpu_transpose.hpp>
#include <cpu/utils/bi_cpu_aux_tensor_handler.hpp>


namespace BatmanInfer {
    namespace cpu {
        namespace {
            /**
             * @brief 预重排B矩阵
             * @tparam TypeInput
             * @tparam TypeWeight
             * @tparam TypeOutput
             * @param gemm_asm 指向 GEMM 操作对象的指针，负责矩阵 B 的预转置
             * @param dst 目标缓冲区，用于存储预转置后的矩阵 B
             * @param src 输入矩阵 B 的指针
             * @param src_ld 输入矩阵 B 的行步幅（行之间的存储间隔）
             * @param src_multi_stride 输入矩阵 B 的多矩阵步幅（批次之间的存储间隔）
             * @param num_threads 使用的线程数
             * @param transpose 是否需要对矩阵 B 进行转置
             */
            template<typename TypeInput, typename TypeWeight, typename TypeOutput>
            void
            run_parallel_pre_transpose_b_array(BatmanGemm::BIGemmCommon<TypeInput, TypeWeight, TypeOutput> *gemm_asm,
                                               BIITensor *dst,
                                               const TypeWeight *src,
                                               int src_ld,
                                               int src_multi_stride,
                                               unsigned int num_threads,
                                               bool transpose) {
                BI_COMPUTE_ERROR_ON(gemm_asm == nullptr);
                BI_COMPUTE_ERROR_ON(num_threads == 0);
                // 获取预转置矩阵 B 所需的总工作量（即矩阵 B 的大小或列数）
                const unsigned int w_size = gemm_asm->get_B_pre_transposed_array_size();

                // 创建一个大小为 num_threads 的工作负载数组，每个线程将处理一部分工作
                std::vector<BIIScheduler::BIWorkload> workloads(num_threads);
                for (unsigned int t = 0; t < num_threads; ++t) {
                    workloads[t] = [=](const ThreadInfo &info) {
                        // 计算当前线程的起始工作范围
                        const unsigned int start = (info.thread_id * w_size) / num_threads;
                        // 计算当前线程的结束工作范围
                        const unsigned int end = ((info.thread_id + 1) * w_size) / num_threads;

                        // 如果起始范围小于结束范围，说明有工作需要处理
                        if (start < end)
                            // 调用 gemm_asm 的部分预转置接口，让当前线程处理矩阵 B 的 [start, end) 部分
                            gemm_asm->pretranspose_B_array_part(dst->buffer(),
                                                                src,
                                                                src_ld,
                                                                src_multi_stride,
                                                                transpose,
                                                                start,
                                                                end);
                    };
                }
                // 调度器利用omp或者线程器进行分线程运行
                BINEScheduler::get().run_tagged_workloads(workloads, "BICpuGemmAssemblyDispatch/pretranspose_B_array");
            }
        } // namespace

        using namespace BatmanInfer::experimental;

        namespace {
            struct Params {
                unsigned int M;
                unsigned int N;
                unsigned int K;
                unsigned int batches;
                unsigned int multis;
                unsigned int sections;
                bool indirect;
            };

            /**
             * @brief 提取参数
             * @param a
             * @param b
             * @param d
             * @param info
             * @return
             */
            Params extract_parameters(const BIITensorInfo *a,
                                      const BIITensorInfo *b,
                                      const BIITensorInfo *d,
                                      const BIAsmGemmInfo &info) {
                BI_COMPUTE_ERROR_ON_NULLPTR(a, b, d);
                Params p{/* M */ static_cast<unsigned int>(d->tensor_shape().y()),
                        /* N */ static_cast<unsigned int>(d->tensor_shape().x()),
                        /* K */ static_cast<unsigned int>(a->tensor_shape().x()),
                        /* batches */ 1,
                        /* multis */ 1,
                        /* sections */ 1,
                        /* indirect */ false};

                if (info.method == BIAsmConvMethod::Conv || info.method == BIAsmConvMethod::Indirect) {
                    p.indirect = true;
                    p.sections = b->tensor_shape()[2] * b->tensor_shape()[3];
                } else {
                    p.multis = b->tensor_shape().z();
                    p.batches = d->tensor_shape().total_size_upper(2) / p.multis;
                }

                // 更新M如果GEMM3D作为输出
                if (info.depth_output_gemm3d != 0) {
                    p.M = d->tensor_shape().y() * d->tensor_shape().z();
                    p.batches = d->tensor_shape().total_size_upper(3) / p.multis;
                }
                return p;
            }

            /**
             * @brief 调度启发式
             *        根据指定的矩阵乘法方法 (GemmMethod) 和数据类型 (DataType)，返回适合的调度策略 (IScheduler::Hints)
             * @param method
             * @param data_type
             * @return
             */
            BIIScheduler::Hints scheduling_hint_heuristic(BatmanGemm::GemmMethod method,
                                                          BIDataType data_type) {
                // 调度汇编内核
                // 调度粒度的阈值，值为 200
                const int granule_threshold = 200;
                // 默认的调度提示(默认在X轴上并行)
                auto scheduling_hint = BIIScheduler::Hints(BIWindow::DimX);
                if (method == BatmanGemm::GemmMethod::GEMM_INTERLEAVED && data_type == BIDataType::F32)
                    scheduling_hint = BIIScheduler::Hints(BIWindow::DimX,
                                                          BIIScheduler::BIStrategyHint::DYNAMIC,
                                                          granule_threshold);
                else if (method == BatmanGemm::GemmMethod::GEMM_INTERLEAVED_2D && (data_type == BIDataType::F32 ||
                                                                                   data_type == BIDataType::F16 ||
                                                                                   data_type == BIDataType::U8 ||
                                                                                   data_type == BIDataType::S8))
                    scheduling_hint = BIIScheduler::Hints(BIIScheduler::split_dimensions_all,
                                                          BIIScheduler::BIStrategyHint::STATIC,
                                                          granule_threshold);
                else if (method == BatmanGemm::GemmMethod::QUANTIZE_WRAPPER_2D && (data_type == BIDataType::QASYMM8 ||
                                                                                   data_type ==
                                                                                   BIDataType::QASYMM8_SIGNED))
                    scheduling_hint = BIIScheduler::Hints(BIIScheduler::split_dimensions_all,
                                                          BIIScheduler::BIStrategyHint::STATIC,
                                                          granule_threshold);

                return scheduling_hint;
            }

            /**
             * @brief 如果ACL没有功能，则使用后备方案
             * @tparam TypeInput
             * @tparam TypeWeight
             * @tparam TypeOutput
             * @tparam OutputStage
             */
            template<typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage = BatmanGemm::Nothing>
            class Fallback : public BICpuGemmAssemblyDispatch::IFallback {
            public:
                /**
                 * 初始化函数的输入和输出
                 *
                 * @param a
                 * @param b
                 * @param c
                 * @param d
                 * @param args
                 * @param gemm_info
                 * @param os
                 */
                void configure(const BIITensorInfo *a,
                               const BIITensorInfo *b,
                               const BIITensorInfo *c,
                               BIITensorInfo *d,
                               BatmanGemm::GemmArgs args,
                               const BIAsmGemmInfo &gemm_info,
                               const OutputStage &os = {});

                /**
                 * 设置要使用的重新量化数据
                 * @param shifts 重新量化的移位值
                 * @param multipliers 重新量化的乘数值
                 * @return 一个元组，分别包含指向移位值和乘数值数据的指针
                 */
                std::tuple<bool, const int32_t *, const int32_t *, const int32_t *>
                set_requantize_data(const std::vector<int32_t> &shifts, const std::vector<int32_t> &multipliers);

                void run(BatmanInfer::BIITensorPack &tensors) override;

                void prepare(BatmanInfer::BIITensorPack &tensors) override;

                bool is_configured() const override;

                experimental::BIMemoryRequirements workspace() const override;

                bool isVarWeightsKernel() const override {
                    if (!_gemm_kernel_asm) {
                        return false;
                    }
                    const BatmanInfer::BIWeightFormat wf =
                            assembly_utils::map_to_batman_compute_weight_format(
                                    _gemm_kernel_asm->get_config().weight_format);
                    return wf != BatmanInfer::BIWeightFormat::UNSPECIFIED && wf != BatmanInfer::BIWeightFormat::ANY;
                }

                void update_quantization_parameters(const BIGEMMLowpOutputStageInfo &output_info,
                                                    const BIQuantizationInfo &a,
                                                    const BIQuantizationInfo &b,
                                                    const bool is_prepared,
                                                    const bool negated_offsets) override {
                    const int32_t negation = negated_offsets ? 1 : -1;
                    const int32_t a_offset = -a.uniform().offset * negation;
                    const int32_t b_offset = -b.uniform().offset * negation;

                    BatmanGemm::Requantize32 gemm_re_quant_info{};
                    if (output_info.gemmlowp_shifts.size() > 1) {
                        const auto re_quantize_data =
                                this->set_requantize_data(output_info.gemmlowp_multipliers,
                                                          output_info.gemmlowp_shifts);
                        gemm_re_quant_info = BatmanGemm::Requantize32(
                                nullptr, 0, a_offset, b_offset, output_info.gemmlowp_offset,
                                (std::get<0>(re_quantize_data)) ? std::get<1>(re_quantize_data) : nullptr,
                                std::get<2>(re_quantize_data),
                                std::get<3>(re_quantize_data), output_info.gemmlowp_min_bound,
                                output_info.gemmlowp_max_bound);
                    } else {
                        gemm_re_quant_info = BatmanGemm::Requantize32(nullptr, 0, a_offset, b_offset,
                                                                      output_info.gemmlowp_offset,
                                                                      -output_info.gemmlowp_shift,
                                                                      output_info.gemmlowp_multiplier,
                                                                      output_info.gemmlowp_min_bound,
                                                                      output_info.gemmlowp_max_bound);
                    }

                    _gemm_kernel_asm->update_quantization_parameters(gemm_re_quant_info);

                    // After update_quantization_parameters(), window may change, reconfigure it.
                    auto *opt = reinterpret_cast<kernel::BICpuGemmAssemblyWrapperKernel<TypeInput, TypeWeight, TypeOutput> *>(
                            _optimised_kernel.get());
                    const BIWindow win = to_window(_gemm_kernel_asm->get_window_size());
                    opt->configure_window(win);

                    _is_prepared = is_prepared;
                }

            private:
                enum AuxTensorIdx {
                    AsmGemmWorkspace = 0,
                    /**
                     * @brief 在传递给 gemm 或 pretranspose_B_array 之前，B（右侧）已被转置。
                     */
                    PrePretransposedB,
                    Pretranspose,
                    Count
                };

                /**
                 * @brief 配置间接缓冲区
                 * @param a Input tensor containing the Matrix A.
                 * @param b Input tensor containing the Matrix B.
                 * @param d Output tensor to store the result of matrix multiplication.
                 * @param info GEMM meta-data
                 */
                void configure_indirect(const BIITensorInfo *a,
                                        const BIITensorInfo *b,
                                        const BIITensorInfo *d,
                                        const BIAsmGemmInfo &info);

                /**
                 * @brief 准备间接缓冲区
                 * @param tensors
                 */
                void prepare_indirect_buffer(BIITensorPack &tensors);

                /** Operator to transpose B before gemm or pretranspose_B_array*/
                std::unique_ptr<BICpuTranspose> _pre_pretranspose_b{nullptr};
                /** 汇编GEMM内核 */
                std::shared_ptr<BatmanGemm::BIGemmCommon<TypeInput, TypeWeight, TypeOutput>> _gemm_kernel_asm{nullptr};
                /** Optimised Arm® Neon™ kernel */
                std::unique_ptr<BIINEKernel> _optimised_kernel{nullptr};
                /** Assembly GEMM workspace tensor info */
                BITensorInfo _workspace_info{};
                /** Pre-pre-transposed B tensor info */
                BITensorInfo _pre_pretransposed_b_info{};
                /** Pre-transpose tensor info */
                BITensorInfo _pretranspose_info{};
                /** Prepared flag */
                bool _is_prepared{false};
                /** GEMM meta-data */
                BIAsmGemmInfo _gemm_info{};
                /** GEMM kernel description */
                BatmanGemm::KernelDescription _kernel_info{};
                /** Per channel quantization shifts */
                std::vector<int32_t> _shifts{};
                std::vector<int32_t> right_shifts{};
                std::vector<int32_t> left_shifts{};
                /** Per channel quantization multipliers */
                std::vector<int32_t> _multipliers{};
                /** Indirect buffer */
                std::vector<const TypeInput *const *> _indirect_arg{};
                std::vector<const TypeInput *> _indirect_buf{};
                std::vector<TypeInput> _indirect_pad{};
                BatmanGemm::BIConvolutionParameters _cp{};
                experimental::BIMemoryRequirements _aux_mem{Count};
                bool _B_pretranspose_required{false};
                bool _is_b_constant{true};
                bool _is_c_constant{true};
                bool _run_pre_pretranspose_b{false};
                bool _B_pre_pretranspose_required{false};
            };
        } // namespace
    }
}