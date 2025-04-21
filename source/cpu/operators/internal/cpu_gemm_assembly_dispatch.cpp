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

#include <data/core/neon/kernels/arm_gemm/gemm_implementation.hpp>


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
                const unsigned int w_size = gemm_asm->get_B_pretranspose_window_size();

                // 创建一个大小为 num_threads 的工作负载数组，每个线程将处理一部分工作(没想到吧，操作很多，但是是单线程)
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
                Params p{
                    /* M */ static_cast<unsigned int>(d->tensor_shape().y()),
                    /* N */ static_cast<unsigned int>(d->tensor_shape().x()),
                    /* K */ static_cast<unsigned int>(a->tensor_shape().x()),
                    /* batches */ 1,
                    /* multis */ 1,
                    /* sections */ 1,
                    /* indirect */ false
                }; // 默认的是[M: 输出张量的y轴, 输出张量的x轴, 输入张量的x轴]

                if (info.method == BIAsmConvMethod::Conv || info.method == BIAsmConvMethod::Indirect) {
                    p.indirect = true;
                    p.sections = b->tensor_shape()[2] * b->tensor_shape()[3];
                } else {
                    p.multis = b->tensor_shape().z(); // b张量的三维的z轴(并行模块)
                    p.batches = d->tensor_shape().total_size_upper(2) / p.multis; // d张量大于维度2, z轴以上的所有进行并行
                }

                // 更新M如果GEMM3D作为输出
                if (info.depth_output_gemm3d != 0) {
                    p.M = d->tensor_shape().y() * d->tensor_shape().z();
                    p.batches = d->tensor_shape().total_size_upper(3) / p.multis;
                }
                return p;
            }

            /**
             * 简化版数据导出
             * @param a
             * @param b
             * @param d
             * @return
             */
            Params extract_simplify_parameters(const BIITensorInfo *a,
                                               const BIITensorInfo *b,
                                               const BIITensorInfo *d) {
                Params p{
                    /* M */ static_cast<unsigned int>(d->tensor_shape().y()),
                    /* N */ static_cast<unsigned int>(d->tensor_shape().x()),
                    /* K */ static_cast<unsigned int>(a->tensor_shape().x()),
                    /* batches */ 1,
                    /* multis */ 1,
                    /* sections */ 1,
                    /* indirect */ false
                }; // 默认的是[M: 输出张量的y轴, 输出张量的x轴, 输入张量的x轴]
                p.multis = b->tensor_shape().z(); // b张量的三维的z轴(并行模块)
                p.batches = d->tensor_shape().total_size_upper(2) / p.multis; // d张量大于维度2, z轴以上的所有进行并行
                return p;
            }

            /**
             * 根据输出张量d刷新batches
             * @param d
             * @param p
             */
            void update_parameters(const BIITensorInfo *d,
                                   Params &p) {
                p.batches = d->tensor_shape().total_size_upper(3) / p.multis;
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
            template<typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage =
                BatmanGemm::Nothing>
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

                void update_configure_parameters(const BatmanInfer::BIITensorInfo *a,
                                                 const BatmanInfer::BIITensorInfo *b,
                                                 const BatmanInfer::BIITensorInfo *d,
                                                 bool is_gemm) override;

                // size_t dynamic_tensor_b(size_t &align) const override;
                //
                // void dynamic_update_info();

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
                    auto *opt = reinterpret_cast<kernel::BICpuGemmAssemblyWrapperKernel<TypeInput, TypeWeight,
                        TypeOutput> *>(
                        _optimised_kernel.get());
                    const BIWindow win = to_window(_gemm_kernel_asm->get_window_size());
                    opt->configure_window(win);

                    _is_prepared = is_prepared;
                }

                bool has_stateless_impl() const override {
                    return _gemm_kernel_asm->get_working_size() == 0;
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
                std::shared_ptr<BatmanGemm::BIGemmCommon<TypeInput, TypeWeight, TypeOutput> > _gemm_kernel_asm{nullptr};
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

            // template<typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
            // void BatmanInfer::cpu::Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::dynamic_update_info() {
            //     // Forcing 128-byte alignment (required by 32-bit kernels)
            //     const size_t B_pretranspose_size = _gemm_kernel_asm->get_B_pretransposed_array_size();
            //     _pretranspose_info = BITensorInfo(BITensorShape(B_pretranspose_size), 1, BIDataType::U8);
            //     _aux_mem[Pretranspose].size = B_pretranspose_size;
            // }
            //
            // template<typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
            // size_t
            // BatmanInfer::cpu::Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::dynamic_tensor_b(
            //     size_t &align) const {
            //     align = 128;
            //     return _gemm_kernel_asm->get_B_pretransposed_array_size();
            // }

            template<typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
            void
            BatmanInfer::cpu::Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::update_configure_parameters(
                const BatmanInfer::BIITensorInfo *a, const BatmanInfer::BIITensorInfo *b,
                const BatmanInfer::BIITensorInfo *d, bool is_gemm) {
                Params p = extract_simplify_parameters(a, b, d);
                _gemm_kernel_asm->set_dynamic_M_size(p.M);
                _gemm_kernel_asm->set_dynamic_batch_size(p.batches);
                _gemm_kernel_asm->set_dynamic_N_size(p.N);
                _gemm_kernel_asm->set_dynamic_nmulti_size(p.multis);
                _gemm_kernel_asm->set_dynamic_K_size(p.K);
                _gemm_kernel_asm->update_parameters();
                BIWindow win = to_window(_gemm_kernel_asm->get_window_size());
                _optimised_kernel->dynamic_configure(win);
                const size_t B_pretranspose_size = _gemm_kernel_asm->get_B_pretransposed_array_size();
                _pretranspose_info = BITensorInfo(BITensorShape(B_pretranspose_size), 1, BIDataType::U8);
                _aux_mem[Pretranspose].size = B_pretranspose_size;
                if (!is_gemm) // 不是GEMM需要更新MatMul的B矩阵
                    _is_prepared = false;
            }

            template<typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
            std::tuple<bool, const int32_t *, const int32_t *, const int32_t *>
            Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::set_requantize_data(
                const std::vector<int32_t> &shifts,
                const std::vector<int32_t> &multipliers) {
                _multipliers = multipliers;
                _shifts = shifts;
                bool need_left = false;
                for (const auto s: _shifts) {
                    left_shifts.push_back(std::max(-s, int32_t(0)));
                    right_shifts.push_back(std::min(-s, int32_t(0)));
                    if (s < 0 && !need_left) {
                        need_left = true;
                    }
                }
                return std::make_tuple(need_left, left_shifts.data(), right_shifts.data(), _multipliers.data());
            }

            template<typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
            void Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::prepare_indirect_buffer(
                BatmanInfer::BIITensorPack &tensors) {
                auto a = tensors.get_const_tensor(BITensorType::ACL_SRC_0);
                const TypeInput *A_ptr = reinterpret_cast<TypeInput *>(a->buffer());
                const int multis = 1;
                const int batches = a->info()->tensor_shape().total_size_upper(3);
                const size_t stride_A = a->info()->strides_in_bytes().y() / sizeof(TypeInput);
                const size_t batch_stride_A = a->info()->strides_in_bytes()[3] / sizeof(TypeInput);
                const size_t multi_stride_A = a->info()->strides_in_bytes()[4] / sizeof(TypeInput);

                const size_t output_hw = _cp.output_height * _cp.output_width;
                const int batch_size = _cp.kernel_height * _cp.kernel_width * output_hw * sizeof(TypeInput);
                const size_t batch_stride = batch_size / sizeof(TypeInput);
                const int multi_size = batch_size * batches;
                const size_t multi_stride = multi_size / sizeof(TypeInput);

                for (int64_t m = 0; m < multis; m++) {
                    for (int64_t b = 0; b < batches; b++) {
                        for (int64_t output_y = 0; output_y < _cp.output_height; output_y++) {
                            for (int64_t output_x = 0; output_x < _cp.output_width; output_x++) {
                                int64_t output_xy = (output_y * _cp.output_width) + output_x;

                                for (int64_t kernel_y = 0; kernel_y < _cp.kernel_height; kernel_y++) {
                                    for (int64_t kernel_x = 0; kernel_x < _cp.kernel_width; kernel_x++) {
                                        int64_t input_x =
                                                (output_x * _cp.output_stride_w) + kernel_x - _cp.padding_left;
                                        int64_t input_y = (output_y * _cp.output_stride_h) + kernel_y - _cp.padding_top;
                                        int64_t kernel_xy = (kernel_y * _cp.kernel_width) + kernel_x;
                                        int64_t input_xy = (input_y * _cp.input_width) + input_x;

                                        if (input_x < 0 || input_x >= _cp.input_width || input_y < 0 ||
                                            input_y >= _cp.input_height) {
                                            _indirect_buf[m * multi_stride + b * batch_stride + kernel_xy * output_hw +
                                                          output_xy] =
                                                    _indirect_pad.data();
                                        } else {
                                            _indirect_buf[m * multi_stride + b * batch_stride + kernel_xy * output_hw +
                                                          output_xy] =
                                                    A_ptr +
                                                    (m * multi_stride_A + b * batch_stride_A + input_xy * stride_A);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            template<typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
            void Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::configure_indirect(const BIITensorInfo *a,
                const BIITensorInfo *b,
                const BIITensorInfo *d,
                const BIAsmGemmInfo &info) {
                BI_COMPUTE_ERROR_ON(
                    !(info.method == BIAsmConvMethod::Conv || info.method == BIAsmConvMethod::Indirect));

                float zeropad = 0.f;
                if (is_data_type_quantized(a->data_type())) {
                    zeropad = a->quantization_info().uniform().offset;
                }

                const auto input_width = static_cast<int64_t>(a->tensor_shape()[1]);
                const auto input_height = static_cast<int64_t>(a->tensor_shape()[2]);
                const auto input_channels = static_cast<int64_t>(a->tensor_shape()[0]);
                const auto kernel_width = static_cast<int64_t>(b->tensor_shape()[2]);
                const auto kernel_height = static_cast<int64_t>(b->tensor_shape()[3]);
                const auto output_width = static_cast<int64_t>(d->tensor_shape()[1]);
                const auto output_height = static_cast<int64_t>(d->tensor_shape()[2]);

                _cp = {
                    input_width,
                    input_height,
                    input_channels,
                    kernel_width,
                    kernel_height,
                    output_width,
                    output_height,
                    info.ps_info.stride().first,
                    info.ps_info.stride().second,
                    1,
                    1,
                    info.padding_top,
                    info.padding_left,
                    zeropad
                };

                if (info.method == BIAsmConvMethod::Conv) {
                    _gemm_kernel_asm->set_convolution_parameters(_cp);
                }

                if (info.method == BIAsmConvMethod::Indirect) {
                    const unsigned int multis = 1;
                    const unsigned int batches = a->tensor_shape().total_size_upper(3);
                    const unsigned int kernel_hw = _cp.kernel_width * _cp.kernel_height;
                    const unsigned int output_hw = _cp.output_width * _cp.output_height;

                    using TypeInputPtr = TypeInput *;
                    const int batch_size = kernel_hw * output_hw * sizeof(TypeInputPtr);
                    const size_t batch_stride = batch_size / sizeof(TypeInputPtr);
                    const int multi_size = batch_size * batches;
                    const size_t multi_stride = multi_size / sizeof(TypeInputPtr);

                    _indirect_buf = std::vector<const TypeInput *>(multi_size * multis);
                    _indirect_arg = std::vector<const TypeInput *const *>(
                        sizeof(TypeInput **) * kernel_hw * multis * batches);
                    _indirect_pad = std::vector<TypeInput>(_cp.input_channels, TypeInput(zeropad));

                    // Set indirect argument
                    int64_t pos = 0;
                    for (int64_t m = 0; m < multis; m++) {
                        for (int64_t b = 0; b < batches; b++) {
                            for (int64_t kernel_xy = 0; kernel_xy < kernel_hw; kernel_xy++) {
                                _indirect_arg[pos++] = &_indirect_buf[m * multi_stride + b * batch_stride +
                                                                      kernel_xy * output_hw];
                            }
                        }
                    }

                    _gemm_kernel_asm->set_indirect_parameters(a->tensor_shape()[0], _indirect_arg.data());
                }
            }

            template<typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
            void Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::configure(const BIITensorInfo *a,
                const BIITensorInfo *b,
                const BIITensorInfo *c,
                BIITensorInfo *d,
                BatmanGemm::GemmArgs args,
                const BIAsmGemmInfo &gemm_info,
                const OutputStage &os) {
                _is_b_constant = b->are_values_constant();
                _is_c_constant = c ? c->are_values_constant() : true;

                _gemm_kernel_asm = BatmanGemm::gemm<TypeInput, TypeWeight, TypeOutput, OutputStage>(args, os);
                if (_gemm_kernel_asm == nullptr) {
                    //configuration not supported: Leave function unconfigured:
                    return;
                }

                BatmanGemm::GemmConfig gemm_cfg = _gemm_kernel_asm->get_config();

                // batman infer wrapper for the Gemm object (see above)
                auto acl_gemm_wrapper = std::make_unique<kernel::BICpuGemmAssemblyWrapperKernel<TypeInput, TypeWeight,
                    TypeOutput> >();
                BI_COMPUTE_ERROR_ON(acl_gemm_wrapper == nullptr);
                acl_gemm_wrapper->configure(_gemm_kernel_asm.get(), gemm_cfg.filter);
                const size_t workspace_size = _gemm_kernel_asm->get_working_size();
                const unsigned int alignment = 4096;
                _workspace_info = BITensorInfo(BITensorShape(workspace_size), 1, BIDataType::U8);
                _aux_mem[AsmGemmWorkspace] =
                        BIMemoryInfo(offset_int_vec(AsmGemmWorkspace), MemoryLifetime::Temporary, workspace_size,
                                     alignment);

                //if we disable this code below in brackets then ConvLayer deadlocks when threads > 1 and
                //the shapes are In=1x1x1024 Weights=1x1x1024x1001 Biases=1001 Out=1x1x1001
                {
                    const unsigned int window_size = _gemm_kernel_asm->get_window_size().total_size();
                    if (window_size < static_cast<unsigned int>(args._maxthreads)) {
                        _gemm_kernel_asm->set_nthreads(window_size);
                    }
                }

                _optimised_kernel = std::move(acl_gemm_wrapper);
                _gemm_info = gemm_info;

                // Check if we need to pre-pretranspose B. Fixed format kernels need no pre-pretranspose.
                _B_pre_pretranspose_required = _gemm_info.transpose_b && !isVarWeightsKernel();
                _B_pretranspose_required = _gemm_kernel_asm->B_pretranspose_required();

                const bool kernel_supports_transpose = _gemm_kernel_asm->B_pretranspose_supports_transpose();
                const bool kernel_can_fuse_transpose = _B_pretranspose_required && kernel_supports_transpose;
                _run_pre_pretranspose_b = _B_pre_pretranspose_required && !kernel_can_fuse_transpose;

                if (_run_pre_pretranspose_b) {
                    _pre_pretranspose_b = std::make_unique<BICpuTranspose>();
                    _pre_pretranspose_b->configure(b, &_pre_pretransposed_b_info);
                    MemoryLifetime lifetime;
                    if (_is_b_constant) {
                        if (_B_pretranspose_required) {
                            // PrePretransposedB tensor is only used in prepare(), but is then succeeded by Pretranspose
                            // So PrePretransposedB can be freed inside prepare()
                            lifetime = MemoryLifetime::Prepare;
                        } else {
                            // PrePretransposedB tensor is only used in prepare(), but is the final transformation of B
                            // So PrePretransposedB needs to persist beyond prepare()
                            lifetime = MemoryLifetime::Persistent;
                        }
                    } else {
                        // PrePretransposedB tensor is always used in run() and doesn't need to persist
                        lifetime = MemoryLifetime::Temporary;
                    }
                    // Forcing 128-byte alignment (required by 32-bit kernels)
                    const unsigned int alignment = 128;
                    _aux_mem[PrePretransposedB] =
                            BIMemoryInfo(offset_int_vec(PrePretransposedB), lifetime,
                                         _pre_pretransposed_b_info.total_size(),
                                         alignment);
                }

                // Check for pre-transposed support
                if (_B_pretranspose_required) {
                    // Fixed format kernels need no pretranspose.
                    BI_COMPUTE_ERROR_ON(BatmanInfer::is_fixed_format(
                        assembly_utils::map_to_batman_compute_weight_format(
                            _gemm_kernel_asm->get_config().weight_format)));
                    // Forcing 128-byte alignment (required by 32-bit kernels)
                    const unsigned int alignment = 128;
                    const size_t B_pretranspose_size = _gemm_kernel_asm->get_B_pretransposed_array_size();
                    _pretranspose_info = BITensorInfo(BITensorShape(B_pretranspose_size), 1, BIDataType::U8);
                    MemoryLifetime lifetime = _is_b_constant ? MemoryLifetime::Persistent : MemoryLifetime::Temporary;
                    _aux_mem[Pretranspose] = BIMemoryInfo(offset_int_vec(Pretranspose), lifetime, B_pretranspose_size,
                                                          alignment);
                }

                // Handle indirect GEMM convolution
                if (gemm_info.method == BIAsmConvMethod::Conv || gemm_info.method == BIAsmConvMethod::Indirect) {
                    configure_indirect(a, b, d, gemm_info);
                }

                if (std::is_same<OutputStage, BatmanGemm::DequantizeFloat>::value) {
                    // Output dequantization is just the two src scales multiplied together
                    _gemm_kernel_asm->set_dequantize_scale(a->quantization_info().uniform().scale *
                                                           b->quantization_info().uniform().scale);
                }
            }

            // 动态更新transpose

            template<typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
            void Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::prepare(BIITensorPack &tensors) {
                if (!_is_prepared) {
                    auto b = tensors.get_const_tensor(BITensorType::ACL_SRC_1);
                    auto c = tensors.get_const_tensor(BITensorType::ACL_SRC_2);
                    BI_COMPUTE_ERROR_ON_NULLPTR(b);

                    // Setup up matrix bias in the assembly kernel, it's just a pointer to matrix C.
                    if (c && c->info()->data_type() == BIDataType::S32) {
                        _gemm_kernel_asm->set_quantized_bias(
                            reinterpret_cast<const int32_t *>(c->buffer() +
                                                              c->info()->offset_first_element_in_bytes()),
                            0);
                    }
                    const BIITensor *b_to_use = b;

                    // Pre-pretranspose B if required
                    CpuAuxTensorHandler pre_pretransposed_b(
                        offset_int_vec(PrePretransposedB), _pre_pretransposed_b_info, tensors,
                        /*pack_inject: no need to inject into tensors*/
                        false,
                        /*bypass_alloc: no need to allocate if pre-pretranspose B is not required as this handle will not be used*/
                        !_run_pre_pretranspose_b);

                    if (_run_pre_pretranspose_b) {
                        BI_COMPUTE_ERROR_ON(_pre_pretranspose_b == nullptr);
                        BIITensorPack pre_pretranspose_pack{
                            {ACL_SRC, b_to_use},
                            {ACL_DST, pre_pretransposed_b.get()}
                        };
                        _pre_pretranspose_b->run(pre_pretranspose_pack);
                        b_to_use = pre_pretransposed_b.get();
                    }

                    // Pretranspose B if required
                    if (_B_pretranspose_required) {
                        // Fixed format kernels need no pretranspose.
                        BI_COMPUTE_ERROR_ON(BatmanInfer::is_fixed_format(
                            assembly_utils::map_to_batman_compute_weight_format(
                                _gemm_kernel_asm->get_config().weight_format)));
                        const int ldb = b_to_use->info()->strides_in_bytes().y() / b_to_use->info()->element_size();
                        const auto in1_ptr = reinterpret_cast<const TypeWeight *>(
                            b_to_use->buffer() + b_to_use->info()->offset_first_element_in_bytes());
                        const int multi_stride_b =
                                b_to_use->info()->strides_in_bytes().z() / b_to_use->info()->element_size();

                        CpuAuxTensorHandler pretranspose(offset_int_vec(Pretranspose), _pretranspose_info, tensors,
                                                         false);

                        BI_COMPUTE_ERROR_ON(pretranspose.get()->buffer() == nullptr);

                        const bool kernel_supports_transpose = _gemm_kernel_asm->B_pretranspose_supports_transpose();
                        run_parallel_pre_transpose_b_array<TypeInput, TypeWeight, TypeOutput>(
                            _gemm_kernel_asm.get(), pretranspose.get(), in1_ptr, ldb, multi_stride_b,
                            BINEScheduler::get().num_threads(),
                            _B_pre_pretranspose_required && kernel_supports_transpose);

                        b->mark_as_unused();
                        // Note that we don't need to mark b_to_use as unused, as if it's been assigned to pre_pretransposed_b,
                        // its memory will be auto-managed by the handler
                    }

                    if (_gemm_info.method == BIAsmConvMethod::Indirect) {
                        prepare_indirect_buffer(tensors);
                    }

                    _is_prepared = true;
                }
            }

            template<typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
            bool Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::is_configured() const {
                return _optimised_kernel != nullptr;
            }

            template<typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
            experimental::BIMemoryRequirements
            Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::workspace() const {
                return _aux_mem;
            }

            template<typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
            void Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::run(BIITensorPack &tensors) {
                auto a = tensors.get_const_tensor(BITensorType::ACL_SRC_0);
                auto b = tensors.get_const_tensor(BITensorType::ACL_SRC_1);
                auto c = tensors.get_const_tensor(BITensorType::ACL_SRC_2);
                auto d = tensors.get_tensor(BITensorType::ACL_DST);
                BI_COMPUTE_ERROR_ON_NULLPTR(a, d);

                // 动态更新B张量数据(变更为最新的形状)
                // dynamic_update_info();

                // 如果源量化是动态的，则仅在运行时更新。
                if (std::is_same<OutputStage, BatmanGemm::DequantizeFloat>::value &&
                    (a->info()->quantization_info().is_dynamic() || b->info()->quantization_info().is_dynamic())) {
                    // Output dequantization is just the two src scales multiplied together
                    _gemm_kernel_asm->set_dequantize_scale(a->info()->quantization_info().uniform().scale *
                                                           b->info()->quantization_info().uniform().scale);
                }

                int lda = a->info()->strides_in_bytes().y() / a->info()->element_size(); // A矩阵的步长
                int ldb = 0; // B的步长
                const int ldd = d->info()->strides_in_bytes().y() / d->info()->element_size(); // D矩阵的步长

                // 确定批次和多实例维度的索引
                const size_t a_batch_idx = _gemm_info.reinterpret_input_as_3d != 0 ? 3 : 2;
                const size_t a_multi_idx = a_batch_idx + 1;
                const size_t d_batch_idx = _gemm_info.depth_output_gemm3d != 0 ? 3 : 2;
                const size_t d_multi_idx = d_batch_idx + 1;

                // 计算批次和多实例的步长
                int batch_stride_a = a->info()->strides_in_bytes()[a_batch_idx] / a->info()->element_size();
                const int batch_stride_d = d->info()->strides_in_bytes()[d_batch_idx] / d->info()->element_size();

                int multi_stride_a = a->info()->strides_in_bytes()[a_multi_idx] / a->info()->element_size();
                int multi_stride_b = 0;
                const int multi_stride_d = d->info()->strides_in_bytes()[d_multi_idx] / d->info()->element_size();
                // 获取输入、权重和输出的数据指针
                auto in0_ptr = reinterpret_cast<const TypeInput *>(a->buffer() +
                                                                   a->info()->offset_first_element_in_bytes());
                // a矩阵的首个指针
                const TypeWeight *in1_ptr = nullptr;
                auto out_ptr = reinterpret_cast<TypeOutput *>(d->buffer() +
                                                              d->info()->offset_first_element_in_bytes()); // d矩阵的数据指针位置

                const BIITensor *b_to_use = b;

                // Pre-pretranspose B if required
                CpuAuxTensorHandler pre_pretransposed_b(
                    offset_int_vec(PrePretransposedB), _pre_pretransposed_b_info, tensors,
                    false /*pack_inject: no need to inject into tensors*/,
                    !_run_pre_pretranspose_b
                    /*bypass_alloc: no need to allocate if pre-pretranspose B is not required as this handle will not be used*/);
                if (b_to_use && !_is_b_constant && _run_pre_pretranspose_b) {
                    BI_COMPUTE_ERROR_ON(_pre_pretranspose_b == nullptr);
                    BIITensorPack pre_pretranspose_pack{
                        {ACL_SRC, b_to_use},
                        {ACL_DST, pre_pretransposed_b.get()}
                    };
                    _pre_pretranspose_b->run(pre_pretranspose_pack);
                    b_to_use = pre_pretransposed_b.get();
                }

                // Check if B is pre-transposed and de-reference if not
                if (b_to_use && !_gemm_kernel_asm->B_is_pretransposed()) {
                    ldb = b_to_use->info()->strides_in_bytes().y() / b_to_use->info()->element_size();
                    multi_stride_b = b_to_use->info()->strides_in_bytes().z() / b_to_use->info()->element_size();
                    in1_ptr = reinterpret_cast<const TypeWeight *>(b_to_use->buffer() +
                                                                   b_to_use->info()->offset_first_element_in_bytes());
                }

                // If necessary, run pretranspose every time if either weights or biases are non-constant
                if ((b_to_use && !_is_b_constant) ||
                    (c && !_is_c_constant && c->info()->data_type() == BIDataType::S32)) {
                    if (c && c->info()->data_type() == BIDataType::S32) {
                        _gemm_kernel_asm->set_quantized_bias(
                            reinterpret_cast<const int32_t *>(c->buffer() +
                                                              c->info()->offset_first_element_in_bytes()),
                            0);
                    }

                    // Pretranspose B if required
                    if (b_to_use && _B_pretranspose_required) {
                        // Fixed format kernels need no pretranspose.
                        BI_COMPUTE_ERROR_ON(BatmanInfer::is_fixed_format(
                            assembly_utils::map_to_batman_compute_weight_format(
                                _gemm_kernel_asm->get_config().weight_format)));
                        const int ldb = b_to_use->info()->strides_in_bytes().y() / b_to_use->info()->element_size();
                        const auto b_ptr = reinterpret_cast<const TypeWeight *>(b_to_use->buffer() +
                            b_to_use->info()->offset_first_element_in_bytes());
                        const int multi_stride_b =
                                b_to_use->info()->strides_in_bytes().z() / b_to_use->info()->element_size();

                        CpuAuxTensorHandler pretranspose(offset_int_vec(Pretranspose), _pretranspose_info, tensors,
                                                         true);
                        BI_COMPUTE_ERROR_ON(pretranspose.get()->buffer() == nullptr);

                        if (_is_b_constant) {
                            _gemm_kernel_asm->requantize_bias(pretranspose.get()->buffer(), b_ptr, ldb, multi_stride_b);
                        } else {
                            const bool kernel_supports_transpose = _gemm_kernel_asm->
                                    B_pretranspose_supports_transpose();
                            run_parallel_pre_transpose_b_array<TypeInput, TypeWeight, TypeOutput>(
                                _gemm_kernel_asm.get(), pretranspose.get(), b_ptr, ldb, multi_stride_b,
                                BINEScheduler::get().num_threads(),
                                _B_pre_pretranspose_required && kernel_supports_transpose);
                        }
                    }
                }

                const auto scheduling_hint = scheduling_hint_heuristic(_kernel_info.method, d->info()->data_type());

                // Set workspace if needed and reset number of threads as buffer manager gets re-created with max_threads
                CpuAuxTensorHandler workspace(offset_int_vec(AsmGemmWorkspace), _workspace_info, tensors, false);
                if (workspace.get()->buffer() != nullptr) {
                    _gemm_kernel_asm->set_working_space(reinterpret_cast<void *>(workspace.get()->buffer()));
                    const unsigned int split_dim = scheduling_hint.split_dimension();
                    const unsigned int window_size = _gemm_kernel_asm->get_window_size().total_size();
                    unsigned int num_threads = BINEScheduler::get().num_threads();
                    if (window_size < num_threads) {
                        num_threads = window_size;
                    }
                    if (split_dim != BIIScheduler::split_dimensions_all) {
                        // Make sure the kernel does not expect more threads than we can actually spawn
                        const unsigned int num_iterations = _optimised_kernel->window().num_iterations(split_dim);
                        num_threads = std::min(num_iterations, num_threads);
                    }
                    _gemm_kernel_asm->set_nthreads(num_threads);
                }

                // Prepare assembly kernel
                prepare(tensors);

                // Setup up matrix bias in the assembly kernel, it's just a pointer to matrix C.
                TypeOutput *bias = nullptr;
                if (c && c->info()->data_type() != BIDataType::S32) {
                    bias = reinterpret_cast<TypeOutput *>(c->buffer() + c->info()->offset_first_element_in_bytes());
                }

                if (_gemm_info.method == BIAsmConvMethod::Indirect) {
                    in0_ptr = nullptr;
                    lda = 0;
                    batch_stride_a = 0;
                    multi_stride_a = 0;
                }

                BITensor in0_tensor;
                in0_tensor.allocator()->init(*(a->info()));
                in0_tensor.allocator()->import_memory(const_cast<TypeInput *>(in0_ptr));

                BITensor in1_tensor;
                if (b) {
                    in1_tensor.allocator()->init(*(b->info()));
                    in1_tensor.allocator()->import_memory(const_cast<TypeWeight *>(in1_ptr));
                }

                BITensor bias_tensor;
                if (c) {
                    bias_tensor.allocator()->init(*(c->info()));
                    bias_tensor.allocator()->import_memory(bias);
                }

                BITensor out_tensor;
                out_tensor.allocator()->init(*(d->info()));
                out_tensor.allocator()->import_memory(out_ptr);

                BIITensorPack gemm_pack{
                    {ACL_SRC_0, &in0_tensor},
                    {ACL_SRC_1, &in1_tensor},
                    {ACL_SRC_2, &bias_tensor},
                    {ACL_SRC_3, workspace.get()},
                    {ACL_DST, &out_tensor}
                };

                // Set gemm parameters
                _gemm_kernel_asm->set_arrays(in0_ptr, lda, batch_stride_a, multi_stride_a, in1_ptr, ldb, multi_stride_b,
                                             out_ptr,
                                             ldd, batch_stride_d, multi_stride_d, bias, 0);

                // // configure the window, if window is change, just dynamic change the window
                // auto is_dynamic_m = _gemm_kernel_asm->set_dynamic_M_size(in0_tensor.info()->tensor_shape().y());
                // auto is_dynamic_batch = _gemm_kernel_asm->set_dynamic_batch_size(
                //     in0_tensor.info()->tensor_shape().z());
                // auto is_dynamic_multi = _gemm_kernel_asm->set_dynamic_nmulti_size(
                //     in0_tensor.info()->tensor_shape()[3]);
                // _gemm_kernel_asm->update_parameters();
                // BIWindow win = to_window(_gemm_kernel_asm->get_window_size());
                // _optimised_kernel->dynamic_configure(win);

                // Schedule
                BINEScheduler::get().schedule_op(_optimised_kernel.get(), scheduling_hint,
                                                 _optimised_kernel->window(),
                                                 gemm_pack);
            }

            template<typename TypeInput, typename TypeWeight, typename TypeOutput>
            void create_arm_gemm(std::unique_ptr<BICpuGemmAssemblyDispatch::IFallback> &arm_gemm,
                                 const BIITensorInfo *a,
                                 const BIITensorInfo *b,
                                 const BIITensorInfo *c,
                                 BIITensorInfo *d,
                                 BatmanGemm::Activation activation,
                                 const BIAsmGemmInfo &info) {
                Params p = extract_parameters(a, b, d, info);
                const CPUInfo &ci = BINEScheduler::get().cpu_info();
                unsigned int num_threads = BINEScheduler::get().num_threads();

                BatmanGemm::GemmConfig cfg;
                cfg.weight_format = assembly_utils::map_to_batman_gemm_weight_format(info.weight_format);
                BatmanGemm::GemmArgs args(&ci, p.M, p.N, p.K, p.sections, p.batches, p.multis, p.indirect, activation,
                                          static_cast<int>(num_threads),
                                          info.fixed_format, info.fast_mode, info.accumulate, &cfg);

                // Create arm_gemm fallback
                auto fallback = std::make_unique<Fallback<TypeInput, TypeWeight, TypeOutput> >();
                fallback->configure(a, b, c, d, args, info);
                arm_gemm = std::move(fallback);
            }

            template<typename TypeInput, typename TypeWeight, typename TypeOutput>
            void create_arm_gemm_dequant(std::unique_ptr<BICpuGemmAssemblyDispatch::IFallback> &arm_gemm,
                                         const BIITensorInfo *a,
                                         const BIITensorInfo *b,
                                         const BIITensorInfo *c,
                                         BIITensorInfo *d,
                                         BatmanGemm::Activation activation,
                                         const BIAsmGemmInfo &info) {
                BI_COMPUTE_UNUSED(activation);

                Params p = extract_parameters(a, b, d, info);
                const CPUInfo &ci = BINEScheduler::get().cpu_info();
                const unsigned int num_threads = BINEScheduler::get().num_threads();

                BatmanGemm::GemmConfig cfg;
                cfg.weight_format = assembly_utils::map_to_batman_gemm_weight_format(info.weight_format);
                BatmanGemm::GemmArgs args(&ci, p.M, p.N, p.K, p.sections, p.batches, p.multis, p.indirect, activation,
                                          static_cast<int>(num_threads),
                                          info.fixed_format, info.fast_mode, info.accumulate, &cfg);

                // Create arm_gemm fallback
                auto fallback = std::make_unique<Fallback<TypeInput, TypeWeight, TypeOutput,
                    BatmanGemm::DequantizeFloat> >();

                // Configure requantization info
                const BIGEMMLowpOutputStageInfo os_info = info.output_stage;

                BatmanGemm::DequantizeFloat gemm_dequant_info{};
                gemm_dequant_info = BatmanGemm::DequantizeFloat(d->quantization_info().uniform().scale);

                fallback->configure(a, b, c, d, args, info, gemm_dequant_info);
                arm_gemm = std::move(fallback);
            }

            template<typename TypeInput, typename TypeWeight, typename TypeOutput>
            void create_arm_gemm_quant(std::unique_ptr<BICpuGemmAssemblyDispatch::IFallback> &arm_gemm,
                                       const BIITensorInfo *a,
                                       const BIITensorInfo *b,
                                       const BIITensorInfo *c,
                                       BIITensorInfo *d,
                                       BatmanGemm::Activation activation,
                                       const BIAsmGemmInfo &info) {
                BI_COMPUTE_UNUSED(activation);
                Params p = extract_parameters(a, b, d, info);
                const CPUInfo &ci = BINEScheduler::get().cpu_info();
                const unsigned int num_threads = BINEScheduler::get().num_threads();

                BatmanGemm::GemmConfig cfg;
                cfg.weight_format = assembly_utils::map_to_batman_gemm_weight_format(info.weight_format);
                BatmanGemm::GemmArgs args(&ci, p.M, p.N, p.K, p.sections, p.batches, p.multis, p.indirect, activation,
                                          static_cast<int>(num_threads),
                                          info.fixed_format, info.fast_mode, info.accumulate, &cfg);

                // Create arm_gemm fallback
                auto fallback = std::make_unique<Fallback<TypeInput, TypeWeight, TypeOutput,
                    BatmanGemm::Requantize32> >();

                // Configure requantization info
                const int32_t negation = info.negated_offsets ? 1 : -1;
                const int32_t a_offset = -a->quantization_info().uniform().offset * negation;
                const int32_t b_offset = -b->quantization_info().uniform().offset * negation;
                const BIGEMMLowpOutputStageInfo os_info = info.output_stage;

                BatmanGemm::Requantize32 gemm_requant_info{};
                if (os_info.gemmlowp_shifts.size() > 1) {
                    const auto requantize_data =
                            fallback->set_requantize_data(os_info.gemmlowp_shifts, os_info.gemmlowp_multipliers);
                    gemm_requant_info = BatmanGemm::Requantize32(
                        nullptr, 0, a_offset, b_offset, os_info.gemmlowp_offset,
                        (std::get<0>(requantize_data)) ? std::get<1>(requantize_data) : nullptr,
                        std::get<2>(requantize_data),
                        std::get<3>(requantize_data), os_info.gemmlowp_min_bound, os_info.gemmlowp_max_bound);
                } else {
                    gemm_requant_info =
                            BatmanGemm::Requantize32(nullptr, 0, a_offset, b_offset, os_info.gemmlowp_offset,
                                                     -os_info.gemmlowp_shift,
                                                     os_info.gemmlowp_multiplier, os_info.gemmlowp_min_bound,
                                                     os_info.gemmlowp_max_bound);
                }

                // Configure fallback
                fallback->configure(a, b, c, d, args, info, gemm_requant_info);
                arm_gemm = std::move(fallback);
            }
        } // namespace

        BICpuGemmAssemblyDispatch::BICpuGemmAssemblyDispatch() : _batman_gemm(nullptr) {
        }

        /**
         * 汇编GEMM的优化
         * @param expected_weight_format 权重张量布局
         * @param a  张量信息a
         * @param b  张量信息b
         * @param c  张量信息c
         * @param d  张量信息d
         * @param info 汇编信息
         * @return
         */
        BIStatus BICpuGemmAssemblyDispatch::has_opt_impl(BatmanInfer::BIWeightFormat &expected_weight_format,
                                                         const BatmanInfer::BIITensorInfo *a,
                                                         const BatmanInfer::BIITensorInfo *b,
                                                         const BatmanInfer::BIITensorInfo *c,
                                                         const BatmanInfer::BIITensorInfo *d,
                                                         const BatmanInfer::cpu::BIAsmGemmInfo &info) {
            // 验证合理性
            BI_COMPUTE_ERROR_ON_NULLPTR(a, b, d); // 验证a, b, d是否为空
            BI_COMPUTE_UNUSED(c); // c 不使用
            BatmanGemm::Activation act = assembly_utils::map_to_batman_gemm_activation(info.activation_info); // 映射激活函数
            Params p = extract_parameters(a, b, d, info); // 提取优化参数信息
            const CPUInfo &ci = BINEScheduler::get().cpu_info(); // 获取CPU信息(L1 Caches, L2 Caches, CPU核心数量)
            unsigned int num_threads = BINEScheduler::get().num_threads(); // 当前线程数量: Mac 10个
            BatmanGemm::GemmConfig cfg;
            cfg.weight_format = assembly_utils::map_to_batman_gemm_weight_format(info.weight_format); // 映射Gemm的权重数据格式
            BatmanGemm::WeightFormat arm_gemm_expected_wf = assembly_utils::map_to_batman_gemm_weight_format(
                expected_weight_format);
            BatmanGemm::GemmArgs args(&ci, p.M, p.N, p.K,
                                      p.sections,
                                      p.batches,
                                      p.multis,
                                      p.indirect, act,
                                      static_cast<int>(num_threads),
                                      info.fixed_format, info.fast_mode, info.accumulate, &cfg);
            // TODO(COMPMID-6595): Incorporate info.transpose_b
            switch (a->data_type()) {
                case BIDataType::F32:
                    BI_COMPUTE_RETURN_ERROR_ON_MSG(
                        !(BatmanGemm::has_opt_gemm<float, float, float, BatmanGemm::Nothing>(arm_gemm_expected_wf,
                            args,
                            {})),
                        "We could not find an optimized kernel for F32 input");
                    break;
#ifdef __aarch64__
                case BIDataType::U8:
                case BIDataType::QASYMM8:
                    if (d->data_type() == BIDataType::S32) {
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                            !(BatmanGemm::has_opt_gemm<uint8_t, uint8_t, uint32_t, BatmanGemm::Nothing>(
                                arm_gemm_expected_wf, args,
                                {})),
                            "We could not find an optimized kernel for U8/QASYMM8 input and U32 output");
                    } else if (b->data_type() == BIDataType::QASYMM8_SIGNED) {
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                            !(BatmanGemm::has_opt_gemm<uint8_t, int8_t, uint8_t, BatmanGemm::Requantize32>(
                                arm_gemm_expected_wf,
                                args, {})),
                            "We could not find an optimized kernel for U8 input with S8 weights and U8 output");
                    } else {
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                            !(BatmanGemm::has_opt_gemm<uint8_t, uint8_t, uint8_t, BatmanGemm::Requantize32>(
                                arm_gemm_expected_wf,
                                args, {})),
                            "We could not find an optimized kernel for U8 input and U8 output");
                    }
                    break;
                case BIDataType::S8:
                case BIDataType::QASYMM8_SIGNED:
                    if (d->data_type() == BIDataType::S32) {
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                            !(BatmanGemm::has_opt_gemm<int8_t, int8_t, int32_t, BatmanGemm::Nothing>(
                                arm_gemm_expected_wf, args,
                                {})),
                            "We could not find an optimized kernel for S8/QASYMM8_SIGNED input and S32 output");
                    } else {
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                            !(BatmanGemm::has_opt_gemm<int8_t, int8_t, int8_t, BatmanGemm::Requantize32>(
                                arm_gemm_expected_wf, args,
                                {})),
                            "We could not find an optimized kernel for S8 input and S8 output");
                    }
                    break;
#endif /* __aarch64__ */

#if defined(BI_COMPUTE_ENABLE_BF16)
                case BIDataType::BFLOAT16: {
                    if (d->data_type() == BIDataType::BFLOAT16) {
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                            !(BatmanGemm::has_opt_gemm<bfloat16, bfloat16, bfloat16, BatmanGemm::Nothing>(
                                arm_gemm_expected_wf,
                                args, {})),
                            "We could not find an optimized kernel for BFLOAT16 input and BFLOAT16 output");
                    } else {
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                            !(BatmanGemm::has_opt_gemm<bfloat16, bfloat16, float, BatmanGemm::Nothing>(
                                arm_gemm_expected_wf, args,
                                {})),
                            "We could not find an optimized kernel for BFLOAT16 input and F32 output");
                    }
                    break;
                }
#endif /* defined(BI_COMPUTE_ENABLE_BF16) */

#if defined(ENABLE_FP16_KERNELS)
                case BIDataType::F16:
                    BI_COMPUTE_RETURN_ERROR_ON_MSG(
                        !(BatmanGemm::has_opt_gemm<float16_t, float16_t, float16_t, BatmanGemm::Nothing>(
                            arm_gemm_expected_wf,
                            args, {})),
                        "We could not find an optimized kernel for F16 input and F16 output");
                    break;
#endif /* ENABLE_FP16_KERNELS */
                default:
                    BI_COMPUTE_RETURN_ERROR_ON_MSG(true, "Unsupported type. Could not find a kernel");
                    break;
            }
            expected_weight_format = assembly_utils::map_to_batman_compute_weight_format(arm_gemm_expected_wf);

            return BIStatus{};
        }

        bool BICpuGemmAssemblyDispatch::has_stateless_impl() const {
            return _batman_gemm->has_stateless_impl();
        }

        /**
         * 验证GEMM汇编代码
         * @param a  张量信息a
         * @param b  张量信息b
         * @param c  偏置值信息c
         * @param d  输出结果
         * @param info 汇编信息
         * @return
         */
        BIStatus
        BICpuGemmAssemblyDispatch::validate(const BatmanInfer::BIITensorInfo *a,
                                            const BatmanInfer::BIITensorInfo *b,
                                            const BatmanInfer::BIITensorInfo *c,
                                            const BatmanInfer::BIITensorInfo *d,
                                            const BatmanInfer::cpu::BIAsmGemmInfo &info) {
            // 不需要用汇编信息和偏置值
            BI_COMPUTE_UNUSED(c, info);
            BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(a, b, d);
            BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(a);
            BI_COMPUTE_RETURN_ERROR_ON_CPU_BF16_UNSUPPORTED(a);
            BI_COMPUTE_RETURN_ERROR_ON_MSG(!(info.reshape_b_only_on_first_run),
                                           "Assembly kernel will not be executed when reshape_b_only_on_first_run is false")
            ;

#ifndef __aarch64__
            BI_COMPUTE_RETURN_ERROR_ON_MSG(a->element_size() == 1, "8bit integer types only supported for aarch64");
#endif /* __aarch64__ */
            // 查看数据格式到底对不对
            BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1,
                                                                BIDataType::U8,
                                                                BIDataType::QASYMM8,
                                                                BIDataType::QASYMM8_SIGNED,
                                                                BIDataType::S8,
                                                                BIDataType::BFLOAT16,
                                                                BIDataType::F16, BIDataType::F32);
            BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(
                b, 1, BIDataType::U8, BIDataType::QASYMM8, BIDataType::QASYMM8_SIGNED,
                BIDataType::QSYMM8_PER_CHANNEL,
                BIDataType::S8,
                BIDataType::BFLOAT16, BIDataType::F16, BIDataType::F32);
            if (is_data_type_quantized_per_channel(b->data_type())) {
                BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, BIDataType::QASYMM8_SIGNED, BIDataType::S8);
            } else if (is_fixed_format_fast_math(info.weight_format)) {
                BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(a, BIDataType::F32);
                BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(b, BIDataType::BFLOAT16);
            } else if (!(a->data_type() == BIDataType::QASYMM8 && b->data_type() == BIDataType::QASYMM8_SIGNED)) {
                BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b);
            }
            BI_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == BIDataType::F32 && d->data_type() != BIDataType::F32,
                                           "Only F32 output supported for F32 input");
            BI_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == BIDataType::F16 && d->data_type() != BIDataType::F16,
                                           "Only F16 output supported for F16 input");
            BI_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == BIDataType::BFLOAT16 &&
                                           (d->data_type() != BIDataType::F32 &&
                                               d->data_type() != BIDataType::BFLOAT16),
                                           "Only F32/BFLOAT16 output supported for BFLOAT16 input");
            BI_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == BIDataType::U8 && d->data_type() != BIDataType::U32,
                                           "Only U32 output supported for U8 input");
            BI_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == BIDataType::S8 && d->data_type() != BIDataType::S32,
                                           "Only S32 output supported for S8 input");
            BI_COMPUTE_RETURN_ERROR_ON_MSG(
                a->data_type() == BIDataType::QASYMM8 &&
                (d->data_type() != BIDataType::QASYMM8 && d->data_type() != BIDataType::S32 &&
                    d->data_type() != BIDataType::F32),
                "Only QASYMM8/S32/F32 output supported for QASYMM8 input");
            BatmanInfer::BIWeightFormat expected_weight_format = BatmanInfer::BIWeightFormat::UNSPECIFIED;
            const BIStatus ret = BICpuGemmAssemblyDispatch::has_opt_impl(expected_weight_format, a, b, c, d, info);
            if (bool(ret) && expected_weight_format != BatmanInfer::BIWeightFormat::ANY) {
                // Correctness check: if the format expected by the kernel is
                // not "any", make sure that the one found matches the format
                // intended by the caller.
                BI_COMPUTE_RETURN_ERROR_ON_MSG(
                    (expected_weight_format != info.weight_format),
                    "The format expected by the kernel does not correspond with the one requested by the user.");
            }
            return ret;
        }

        bool BICpuGemmAssemblyDispatch::is_activation_supported(const BIActivationLayerInfo &activation) {
            BatmanGemm::Activation act = assembly_utils::map_to_batman_gemm_activation(activation);
            return act.type != BatmanGemm::Activation::Type::None;
        }

        void BICpuGemmAssemblyDispatch::configure(
            const BIITensorInfo *a, const BIITensorInfo *b, const BIITensorInfo *c, BIITensorInfo *d,
            const BIAsmGemmInfo &info) {
            BI_COMPUTE_ERROR_ON_NULLPTR(a, b, d);
            BatmanGemm::Activation act = assembly_utils::map_to_batman_gemm_activation(
                info.activation_info); // 映射激活函数(从info到GEMM激活函数)

            //If we don't support a combination of data types, silently return: it is the caller's responsibility
            // to check if configure() was successful via is_configured()
            if (!BICpuGemmAssemblyDispatch::validate(a, b, c, d, info)) {
                return;
            }

            switch (a->data_type()) {
                case BIDataType::F32:
                    create_arm_gemm<float, float, float>(_batman_gemm, a, b, c, d, act, info);
                    break;
#ifdef __aarch64__
                case BIDataType::U8:
                case BIDataType::QASYMM8:
                    if (b->data_type() == BIDataType::S8 || b->data_type() == BIDataType::QASYMM8_SIGNED) {
                        if (d->data_type() == BIDataType::F32) {
                            create_arm_gemm_dequant<uint8_t, int8_t, float>(_batman_gemm, a, b, c, d, act, info);
                        } else {
                            create_arm_gemm_quant<uint8_t, int8_t, uint8_t>(_batman_gemm, a, b, c, d, act, info);
                        }
                    } else if (d->data_type() == BIDataType::S32) {
                        create_arm_gemm<uint8_t, uint8_t, uint32_t>(_batman_gemm, a, b, c, d, act, info);
                    } else {
                        create_arm_gemm_quant<uint8_t, uint8_t, uint8_t>(_batman_gemm, a, b, c, d, act, info);
                    }
                    break;
                case BIDataType::S8:
                case BIDataType::QASYMM8_SIGNED:
                    if (d->data_type() == BIDataType::S32) {
                        create_arm_gemm<int8_t, int8_t, int32_t>(_batman_gemm, a, b, c, d, act, info);
                    } else if (d->data_type() == BIDataType::F32) {
                        create_arm_gemm_dequant<int8_t, int8_t, float>(_batman_gemm, a, b, c, d, act, info);
                    } else {
                        create_arm_gemm_quant<int8_t, int8_t, int8_t>(_batman_gemm, a, b, c, d, act, info);
                    }
                    break;
#endif /* __aarch64__ */
#if defined(BI_COMPUTE_ENABLE_BF16)
                case BIDataType::BFLOAT16:
                    if (d->data_type() == BIDataType::BFLOAT16) {
                        create_arm_gemm<bfloat16, bfloat16, bfloat16>(_batman_gemm, a, b, c, d, act, info);
                    } else {
                        create_arm_gemm<bfloat16, bfloat16, float>(_batman_gemm, a, b, c, d, act, info);
                    }
                    break;
#endif /* defined(ARM_COMPUTE_ENABLE_BF16) */
#ifdef ENABLE_FP16_KERNELS
                case BIDataType::F16:
                    create_arm_gemm<float16_t, float16_t, float16_t>(_batman_gemm, a, b, c, d, act, info);
                    break;
#endif /* ENABLE_FP16_KERNELS */
                default:
                    break;
            }
        }

        void BICpuGemmAssemblyDispatch::prepare(BIITensorPack &tensors) {
            BI_COMPUTE_ERROR_ON(_batman_gemm == nullptr);
            _batman_gemm->prepare(tensors);
        }

        bool BICpuGemmAssemblyDispatch::is_configured() const {
            return _batman_gemm && _batman_gemm->is_configured();
        }

        void BICpuGemmAssemblyDispatch::run(BIITensorPack &tensors) {
            BI_COMPUTE_ERROR_ON(_batman_gemm == nullptr);
            _batman_gemm->run(tensors);
        }

        experimental::BIMemoryRequirements BICpuGemmAssemblyDispatch::workspace() const {
            BI_COMPUTE_ERROR_ON(_batman_gemm == nullptr);
            return _batman_gemm->workspace();
        }

        void BICpuGemmAssemblyDispatch::update_quantization_parameters(const BIGEMMLowpOutputStageInfo &output_info,
                                                                       const BIQuantizationInfo &a,
                                                                       const BIQuantizationInfo &b,
                                                                       const bool is_prepared,
                                                                       const bool negated_offsets) {
            BI_COMPUTE_ERROR_ON(_batman_gemm == nullptr);
            _batman_gemm->update_quantization_parameters(output_info, a, b, is_prepared, negated_offsets);
        }

        void BICpuGemmAssemblyDispatch::dynamic_tensor_b_size(const BIITensorInfo *a, const BIITensorInfo *b,
                                                              const BIITensorInfo *d, bool is_gemm) {
            _batman_gemm->update_configure_parameters(a, b, d, is_gemm);
        }
    } // namespace cpu
}
