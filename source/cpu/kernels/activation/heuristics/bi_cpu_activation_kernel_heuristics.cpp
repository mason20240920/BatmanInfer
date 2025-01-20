//
// Created by Mason on 2025/1/9.
//

#include "cpu/kernels/activation/heuristics/bi_cpu_activation_kernel_heuristics.hpp"

#include "data/core/utils/data_type_utils.hpp"
#include "data/core/bi_vlidate.hpp"
#include "runtime/neon/bi_ne_scheduler.hpp"

#include "common/bi_registers.hpp"
#include "data/core/helpers/bi_window_helpers.hpp"
#include "cpu/kernels/activation/list.hpp"


#include <map>
#include <vector>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace heuristics {
                namespace {
                    bool is_fp16_lut_supported(BatmanInfer::BIActivationLayerInfo::ActivationFunction func) {
                        return func == BatmanInfer::BIActivationLayerInfo::ActivationFunction::LOGISTIC ||
                               func == BatmanInfer::BIActivationLayerInfo::ActivationFunction::TANH;
                    }

                    using KernelList = std::vector<BICpuActivationKernelHeuristics::BIActivationKernel>;
                    using KernelMap = std::map<BIDataType, KernelList>;

                    static const KernelList fp32_kernels = {
                            {"neon_fp32_activation",
                             [](const ActivationDataTypeISASelectorData &data) {
                                 BI_COMPUTE_UNUSED(data);
                                 return true;
                             },
                             REGISTER_FP32_NEON(BatmanInfer::cpu::neon_fp32_activation)},
                    };

                    static const KernelList fp16_kernels = {
                            {"neon_fp16_activation", [](
                                    const ActivationDataTypeISASelectorData &data) { return data.isa.fp16; },
                             REGISTER_FP16_NEON(BatmanInfer::cpu::neon_fp16_activation)},
                    };

                    static const KernelList qasymm8_kernels = {
#ifdef __aarch64__
                            {// Neon LUT implementantion takes precedence
                                    "neon_q8_activation_lut",
                                    [](const ActivationDataTypeISASelectorData &data) {
                                        return data.f != BatmanInfer::BIActivationLayerInfo::ActivationFunction::RELU;
                                    },
                                    REGISTER_Q8_NEON(BatmanInfer::cpu::neon_q8_activation_lut)},
#endif // __aarch64__
                            {"neon_qu8_activation",
                             [](const ActivationDataTypeISASelectorData &data) {
                                 BI_COMPUTE_UNUSED(data);
                                 return true;
                             },
                             REGISTER_QASYMM8_NEON(BatmanInfer::cpu::neon_qasymm8_activation)},
                    };

                    static const KernelList qasymm8_signed_kernels = {
#ifdef __aarch64__
                            {// Neon LUT implementantion takes precedence
                                    "neon_q8_activation_lut",
                                    [](const ActivationDataTypeISASelectorData &data) {
                                        return data.f != BatmanInfer::BIActivationLayerInfo::ActivationFunction::RELU;
                                    },
                                    REGISTER_Q8_NEON(BatmanInfer::cpu::neon_q8_activation_lut)},
#endif // __aarch64__
                            {"neon_qs8_activation",
                             [](const ActivationDataTypeISASelectorData &data) {
                                 BI_COMPUTE_UNUSED(data);
                                 return true;
                             },
                             REGISTER_QASYMM8_SIGNED_NEON(BatmanInfer::cpu::neon_qasymm8_signed_activation)},
                    };

                    static const KernelList qsymm16_kernels = {
                            {"neon_qs16_activation",
                             [](const ActivationDataTypeISASelectorData &data) {
                                 BI_COMPUTE_UNUSED(data);
                                 return true;
                             },
                             REGISTER_QSYMM16_NEON(BatmanInfer::cpu::neon_qsymm16_activation)},
                    };

                    static const KernelMap kernels = {{BIDataType::F32,            fp32_kernels},
                                                      {BIDataType::F16,            fp16_kernels},
                                                      {BIDataType::QASYMM8,        qasymm8_kernels},
                                                      {BIDataType::QASYMM8_SIGNED, qasymm8_signed_kernels},
                                                      {BIDataType::QSYMM16,        qsymm16_kernels}};

                    /**
                     * 查找第一个大于输入值的元素的索引
                     * @note 二分查找对于小数组并没有太大优势，
                     *       因此我们保持实现简单。
                     *
                     * @param arr 输入数组
                     * @param len 输入数组的长度
                     * @param x   用于比较的元素
                     * @return 找到的索引
                     */
                    size_t find_ind_lte_elm(const size_t *arr, size_t len, size_t x) {
                        BI_COMPUTE_ERROR_ON_NULLPTR(arr);
                        for (size_t i = 0; i < len; ++i) {
                            if (x <= arr[i]) {
                                return i;
                            }
                        }

                        return len - 1;
                    }


                    size_t calculate_mws(const CPUModel cpu_model,
                                         BIDataType dtype, const
                                         BatmanInfer::BIActivationLayerInfo &act_info,
                                         size_t problem_size) {
                        // This number is loosely chosen as threading overhead in each platform varies wildly.
                        size_t mws = 1529;

                        if (cpu_model == CPUModel::V1) {
                            // If max_threads is smaller than the number of threads suggested in the heuristics,
                            //
                            const size_t max_threads = BINEScheduler::get().num_threads();

                            constexpr int32_t compute_heavy_arr_fp32_len = 26;
                            static const size_t compute_heavy_arr_fp32[2][compute_heavy_arr_fp32_len] = {
                                    {2000, 4000, 5000, 6000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000,
                                                                                                                       80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 2000000},
                                    {1,    2,    3,    4,    5,    6,    7,     9,     12,    14,    15,    18,    20, 22,    25,    29,     36,     43,     48,     53,     57,     58,     59,     60,     62,      max_threads}};

                            constexpr int32_t compute_light_arr_fp32_len = 20;
                            static const size_t compute_light_arr_fp32[2][compute_light_arr_fp32_len] = {
                                    {30000, 40000, 50000, 70000, 80000, 90000, 100000, 200000, 300000, 400000,
                                                                                                           500000, 600000, 700000, 900000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000},
                                    {1,     2,     3,     4,     6,     8,     10,     13,     15,     18, 21,     23,     24,     25,     30,      38,      45,      53,      60,      max_threads}};

                            constexpr int32_t compute_heavy_arr_fp16_len = 24;
                            static const size_t compute_heavy_arr_fp16[2][compute_heavy_arr_fp16_len] = {
                                    {10000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000,
                                                                                                                         500000, 800000, 900000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 8000000, 10000000, 20000000},
                                    {1,     2,     3,     5,     6,     7,     8,     10,    13,     17,     20,     23, 25,     28,     32,     37,      43,      49,      55,      58,      60,      61,      62,       max_threads}};

                            constexpr int32_t compute_light_arr_fp16_len = 20;
                            static const size_t compute_light_arr_fp16[2][compute_light_arr_fp16_len] = {
                                    {30000, 40000, 50000, 70000, 80000, 90000, 100000, 200000, 300000, 400000,
                                                                                                           500000, 600000, 700000, 900000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000},
                                    {1,     2,     3,     4,     6,     8,     10,     13,     15,     18, 21,     23,     24,     25,     30,      38,      45,      53,      60,      max_threads}};

                            constexpr int32_t s8_arr_len = 24;
                            static const size_t s8_arr[2][s8_arr_len] = {
                                    {7000, 8000, 9000, 10000, 20000, 30000, 40000, 60000, 70000, 90000, 100000, 200000,
                                                                                                                    300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 2000000, 3000000, 8000000, 9000000},
                                    {1,    2,    3,    4,     6,     7,     10,    11,    13,    15,    19,     23, 26,     31,     37,     40,     44,     48,     52,     54,      58,      61,      62,      max_threads}};

                            const size_t dtype_len = data_size_from_type(dtype);

                            const size_t *size_arr = nullptr;
                            const size_t *nthread_arr = nullptr;
                            size_t arr_len = 0;

                            switch (act_info.activation()) {
                                case BatmanInfer::BIActivationLayerInfo::ActivationFunction::LOGISTIC:
                                case BatmanInfer::BIActivationLayerInfo::ActivationFunction::SWISH:
                                case BatmanInfer::BIActivationLayerInfo::ActivationFunction::ELU:
                                case BatmanInfer::BIActivationLayerInfo::ActivationFunction::GELU:
                                case BatmanInfer::BIActivationLayerInfo::ActivationFunction::SOFT_RELU:
                                case BatmanInfer::BIActivationLayerInfo::ActivationFunction::TANH: {
                                    switch (dtype_len) {
                                        case 4:
                                            size_arr = &compute_heavy_arr_fp32[0][0];
                                            nthread_arr = &compute_heavy_arr_fp32[1][0];
                                            arr_len = compute_heavy_arr_fp32_len;
                                            break;
                                        case 2:
                                            size_arr = &compute_heavy_arr_fp16[0][0];
                                            nthread_arr = &compute_heavy_arr_fp16[1][0];
                                            arr_len = compute_heavy_arr_fp16_len;
                                            break;
                                        case 1:
                                        default:
                                            size_arr = &s8_arr[0][0];
                                            nthread_arr = &s8_arr[1][0];
                                            arr_len = s8_arr_len;
                                            break;
                                    }
                                    break;
                                }
                                default: {
                                    switch (dtype_len) {
                                        case 4:
                                            size_arr = &compute_light_arr_fp32[0][0];
                                            nthread_arr = &compute_light_arr_fp32[1][0];
                                            arr_len = compute_light_arr_fp32_len;
                                            break;
                                        case 2:
                                            size_arr = &compute_light_arr_fp16[0][0];
                                            nthread_arr = &compute_light_arr_fp16[1][0];
                                            arr_len = compute_light_arr_fp16_len;
                                            break;
                                        case 1:
                                        default:
                                            size_arr = &s8_arr[0][0];
                                            nthread_arr = &s8_arr[1][0];
                                            arr_len = s8_arr_len;
                                            break;
                                    }
                                    break;
                                }
                            }

                            const size_t ind = find_ind_lte_elm(size_arr, arr_len, problem_size);
                            const size_t nthreads = std::min(nthread_arr[ind], max_threads);
                            mws = (problem_size + nthreads - 1) / nthreads;
                        }

                        return mws;
                    }

                } // namespace

                BICpuActivationKernelHeuristics::BICpuActivationKernelHeuristics(const BIITensorInfo *src,
                                                                                 const BIITensorInfo *dst,
                                                                                 const BIActivationLayerInfo &activation_info) {
                    BI_COMPUTE_UNUSED(dst);

                    // 设置内核
                    const BIDataType dtype = src->data_type();
                    ActivationDataTypeISASelectorData selector{dtype,
                                                               CPUInfo::get().get_cpu_model(),
                                                               CPUInfo::get().get_isa(),
                                                               activation_info.activation()};
                    const CPUModel cpu_model = CPUInfo::get().get_cpu_model();
                    choose_kernel(selector);

                    // 设置窗口和调度提示
                    int split_dim;
                    std::tie(_window, split_dim) = calculate_squashed_or_max_window(*src);

                    // 使用 SME 内核在 Y 维度折叠窗口
                    if (std::string(_kernel->name) == "sme2_fp32_logistic")
                        _window = _window.collapse(_window, BIWindow::DimY);

                    _hint = BIIScheduler::Hints(split_dim);

                    // 设置最小的工作负载
                    if (split_dim == BIWindow::DimX)
                        _mws = calculate_mws(cpu_model,
                                             src->data_type(),
                                             activation_info.activation(),
                                             src->tensor_shape().x());
                }

                size_t BICpuActivationKernelHeuristics::mws() const {
                    return _mws;
                }

                const BIWindow &BICpuActivationKernelHeuristics::window() const {
                    return _window;
                }

                const BICpuActivationKernelHeuristics::BIActivationKernel *BICpuActivationKernelHeuristics::kernel() {
                    return _kernel;
                }

                const BIIScheduler::Hints &BICpuActivationKernelHeuristics::scheduler_hint() const {
                    return _hint;
                }

                void BICpuActivationKernelHeuristics::choose_kernel(ActivationDataTypeISASelectorData &selector) {
                    const auto &klist = kernels.find(selector.dt);
                    if (klist == kernels.end())
                        return;
                    for (const auto &uk: klist->second)
                        if (uk.is_selected(selector) && uk.ukernel != nullptr) {
                            _kernel = &uk;
                            return;
                        }
                }
            }
        }
    }
}