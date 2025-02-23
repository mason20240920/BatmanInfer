//
// Created by Mason on 2025/1/13.
//

#include <cpu/kernels/assembly/bi_arm_gemm.hpp>
#include "kernel_weight_format.hpp"

#include <cstdint>
#include <functional>
#include <iostream>

namespace BatmanGemm {
    /* Structure describing an implementation.  For each supported combination
 * of types, a static list of these structures is built up to describe the
 * implementations available.
 */
    template<typename Tlop, typename Trop, typename Tret, class OutputStage = Nothing>
    struct GemmImplementation {
        const GemmMethod method;
        const char *name;
        const KernelWeightFormat kernel_weight_format = KernelWeightFormat::NON_FIXED;
        std::function<bool(const GemmArgs &, const OutputStage &)> is_supported = {};
        std::function<uint64_t(const GemmArgs &, const OutputStage &)> cycle_estimate = {};
        std::function<BIGemmCommon<Tlop, Trop, Tret> *(
                const GemmArgs &, const OutputStage &)>
                instantiate = {};

        bool do_is_supported(const GemmArgs &args, const OutputStage &os) const {
            // Check supplied is_supported() function first.
            if (is_supported != nullptr && !is_supported(args, os)) {
                return false;
            }

            // Check weight format is appropriate.
            if (args._fixed_format == false) {
                // Can't return a fixed format kernel if we weren't asked for one.
                return (kernel_weight_format == KernelWeightFormat::NON_FIXED);
            } else {
                // Fixed format kernel requested: if this is a non-fixed format kernel we can't use it.
                if (kernel_weight_format == KernelWeightFormat::NON_FIXED) {
                    return false;
                }

                // If there's no config, or the config says ANY then this one is OK.
                if (!args._cfg || args._cfg->weight_format == WeightFormat::ANY) {
                    return true;
                }

                // If we get here it means there is a config and it specifies a format.  Check it matches this kernel.
                // NOTE: this will execute SVE instructions if it's an SVE kernel, so it's important that is_supported()
                // was called above first.
                return (args._cfg->weight_format == get_weight_format(kernel_weight_format, sizeof(Tlop)));
            }
        }

        uint64_t do_cycle_estimate(const GemmArgs &args, const OutputStage &os) const {
            if (cycle_estimate != nullptr) {
                return cycle_estimate(args, os);
            } else {
                return 0;
            }
        }

        BIGemmCommon<Tlop, Trop, Tret> *do_instantiate(const GemmArgs &args, const OutputStage &os) const {
            return instantiate(args, os);
        }

        static GemmImplementation with_estimate(GemmMethod m, const char *n,
                                                std::function<bool(const GemmArgs &, const OutputStage &)> is_supported,
                                                std::function<uint64_t(const GemmArgs &,
                                                                       const OutputStage &)> cycle_estimate,
                                                std::function<BIGemmCommon<Tlop, Trop, Tret> *(const GemmArgs &,
                                                                                               const OutputStage &)

                                                > instantiate) {
            GemmImplementation impl(m, n);

            impl.is_supported = is_supported;
            impl.cycle_estimate = cycle_estimate;
            impl.instantiate = instantiate;

            return impl;
        }

        GemmImplementation(const GemmImplementation &) = default;

        GemmImplementation &operator=(const GemmImplementation &) = default;

        GemmImplementation(GemmMethod m, const char *n) : method(m), name(n) {}

        GemmImplementation(GemmMethod m, const char *n,
                           std::function<bool(const GemmArgs &, const OutputStage &)> is_supported,
                           std::function<bool(const GemmArgs &, const OutputStage &)> is_recommended,
                           std::function<BIGemmCommon<Tlop, Trop, Tret> *(const GemmArgs &, const OutputStage &)

                           > instantiate) :

                method(m), name(n), is_supported(is_supported),
                cycle_estimate([is_recommended](const GemmArgs &args, const OutputStage &os) {
                                   return (is_recommended == nullptr) ? 0 : (is_recommended(args, os) ? 0 : UINT64_MAX);
                               }

                ),
                instantiate(instantiate) {}

        GemmImplementation(GemmMethod m, const char *n, KernelWeightFormat kwf,
                           std::function<bool(const GemmArgs &, const OutputStage &)> is_supported,
                           std::function<bool(const GemmArgs &, const OutputStage &)> is_recommended,
                           std::function<BIGemmCommon<Tlop, Trop, Tret> *(const GemmArgs &, const OutputStage &)

                           > instantiate) :

                method(m), name(n), kernel_weight_format(kwf), is_supported(is_supported),
                cycle_estimate([is_recommended](const GemmArgs &args, const OutputStage &os) {
                                   return (is_recommended == nullptr) ? 0 : (is_recommended(args, os) ? 0 : UINT64_MAX);
                               }

                ),
                instantiate(instantiate) {}
    };

/* Slightly different version of above for straightforward GEMMs with no
 * output stage, so the std::functions there don't have to deal with the
 * unnecessary second argument.  */
    template<typename Tlop, typename Trop, typename Tret>
    struct GemmImplementation<Tlop, Trop, Tret, Nothing> {
        const GemmMethod method;
        const char *name;
        const KernelWeightFormat kernel_weight_format = KernelWeightFormat::NON_FIXED;
        std::function<bool(const GemmArgs &)> is_supported = {};
        std::function<uint64_t(const GemmArgs &)> cycle_estimate = {};
        std::function<BIGemmCommon<Tlop, Trop, Tret> *(
                const GemmArgs &)>
                instantiate = {};

        bool do_is_supported(const GemmArgs &args, const Nothing &) const {
            // Check supplied is_supported() function first.
            if (is_supported != nullptr && !is_supported(args)) {
                return false;
            }

            // Check weight format is appropriate.
            if (args._fixed_format == false) {
                // Can't return a fixed format kernel if we weren't asked for one.
                return (kernel_weight_format == KernelWeightFormat::NON_FIXED);
            } else {
                // Fixed format kernel requested: if this is a non-fixed format kernel we can't use it.
                if (kernel_weight_format == KernelWeightFormat::NON_FIXED) {
                    return false;
                }

                // If there's no config, or the config says ANY then this one is OK.
                if (!args._cfg || args._cfg->weight_format == WeightFormat::ANY) {
                    return true;
                }

                // If we get here it means there is a config and it specifies a format.  Check it matches this kernel.
                // NOTE: this will execute SVE instructions if it's an SVE kernel, so it's important that is_supported()
                // was called above first.
                return (args._cfg->weight_format == get_weight_format(kernel_weight_format, sizeof(Tlop)));
            }
        }

        uint64_t do_cycle_estimate(const GemmArgs &args, const Nothing &) const {
            if (cycle_estimate != nullptr) {
                return cycle_estimate(args);
            } else {
                return 0;
            }
        }

        BIGemmCommon<Tlop, Trop, Tret> *do_instantiate(const GemmArgs &args, const Nothing &) const {
            return instantiate(args);
        }

        static GemmImplementation with_estimate(GemmMethod m, const char *n,
                                                std::function<bool(const GemmArgs &)> is_supported,
                                                std::function<uint64_t(const GemmArgs &)> cycle_estimate,
                                                std::function<BIGemmCommon<Tlop, Trop, Tret> *(const GemmArgs &)

                                                > instantiate) {
            GemmImplementation impl(m, n);

            impl.is_supported = is_supported;
            impl.cycle_estimate = cycle_estimate;
            impl.instantiate = instantiate;

            return impl;
        }

        static GemmImplementation with_estimate(GemmMethod m, const char *n, KernelWeightFormat f,
                                                std::function<bool(const GemmArgs &)> is_supported,
                                                std::function<uint64_t(const GemmArgs &)> cycle_estimate,
                                                std::function<BIGemmCommon<Tlop, Trop, Tret> *(const GemmArgs &)

                                                > instantiate) {
            GemmImplementation impl(m, n, f);

            impl.is_supported = is_supported;
            impl.cycle_estimate = cycle_estimate;
            impl.instantiate = instantiate;

            return impl;
        }

        GemmImplementation(const GemmImplementation &) = default;

        GemmImplementation &operator=(const GemmImplementation &) = default;

        GemmImplementation(GemmMethod m, const char *n, KernelWeightFormat f = KernelWeightFormat::NON_FIXED) : method(
                m), name(n), kernel_weight_format(f) {}

        GemmImplementation(GemmMethod m, const char *n,
                           std::function<bool(const GemmArgs &)> is_supported,
                           std::function<bool(const GemmArgs &)> is_recommended,
                           std::function<BIGemmCommon<Tlop, Trop, Tret> *(const GemmArgs &)

                           > instantiate) :
                method(m), name(n), is_supported(is_supported),
                cycle_estimate([is_recommended](const GemmArgs &args)
                                       -> uint64_t {
                    return (is_recommended == nullptr) ? 0 : (is_recommended(args) ? 0 : UINT64_MAX);
                }),
                instantiate(instantiate) {}

        GemmImplementation(GemmMethod m, const char *n, KernelWeightFormat kwf,
                           std::function<bool(const GemmArgs &)> is_supported,
                           std::function<bool(const GemmArgs &)> is_recommended,
                           std::function<BIGemmCommon<Tlop, Trop, Tret> *(const GemmArgs &)

                           > instantiate) :
                method(m), name(n), kernel_weight_format(kwf), is_supported(is_supported),
                cycle_estimate([is_recommended](const GemmArgs &args)
                                       -> uint64_t {
                    return (is_recommended == nullptr) ? 0 : (is_recommended(args) ? 0 : UINT64_MAX);
                }),
                instantiate(instantiate) {}
    };

/* Provides the list of implementation descriptors which is processed by the
 * other functions.
 *
 * A specialised version is provided for each supported combination of types.
 * The end of the list is indicated by a sentinel descriptor with
 * method==GemmMethod::DEFAULT.  */
    template<typename Tlop, typename Trop, typename Tret, class OutputStage = Nothing>
    const GemmImplementation<Tlop, Trop, Tret, OutputStage> *gemm_implementation_list();


    /**
     * 用给定的参数选择一个GEMM的实现汇编代码(其中用评估function进行运行计算)
     *
     * 本逻辑从候选列表中选取满足以下条件的实现方法：
     * 1. 支持请求的问题参数
     * 2. 匹配提供的筛选条件（方法类型和/或名称字符串匹配）
     * 3. 提供最低周期估算值（特殊值0表示立即选用该方案）
     *
     * 若没有方法满足参数要求并通过筛选：
     * - 返回false
     * - 不修改提供的指针引用
     *
     * @tparam Tlop float16_t
     * @tparam Trop float16_t
     * @tparam Tret float16_t
     * @tparam OutputStage
     * @param args 参数
     * @param os 输出策略
     * @param impl 实现的函数
     * @return
     */
    template<typename Tlop, typename Trop, typename Tret, class OutputStage>
    bool find_implementation(const GemmArgs &args, const OutputStage &os,
                             const GemmImplementation<Tlop, Trop, Tret, OutputStage> *&impl) {
        auto gemms = gemm_implementation_list<Tlop, Trop, Tret, OutputStage>(); // 获取汇编的函数数组指针
        const GemmConfig *cfg = args._cfg;

        const GemmImplementation<Tlop, Trop, Tret, OutputStage> *saved_impl = nullptr;
        uint64_t best_estimate = 0;

        // 根据评估函数给出效率最高的汇编内核Kernel
        for (const GemmImplementation<Tlop, Trop, Tret, OutputStage> *i = gemms;
             i->method != GemmMethod::DEFAULT; i++) {
            /* Skip if this implementation doesn't support these args. */
            if (!i->do_is_supported(args, os)) {
                continue;
            }

            /* Skip if a specific method is requested and this is a different one. */
            if (cfg && cfg->method != GemmMethod::DEFAULT && i->method != cfg->method) {
                continue;
            }

            /* Skip if a filter is to be applied and it doesn't match. */
            if (cfg && cfg->filter != "" && !strstr(i->name, cfg->filter.c_str())) {
                continue;
            }

            /* Test the cycle estimate */
            uint64_t estimate = i->do_cycle_estimate(args, os);

            /* Short circuit - if the estimate is zero, return this one immediately. */
            if (estimate == 0) {
                impl = i;
                return true;
            }

            /* Otherwise, remember this is our best so far if we don't yet have
             * a valid candidate, or we beat the estimate.  */
            if ((saved_impl == nullptr) || (estimate < best_estimate)) {
                saved_impl = i;
                best_estimate = estimate;
            }
        }

        /* Return whichever method gave the best estimate. */
        if (saved_impl != nullptr) {
            impl = saved_impl;
            return true;
        }

        return false;
    }

    template<typename Tlop, typename Trop, typename Tret, class OutputStage>
    std::vector<KernelDescription> get_compatible_kernels(const GemmArgs &args, const OutputStage &os) {
        std::vector<KernelDescription> res;

        /* Find out what the default implementation in so we can set the flag accordingly later. */
        const GemmImplementation<Tlop, Trop, Tret, OutputStage> *default_impl;
        find_implementation(args, os, default_impl);

        auto gemms = gemm_implementation_list<Tlop, Trop, Tret, OutputStage>();

        for (const GemmImplementation<Tlop, Trop, Tret, OutputStage> *i = gemms;
             i->method != GemmMethod::DEFAULT; i++) {
            /* Check that this implementation supports the presented problem. */

            if (!i->do_is_supported(args, os)) {
                continue;
            }

            res.push_back(KernelDescription(i->method, i->name, i == default_impl, i->do_cycle_estimate(args, os)));
        }

        return res;
    }

    template<typename Tlop, typename Trop, typename Tret, class OutputStage>
    bool has_opt_gemm(WeightFormat &wf, const GemmArgs &args, const OutputStage &os) {
        const GemmImplementation<Tlop, Trop, Tret, OutputStage> *impl;
        const bool success = find_implementation<Tlop, Trop, Tret, OutputStage>(args, os, impl);
        // TODO [!!!Important] check your gemm best shape for calculating
        if (success)
            wf = UniqueGemmCommon<Tlop, Trop, Tret>(impl->do_instantiate(args, os))->get_config().weight_format;
        return success;
    }

    template<typename Tlop, typename Trop, typename Tret, class OutputStage>
    UniqueGemmCommon<Tlop, Trop, Tret> gemm(const GemmArgs &args, const OutputStage &os) {
        const GemmImplementation<Tlop, Trop, Tret, OutputStage> *impl;

        if (find_implementation<Tlop, Trop, Tret, OutputStage>(args, os, impl)) {
            return UniqueGemmCommon<Tlop, Trop, Tret>(impl->do_instantiate(args, os));
        }

        return UniqueGemmCommon<Tlop, Trop, Tret>(nullptr);
    }

    template<typename Tlop, typename Trop, typename Tret, class OutputStage>
    KernelDescription get_gemm_method(const GemmArgs &args, const OutputStage &os) {
        const GemmImplementation<Tlop, Trop, Tret, OutputStage> *impl;

        if (find_implementation<Tlop, Trop, Tret>(args, os, impl)) {
            return KernelDescription(impl->method, impl->name);
        }

        /* This shouldn't happen - there should always be at least one valid implementation. */
        return KernelDescription();
    }
}