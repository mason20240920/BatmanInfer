//
// Created by Mason on 2025/1/5.
//

#ifndef BATMANINFER_CPU_GEMM_ASSEMBLY_DISPATCH_HPP
#define BATMANINFER_CPU_GEMM_ASSEMBLY_DISPATCH_HPP

#include "function_info/bi_activationLayerInfo.h"
#include "function_info/bi_GEMMInfo.h"

#include "data/core/common/bi_core_common_macros.hpp"
#include "cpu/bi_i_cpu_operator.hpp"
#include "cpu/kernels/assembly/bi_arm_gemm.hpp"

namespace BatmanInfer {
    namespace cpu {
        /**
         * @brief 卷积方法: 汇编Gemm接口
         */
        enum class BIAsmConvMethod {
            Im2Col,
            Indirect,
            Conv
        };

        struct BIAsmGemmInfo {
            BIAsmConvMethod             method{BIAsmConvMethod::Im2Col};
            BIPadStrideInfo             ps_info{};
            BIActivationLayerInfo       activation_info{};
            BIGEMMLowpOutputStageInfo   output_stage{};
            bool                        negated_offsets{true};
            bool                        reinterpret_input_as_3d{false};
            bool                        depth_output_gemm3d{false};
            int64_t                     padding_top{0};
            int64_t                     padding_left{0};
            float                       padding_value{0.f};
            bool                        fast_mode{false};
            bool                        fixed_format{false};
            BatmanInfer::BIWeightFormat weight_format{BatmanInfer::BIWeightFormat::UNSPECIFIED};
            bool                        reshape_b_only_on_first_run{true};
            bool                        accumulate{false};
            /** Whether we want to perform an additional transpose of b before passing it to gemm or pretranspose_B_array
             * @note This transpose b operation is also considered a form of "reshape" or "transform", so should be counted for
             *       by the reshape_b_only_on_first_run flag
             * @note This flag will be silently ignored (assumed to be false) when the weight_format is a fixed format. Because
             *       fixed format kernels do not accept weights (B) with any prior transformations
             */
            bool                        transpose_b{false};
        };

        /**
         * @brief 汇编内核粘合代码
         */
        class BICpuGemmAssemblyDispatch : public BIICpuOperator {
        public:
            /** Constructor */
            BICpuGemmAssemblyDispatch();

            /** Default destructor */
            ~BICpuGemmAssemblyDispatch() = default;

            BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuGemmAssemblyDispatch);

            class IFallback {
            public:
                virtual void run(BIITensorPack &tensors) = 0;

                virtual void prepare(BIITensorPack &tensors) = 0;

                virtual experimental::BIMemoryRequirements workspace() const = 0;

                virtual bool is_configured() const = 0;

                virtual bool isVarWeightsKernel() const = 0;

                virtual void update_quantization_parameters(const BIGEMMLowpOutputStageInfo &,
                                                            const BIQuantizationInfo &,
                                                            const BIQuantizationInfo &,
                                                            const bool,
                                                            const bool) = 0;

                virtual ~IFallback() = default;

                virtual bool has_stateless_impl() const = 0;
            };

        public:
            /**
             * @brief 如果支持，则创建一个 Compute Library 函数，否则回退到 arm_gemm 函数。
              *
              * @note 配置 "batches"
              * @p a, @p b 和 @p d 的形状排列如下：
              *     最低维度 <-> 最高维度
              * a: [K, M, Batch, Multi]
              * b: [N, K, Multi]
              * d: [N, M, Batch, Multi]
              *
              * "Batch" 表示 "Batch" 个 MxK 的张量 a 的切片与单个 KxN 的 b 切片相乘。
              * "Multi" 表示 "Multi" 次 a 与 b 的独立乘法。
              *
              * 例如，以下是一些输入形状配置的示例：
              *
              * (1) 普通的二维 GEMM
              * a: [K=3, M=4]
              * b: [N=5, K=3]
              * d: [N=5, M=4]
              *
              * (2) 共享 b 的 a 的批次（例如基于 GEMM 的批量卷积，其中 b 是共享的）
              * a: [K=3, M=4, Batch=9]
              * b: [N=5, K=3]
              * d: [N=5, M=4, Batch=9]
              *
              * (3) 独立 GEMM 的 "批次"（例如批量矩阵乘法）
              * a: [K=3, M=4, Batch=1, Multi=7]
              * b: [N=5, K=3, Multi=7]
              * d: [N=5, M=4, Batch=1, Multi=7]
              *
              * (4) 独立 GEMM 的 "批次"，其中 b 也是共享的
              * a: [K=3, M=4, Batch=4, Multi=7]
              * b: [N=5, K=3, Multi=7]
              * d: [N=5, M=4, Batch=4, Multi=7]
              *
             * @param a
             * @param b
             * @param c
             * @param d
             * @param info
             */
            void configure(
                    const BIITensorInfo *a,
                    const BIITensorInfo *b,
                    const BIITensorInfo *c,
                    BIITensorInfo *d,
                    const BIAsmGemmInfo &info);

            /** 指示此函数是否可以用于处理给定的参数。
 *
 * @param[in] a    输入张量信息（矩阵 A）
 * @param[in] b    输入张量信息（矩阵 B）
 * @param[in] c    输入张量信息（矩阵 C），用于传递量化计算的偏置
 * @param[in] d    输出张量，用于存储矩阵乘法的结果。支持的数据类型与 @p input0 相同。
 * @param[in] info GEMM 元数据
 *
 * @return 一个状态。
 */
            static BIStatus validate(const BIITensorInfo *a,
                                     const BIITensorInfo *b,
                                     const BIITensorInfo *c,
                                     const BIITensorInfo *d,
                                     const BIAsmGemmInfo &info);

/** 指示是否存在可以用于处理给定参数的最优汇编实现。
 *
 * 此方法的用途与 @ref NEGEMMConvolutionLayer::has_opt_impl 相同，
 * 唯一的区别是需要通过参数 info 传递 BatmanInfer::WeightFormat 的值。
 *
 * @return 一个状态。
 */
            static BIStatus has_opt_impl(BatmanInfer::BIWeightFormat &weight_format,
                                         const BIITensorInfo *a,
                                         const BIITensorInfo *b,
                                         const BIITensorInfo *c,
                                         const BIITensorInfo *d,
                                         const BIAsmGemmInfo &info);


            /** 检查是否支持无状态实现
             *
             * 到目前为止，已实现无状态的 arm_gemm 内核是那些不需要任何工作空间的内核。
             * 一旦所有内核都实现了无状态，我们可以通过始终返回 true 来弃用它，并最终完全删除它。
             *
             * @return 如果支持无状态执行，则返回 true，否则返回 false。
             */
            bool has_stateless_impl() const;

            /** 检查 gemm 汇编调度器是否支持激活功能
             *
             * @param[in] activation 要检查的激活功能
             *
             * @return 如果支持激活，则返回 true，否则返回 false。
             */
            static bool is_activation_supported(const BIActivationLayerInfo &activation);

            /** 函数是否成功配置？
             *
             * @return 如果函数已配置并准备运行，则返回 true。
             */
            bool is_configured() const;

            /** 指示卷积是否在可变权重模式下执行。
             *
             * 类似于 @ref CpuGemm::isVarWeightsKernel。
             */
            bool isVarWeightsKernel() const {
                return _batman_gemm && _batman_gemm->isVarWeightsKernel();
            }

            // 继承方法覆写
            void prepare(BatmanInfer::BIITensorPack &constants) override;

            // 运行工具
            void run(BatmanInfer::BIITensorPack &tensors) override;

            experimental::BIMemoryRequirements workspace() const override;

        private:
            /**
             * @brief 用于 arm_gemm 回退的接口
             */
            std::unique_ptr<IFallback> _batman_gemm;
        };
    }
}

#endif //BATMANINFER_CPU_GEMM_ASSEMBLY_DISPATCH_HPP
