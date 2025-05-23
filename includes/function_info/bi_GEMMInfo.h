//
// Created by holynova on 2024/12/31.
//

#ifndef BATMANINFER_BI_GEMMINFO_H
#define BATMANINFER_BI_GEMMINFO_H

#include "data/core/core_types.hpp"
#include "function_info/bi_activationLayerInfo.h"

#include <vector>

namespace BatmanInfer {

    class BIITensorInfo;

    /** GEMMLowp output stage type */
    enum class BIGEMMLowpOutputStageType {
        NONE,                     /**< No quantization */
        QUANTIZE_DOWN,            /**< Quantize using an integer multiplication */
        QUANTIZE_DOWN_FIXEDPOINT, /**< Quantize using a fixed point multiplication */
        QUANTIZE_DOWN_FLOAT       /**< Quantize using a floating point multiplication */
    };

    /** GEMMLowp output stage info */
    struct BIGEMMLowpOutputStageInfo {
        BIGEMMLowpOutputStageType type{BIGEMMLowpOutputStageType::NONE}; /**< GEMMLowp output stage type */
        int32_t gemmlowp_offset{0}; /**< GEMMLowp output stage offset used for quantizing to QASYMM8 */
        int32_t gemmlowp_multiplier{
                0};             /**< GEMMLowp output stage multiplier used for quantizing to QASYMM8 */
        int32_t gemmlowp_shift{0};                  /**< GEMMLowp output stage shift used for quantizing to uint8 */
        int32_t gemmlowp_min_bound{
                std::numeric_limits<int32_t>::
                lowest()}; /**< GEMMLowp min value used to saturate down the output result before converting back to QASYMM8 */
        int32_t gemmlowp_max_bound{
                std::numeric_limits<int32_t>::
                max()}; /**< GEMMLowp max value used to saturate down the output result before converting back to QASYMM8 */
        std::vector<int32_t> gemmlowp_multipliers{}; /**< GEMMLowp output stage multiplier used for quantizing to QASYMM8 */
        std::vector<int32_t> gemmlowp_shifts{};      /**< GEMMLowp output stage multiplier used for quantizing to QASYMM8 */
        float gemmlowp_real_multiplier{0}; /**< GEMMLowp output stage real multiplier used for quantizing to QASYMM8 */
        bool is_quantized_per_channel{false}; /**< GEMMLowp quantized per-channel flag */
        BIDataType output_data_type{
                BIDataType::UNKNOWN}; /**< Output tensor data type to use if the output is not initialized */

        bool operator==(const BIGEMMLowpOutputStageInfo &rhs) const {
            return type == rhs.type && gemmlowp_offset == rhs.gemmlowp_offset &&
                   gemmlowp_multiplier == rhs.gemmlowp_multiplier && gemmlowp_shift == rhs.gemmlowp_shift &&
                   gemmlowp_min_bound == rhs.gemmlowp_min_bound && gemmlowp_max_bound == rhs.gemmlowp_max_bound &&
                   gemmlowp_multipliers == rhs.gemmlowp_multipliers && gemmlowp_shifts == rhs.gemmlowp_shifts &&
                   gemmlowp_real_multiplier == rhs.gemmlowp_real_multiplier &&
                   is_quantized_per_channel == rhs.is_quantized_per_channel && output_data_type == rhs.output_data_type;
        }

        bool operator!=(const BIGEMMLowpOutputStageInfo &rhs) const {
            return !(*this == rhs);
        }
    };

    /** GEMM information class. This class stores the necessary information to compute GEMM functions
     *
     * This object also contains the information about how matrix A and matrix B have been reshaped
     *
     */
    class GEMMInfo {
    public:
        /** Default constructor */
        GEMMInfo() noexcept
                : _is_a_reshaped(false),
                  _is_b_reshaped(false),
                  _reshape_b_only_on_first_run(true),
                  _depth_output_gemm3d(0),
                  _reinterpret_input_as_3d(false),
                  _retain_internal_weights(false),
                  _gemmlowp_output_stage(),
                  _fast_math(false),
                  _fp_mixed_precision(false),
                  _broadcast_bias(false),
                  _pretranspose_A(false),
                  _pretranspose_B(false),
                  _activation_info(),
                  _fixed_format(false),
                  _weight_format(BatmanInfer::BIWeightFormat::UNSPECIFIED),
                  _accumulate(false) {
        }

        /** Constructor
         *
         * @param[in] is_a_reshaped               True if the matrix A has been reshaped
         * @param[in] is_b_reshaped               True if the matrix B has been reshaped
         * @param[in] reshape_b_only_on_first_run Reshape matrix B only for the first run
         * @param[in] depth_output_gemm3d         (Optional) Depth (third dimension) of the output tensor to be used with the GEMM3D kernel
         *                                        If 0 the output will not be reinterpreted as 3D. Default 0
         * @param[in] reinterpret_input_as_3d     (Optional) Reinterpret the input as 3D tensor. (i.e. this flag should be set to true when GEMM is used
         *                                        to perform 1x1 convolutions with the NHWC data layout)
         * @param[in] retain_internal_weights     (Optional) Retain the weights tensor from previous run
         * @param[in] gemmlowp_output_stage       (Optional) GEMMLowp Output stage info
         * @param[in] fp_mixed_precision          (Optional) Use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy.
         * @param[in] fast_math                   (Optional) Use a data type of shorter width to improve performance
         * @param[in] broadcast_bias              (Optional) Broadcast the shape of the bias tensor from a vector to a matrix.
         * @param[in] activation_info             (Optional) Activation to apply after the matrix multiplication
         * @param[in] fixed_format                (Optional) Specify the selection of fixed format kernels for variable weights support in GEMM. These kernels expect the weights tensor to be in amemory format that is fixed by the kernel itself. For more information, see arm_compute::WeightFormat.
         * @param[in] weight_format               (Optional) arm_gemm:WeightFormat enumeration requested by the user. Default is arm_compute::WeightFormat::UNSPECIFIED.
         * @param[in] pretranspose_B              (Optional) Pretranspose matrix B (transposition of its lowest 2 dimensions), in addition to and before, any further transformations of B
         * @param[in] accumulate                  (Optional) Whether to accumulate in destination or not
         */
        GEMMInfo(bool is_a_reshaped,
                 bool is_b_reshaped,
                 bool reshape_b_only_on_first_run,
                 int depth_output_gemm3d = 0,
                 bool reinterpret_input_as_3d = false,
                 bool retain_internal_weights = false,
                 BIGEMMLowpOutputStageInfo gemmlowp_output_stage = BIGEMMLowpOutputStageInfo(),
                 bool fp_mixed_precision = false,
                 bool fast_math = false,
                 bool broadcast_bias = false,
                 const BIActivationLayerInfo &activation_info = BIActivationLayerInfo(),
                 bool fixed_format = false,
                 BatmanInfer::BIWeightFormat weight_format = BatmanInfer::BIWeightFormat::UNSPECIFIED,
                 bool pretranspose_B = false,
                 bool accumulate = false) noexcept
                : _is_a_reshaped(is_a_reshaped),
                  _is_b_reshaped(is_b_reshaped),
                  _reshape_b_only_on_first_run(reshape_b_only_on_first_run),
                  _depth_output_gemm3d(depth_output_gemm3d),
                  _reinterpret_input_as_3d(reinterpret_input_as_3d),
                  _retain_internal_weights(retain_internal_weights),
                  _gemmlowp_output_stage(gemmlowp_output_stage),
                  _fast_math(fast_math),
                  _fp_mixed_precision(fp_mixed_precision),
                  _broadcast_bias(broadcast_bias),
                  _pretranspose_A(false),
                  _pretranspose_B(pretranspose_B),
                  _activation_info(activation_info),
                  _fixed_format(fixed_format),
                  _weight_format(weight_format),
                  _accumulate(accumulate) {
        }

        /** Flag which specifies if the matrix A has been reshaped
         *
         * @return True if the matrix A has been reshaped
         */
        bool is_a_reshaped() const {
            return _is_a_reshaped;
        };

        /** Flag which specifies if the matrix B has been reshaped
         *
         * @return True if the matrix B has been reshaped
         */
        bool is_b_reshaped() const {
            return _is_b_reshaped;
        };

        /** 标记位: 用于指定是否仅针对矩阵 B 的第一个进行重塑。
         *
         * @note 当 GEMM 用于加速卷积层时，此标志位可以设置为 TRUE。
         *
         * @return True if the reshaped of matrix B happens only for the first run
         */
        bool reshape_b_only_on_first_run() const {
            return _reshape_b_only_on_first_run;
        };

        /** Depth of the output when GEMM output is reinterpreted as 3D tensor
         *
         * @return the depth of the output tensor
         */
        int depth_output_gemm3d() const {
            return _depth_output_gemm3d;
        };

        /** Flag which specifies if the input tensor has to be reinterpreted as 3D
         *
         * @return True if the input tensor has to be reinterpreted as 3D tensor
         */
        bool reinterpret_input_as_3d() const {
            return _reinterpret_input_as_3d;
        };

        /** Flag which specifies if the weights tensor has to be retained from previous run
         *
         * @return True if the weights tensor has to be retained
         */
        bool retain_internal_weights() const {
            return _retain_internal_weights;
        };

        /** GEMMLowp output stage
         *
         * @return the GEMMLowp output stage info
         */
        BIGEMMLowpOutputStageInfo gemmlowp_output_stage() const {
            return _gemmlowp_output_stage;
        };

        /** Sets GEMMLowp output stage
         *
         * @param[in] output_stage Output stage to set
         */
        void set_gemmlowp_output_stage(BIGEMMLowpOutputStageInfo &output_stage) {
            _gemmlowp_output_stage = output_stage;
        };

        /** Flag which specifies if a wider accumulator should be used.
         *
         * @return True if a wider accumulator has to be used
         */
        bool fp_mixed_precision() const {
            return _fp_mixed_precision;
        };

        /** Flag which specifies if a shorter accumulator to be used.
         *
         * @return True if a shorter accumulator has to be used
         */
        bool fast_math() const {
            return _fast_math;
        };

        /** Set fast math flag
         *
         * @param[in] fast_math Flag to set
         */
        void set_fast_math(bool fast_math) {
            _fast_math = fast_math;
        }

        /** Flag which specifies whether to broadcast the shape of the bias tensor.
         *
         * @return True if the shape of the bias tensor is to be broadcasted.
         */
        bool broadcast_bias() const {
            return _broadcast_bias;
        };

        /** Flag which specifies whether A should be pre-transposed if supported.
         *
         * @return True if A should be pre-transposed else false.
         */
        bool pretranspose_A() const {
            return _pretranspose_A;
        };

        /** Set pre-transpose A flag
         *
         * @param[in] flag Flag to set
         */
        void set_pretranspose_A(bool flag) {
            _pretranspose_A = flag;
        }

        /** Flag which specifies whether b should be pre-transposed if supported.
         * More concretely, the "pre-transpose" is the transposition of the b tensor's lowest 2 dimensions
         * If specified true, this pre-transpose will occur in addition to and before, any further transformations of the b matrix
         *
         * @return True if b should be pre-transposed else false.
         */
        bool pretranspose_B() const {
            return _pretranspose_B;
        };

        /** Set pre-transpose b flag
         *
         * @param[in] flag Flag to set
         */
        void set_pretranspose_B(bool flag) {
            _pretranspose_B = flag;
        }

        /** Activation layer to apply after the matrix multiplication
         *
         * @return ActivationLayerInfo object
         */
        BIActivationLayerInfo activation_info() const {
            return _activation_info;
        }

        /** Set activation layer info
         *
         * @param[in] activation_info ActivationLayerInfo object to set
         */
        void set_activation_info(const BIActivationLayerInfo &activation_info) {
            _activation_info = activation_info;
        }

        /** Flag which specifies if the GEMM operation is running fixed-format kernels.
         *
         * @return True if the GEMM operation is running fixed-format kernel else false.
         */
        bool fixed_format() const {
            return _fixed_format;
        }

        /** Flag which specifies if GEMM should accumulate the result in destination or not.
         *
         * @return True if GEMM is accumulating the result.
         */
        bool accumulate() const {
            return _accumulate;
        }

        /** Set fixed-format flag
         *
         * @param[in] fixed_format sets whether or not to use fixed-format kernels
         */
        void set_fixed_format(bool fixed_format) {
            _fixed_format = fixed_format;
        }

        /** Set accumulate flag
         *
         * @param[in] accumulate sets whether or not to use accumulation
         */
        void set_accumulate(bool accumulate) {
            _accumulate = accumulate;
        }

        BatmanInfer::BIWeightFormat weight_format() const {
            return _weight_format;
        }

        /** Set weight format to be used
         *
         * @param[in] weight_format arm_compute::WeightFormat enumeration
         */
        void set_weight_format(BatmanInfer::BIWeightFormat weight_format) {
            _weight_format = weight_format;
        }

    private:
        bool _is_a_reshaped;
        bool _is_b_reshaped;
        bool _reshape_b_only_on_first_run;
        int _depth_output_gemm3d;
        bool _reinterpret_input_as_3d;
        bool _retain_internal_weights;
        BIGEMMLowpOutputStageInfo _gemmlowp_output_stage;
        bool _fast_math;
        bool _fp_mixed_precision;
        bool _broadcast_bias;
        bool _pretranspose_A;
        bool _pretranspose_B;
        BIActivationLayerInfo _activation_info;
        bool _fixed_format;
        BatmanInfer::BIWeightFormat _weight_format;
        bool _accumulate;
    };


} // namespace BatmanInfer

#endif //BATMANINFER_BI_GEMMINFO_H
