//
// Created by holynova on 2025/1/17.
//

#pragma once

#include "function_info/bi_fullyConnectedLayerInfo.h"
#include "runtime/bi_i_function.hpp"
#include "runtime/bi_i_memory_manager.hpp"
#include "runtime/bi_i_weights_manager.hpp"
#include "runtime/bi_memory_manager_on_demand.hpp"
#include "runtime/neon/functions/bi_ne_transpose.hpp"
#include "runtime/bi_tensor.hpp"

#include <memory>

namespace BatmanInfer {

    namespace weights_transformations {

        /** Basic function to manage the reshape weights generated from @ref NETranspose */
        class BINEFullyConnectedLayerReshapeWeightsManaged : public BIITransformWeights
        {
        public:
            void run() override
            {
                _output.allocator()->allocate();
                _func.run();
                _reshape_run = true;
            }

            void release() override
            {
                _output.allocator()->free();
            }

            BIITensor *get_weights() override
            {
                return &_output;
            }

            uint32_t uid() override
            {
                return _uid;
            }

            void configure(const BIITensor *input)
            {
                _func.configure(input, &_output);
            }

        private:
            static constexpr uint32_t _uid = 0x0;
            BITensor                  _output{};
            BINETranspose             _func{};
        };

    } // namespace weights_transformations

    /** Basic function to compute a Fully Connected layer. This function calls the following kernels:
     *  -# cpu::kernels::CpuIm2ColKernel (called when the input comes from a convolutional layer)
     *  -# @ref NETranspose (if @p are_weights_reshaped is set to false and transpose_weights is set to true ) (called once)
     *  -# @ref NEGEMM or @ref NEGEMMLowpMatrixMultiplyCore (if quantized asymmetric)
     *  -# cpu::kernels::CpuGemmMatrixAdditionKernel or @ref NEGEMMLowpOutputStage (if quantized asymmetric) (if @p biases is not equal to nullptr)
     *
     * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
     */
    class BINEFullyConnectedLayer : public BIIFunction
    {
    public:
        /** Constructor */
        BINEFullyConnectedLayer(std::shared_ptr<BIIMemoryManager> memory_manager, BIIWeightsManager *weights_manager = nullptr);
        BINEFullyConnectedLayer() : BINEFullyConnectedLayer(BIMemoryManagerOnDemand::make_default())
        {
        }
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEFullyConnectedLayer(const BINEFullyConnectedLayer &) = delete;
        /** Prevent instances of this class from being moved (As this class contains pointers) */
        BINEFullyConnectedLayer(BINEFullyConnectedLayer &&) = delete;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEFullyConnectedLayer &operator=(const BINEFullyConnectedLayer &) = delete;
        /** Prevent instances of this class from being moved (As this class contains pointers) */
        BINEFullyConnectedLayer &operator=(BINEFullyConnectedLayer &&) = delete;
        /** Default destructor */
        ~BINEFullyConnectedLayer();
        /** Set the input and output tensors.
         *
         * Valid data layouts:
         * - NHWC
         * - NCHW
         *
         * Valid data type configurations:
         * |src0           |src1               |src2   |dst            |
         * |:--------------|:------------------|:------|:--------------|
         * |F16            |F16                |F16    |F16            |
         * |F32            |F32                |F32    |F32            |
         * |QASYMM8        |QASYMM8            |S32    |QASYMM8        |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
         *
         * @param[in]  input        Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
         * @param[in]  weights      Weights tensor. The weights must be 2 dimensional.
         *                          If this function is called after a Convolution Layer, the (transposed) weights will have as many rows as the product of the first 3 input's dimensions.
         *                          If it is called after another FullyConnected Layer, the (transposed) weights will have as many rows as the input's first dimension.
         *                          Data type supported: Same as @p input.
         * @param[in]  biases       Bias tensor. Can be nullptr. Data type supported: Same as @p weights, S32 if @p weights is QASYMM8/QASYMM8_SIGNED.
         * @param[out] output       Destination tensor. Its shape should be equal to the output of a matrix multiplication between:
         *                          - The output of im2col on the input and the (transposed) 2D weights, if the function is called after a Convolution Layer
         *                          - The input tensor and the (transposed) 2D weights, if the function is called after another FullyConnected Layer.
         *                          Data type supported: Same as @p input.
         * @param[in]  fc_info      (Optional) Fully connected layer additional info
         * @param[in]  weights_info (Optional) Stores neccessary compute information when weights are already reshaped
         */
        void configure(const BIITensor          *input,
                       const BIITensor          *weights,
                       const BIITensor          *biases,
                       BIITensor                *output,
                       BIFullyConnectedLayerInfo fc_info      = BIFullyConnectedLayerInfo(),
                       const BIWeightsInfo      &weights_info = BIWeightsInfo());
        /** Static function to check if given info will lead to a valid configuration of @ref NEFullyConnectedLayer
         *
         * Similar to @ref NEFullyConnectedLayer::configure()
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo      *input,
                                 const BIITensorInfo      *weights,
                                 const BIITensorInfo      *biases,
                                 const BIITensorInfo      *output,
                                 BIFullyConnectedLayerInfo fc_info      = BIFullyConnectedLayerInfo(),
                                 const BIWeightsInfo      &weights_info = BIWeightsInfo());

        /** Static function that queries whether fixed-format kernel exists for a given problem description
         *
         * @param[out] expected_weight_format Format in which weights should be for found fixed format kernel
         * @param[in]  input                  Source tensor
         * @param[in]  weights                Weights tensor.
         * @param[in]  biases                 Bias tensor. Can be nullptr. Data type supported: Same as @p weights, S32 if @p weights is QASYMM8/QASYMM8_SIGNED.
         * @param[in]  output                 Destination tensor
         * @param[in]  fc_info                Fully connected layer additional info
         * @param[in]  weights_info           Describes weights shape
         *
         * @return a status
         */
        static BIStatus has_opt_impl(BatmanInfer::BIWeightFormat     &expected_weight_format,
                                     const BIITensorInfo             *input,
                                     const BIITensorInfo             *weights,
                                     const BIITensorInfo             *biases,
                                     const BIITensorInfo             *output,
                                     const BIFullyConnectedLayerInfo &fc_info,
                                     const BIWeightsInfo             &weights_info);

        //Inherited methods override
        void run() override;
        void prepare() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace BatmanInfer
