//
// Created by Mason on 2024/11/14.
//

#include <layer/detail/gemm.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <data/tensor_util.hpp>
#include <omp.h>

namespace BatmanInfer {
    InferStatus GemmLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                   std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the gemm layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the gemm "
                       << "layer do not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        if (weights_.empty()) {
            LOG(ERROR) << "The number of kernel matrix in the gemm layer should "
                          "be greater than zero";
            return InferStatus::bInferFailedWeightParameterError;
        }

        // 获取batch size
        const uint32_t batch_size = inputs.size();

        for (uint32_t i = 0; i < batch_size; ++i) {
            const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
            std::shared_ptr<Tensor<float>> &output = outputs.at(i);

            if (input == nullptr || input->empty()) {
                LOG(ERROR) << "The input tensor is null or empty";
                return InferStatus::bInferFailedInputEmpty;
            }

            if (output == nullptr || output->empty()) {
                output = std::make_shared<Tensor<float>>(1, gemm_height_, 1);
            }

            // 确保输出的尺寸正确
            CHECK(output->rows() == gemm_height_ && output->cols() == 1)
                 << "The output tensor size is incorrect";

            // 使用OpenMP 并行化矩阵乘法和加法
#pragma omp parallel for
            for (int j = 0; j < gemm_height_; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < gemm_width_; ++k) {
                    sum += input->at(0, 0, k) * weights_.at(j * gemm_width_ + k);
                }
                output->at(0, j, 0) = alpha_ * sum + beta_ * bias_.at(j);
            }
        }

        return InferStatus::bInferSuccess;
    }

    ParseParameterAttrStatus GemmLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                    std::shared_ptr<Layer> &gemm_layer) {
        CHECK(op != nullptr) << "Convolution operator is nullptr";
        const std::map <std::string, std::shared_ptr<RuntimeParameter>> &params = op->params;

        if (params.find("alpha") == params.end()) {
            LOG(ERROR) << "Can not find the alpha parameter";
            return ParseParameterAttrStatus::bParameterMissingAlpha;
        }

        auto alpha_param = std::dynamic_pointer_cast<RuntimeParameterFloat>(params.at("alpha"));
        if (alpha_param == nullptr) {
            LOG(ERROR) << "Can not find the alpha parameter";
            return ParseParameterAttrStatus::bParameterMissingAlpha;
        }

        if (params.find("beta") == params.end()) {
            LOG(ERROR) << "Can not find the beta parameter";
            return ParseParameterAttrStatus::bParameterMissingBeta;
        }

        auto beta_param = std::dynamic_pointer_cast<RuntimeParameterFloat>(params.at("beta"));
        if (beta_param == nullptr) {
            LOG(ERROR) << "Can not find the beta parameter";
            return ParseParameterAttrStatus::bParameterMissingBeta;
        }

        if (params.find("transB") == params.end()) {
            LOG(ERROR) << "Can not find the transB parameter";
            return ParseParameterAttrStatus::bParameterMissingTransB;
        }

        auto transB_param = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("transB"));
        if (transB_param == nullptr) {
            LOG(ERROR) << "Can not find the transB parameter";
            return ParseParameterAttrStatus::bParameterMissingTransB;
        }

        // 获取Attributes

        const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &attributes = op->attribute;

        if (attributes.size() != 2) {
            LOG(ERROR) << "Can not find the weights and bias params";
            return ParseParameterAttrStatus::bAttrMissingWeight;
        }
        auto weight = attributes.at("fc.weight");
        auto bias = attributes.at("fc.bias");

        auto gemm_width = weight->shape.at(0);
        auto gemm_height = weight->shape.at(1);

        if (gemm_width != bias->shape.at(0)) {
            LOG(ERROR) << "The weights and bias params not same cols";
            return ParseParameterAttrStatus::bAttrMissingOutFeatures;
        }

        gemm_layer = std::make_shared<GemmLayer>(
                alpha_param->value,
                beta_param->value,
                transB_param->value,
                gemm_width,
                gemm_height);

        // load weights
        const std::vector<float>& weight_value = weight->weight_data;
        gemm_layer->set_weights(weight_value);

        const std::vector<float>& bias_value = bias->weight_data;
        gemm_layer->set_bias(bias_value);

        auto gemm_layer_derived = std::dynamic_pointer_cast<GemmLayer>(gemm_layer);
        CHECK(gemm_layer_derived != nullptr);

        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    GemmLayer::GemmLayer(float alpha,
                         float beta,
                         int32_t trans_b,
                         int32_t row,
                         int32_t cols) : ParamLayer("Gemm"),
                         alpha_(alpha),
                         beta_(beta),
                         trans_b_(trans_b),
                         gemm_height_(row),
                         gemm_width_(cols){
        this->InitWeightParam(1, 1, row, cols);
        this->InitBiasParam(1, 1, 1, cols);
    }

    LayerRegistererWrapper bGemmGetInstance("Gemm", GemmLayer::GetInstance);
}