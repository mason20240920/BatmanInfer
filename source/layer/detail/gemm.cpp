//
// Created by Mason on 2024/11/14.
//

#include <layer/detail/gemm.hpp>
#include <layer/abstract/layer_factory.hpp>
#include <data/tensor_util.hpp>

namespace BatmanInfer {
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
                         trans_b_(trans_b){
        this->InitWeightParam(1, 1, row, cols);
        this->InitBiasParam(1, 1, 1, cols);
    }

    LayerRegistererWrapper bGemmGetInstance("Gemm", GemmLayer::GetInstance);
}