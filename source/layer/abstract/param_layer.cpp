//
// Created by Mason on 2024/10/29.
//

#include <layer/abstract/param_layer.hpp>
#include <glog/logging.h>

namespace BatmanInfer {
    ParamLayer::ParamLayer(const std::string &layer_name) : Layer(layer_name) {}

    void ParamLayer::InitBiasParam(const uint32_t param_count,
                                   const uint32_t param_channel,
                                   const uint32_t param_height,
                                   const uint32_t param_width) {
        bias_ = std::make_shared<ftensor>(param_count,
                                          param_channel,
                                          param_height,
                                          param_width);
    }

    void ParamLayer::InitWeightParam(const uint32_t param_count,
                                     const uint32_t param_channel,
                                     const uint32_t param_height,
                                     const uint32_t param_width) {
        // 一个大小为 param_count 的矩阵数组
        weights_ = std::make_shared<ftensor>(param_count,
                                             param_channel,
                                             param_height,
                                             param_width);
    }

    const std::shared_ptr<Tensor<float>> &ParamLayer::weights() const {
        return weights_;
    }

    const std::shared_ptr<Tensor<float>> &ParamLayer::bias() const {
        return bias_;
    }

    void ParamLayer::set_weights(
            const std::shared_ptr<Tensor<float>> &weights) {
        CHECK(this->weights_ != nullptr);
        CHECK(this->weights_->rows() == weights->rows());
        CHECK(this->weights_->cols() == weights->cols());
        CHECK(this->weights_->channels() == weights->channels());
        weights_ = weights;
    }

    void ParamLayer::set_bias(const std::shared_ptr<Tensor<float>> &bias) {
        CHECK(this->bias_ != nullptr);
        CHECK(this->bias_->rows() == bias->rows());
        CHECK(this->bias_->cols() == bias->cols());
        CHECK(this->bias_->channels() == bias->channels());
        bias_ = bias;
    }

    void ParamLayer::set_weights(const std::vector<float> &weights) {
        const uint32_t elem_size = weights.size();
        CHECK_EQ(weights_->size(), elem_size);
        weights_->Fill(weights);
    }

    void ParamLayer::set_bias(const std::vector<float> &bias) {
        const uint32_t elem_size = bias.size();
        CHECK_EQ(bias_->size(), elem_size);
        bias_->Fill(bias);
    }
}