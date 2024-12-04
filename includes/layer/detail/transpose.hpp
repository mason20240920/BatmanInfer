//
// Created by Mason on 2024/12/4.
//

#ifndef BATMANINFER_TRANSPOSE_HPP
#define BATMANINFER_TRANSPOSE_HPP

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class TransposeLayer: public NonParamLayer {
    public:
        explicit TransposeLayer(std::vector<int> perms);

        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float> > > &inputs,
                            std::vector<std::shared_ptr<Tensor<float> > > &outputs) override;


        static ParseParameterAttrStatus CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                       std::shared_ptr<Layer> &transpose_layer);

    private:
        std::vector<int> perms_;
    };
}

#endif //BATMANINFER_TRANSPOSE_HPP
