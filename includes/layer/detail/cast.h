//
// Created by Mason on 2024/12/2.
//

#ifndef BATMANINFER_CAST_H
#define BATMANINFER_CAST_H

#include <layer/abstract/non_param_layer.hpp>

namespace BatmanInfer {
    class CastLayer : public NonParamLayer {
    public:
        explicit CastLayer(RuntimeDataType dtype);

        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float> > > &inputs,
                            std::vector<std::shared_ptr<Tensor<float> > > &outputs) override;


        static ParseParameterAttrStatus CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                       std::shared_ptr<Layer> &cast_layer);

    private:
        RuntimeDataType dtype_;
    };
}

#endif //BATMANINFER_CAST_H
