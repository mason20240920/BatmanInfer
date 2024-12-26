//
// Created by Mason on 2024/12/25.
//

#ifndef BATMANINFER_TENSOR_DESCRIPTION_HPP
#define BATMANINFER_TENSOR_DESCRIPTION_HPP

#include <support/i_clone_able.h>
#include <data/bi_tensor_shape.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/quantization_info.hpp>

#include <memory>

namespace BatmanInfer {
    namespace tensor {
        struct BITensorDescriptor final : public misc::ICloneable<BITensorDescriptor> {
            BITensorDescriptor() = default;

            /**
             * @brief 初始化
             * @param tensor_shape 张量形状
             * @param tensor_data_type 张量数据类型
             * @param tensor_quant_info 张量量化信息
             * @param tensor_target 张量声明地址
             */
            BITensorDescriptor(BITensorShape tensor_shape,
                             BIDataType tensor_data_type,
                             BIQuantizationInfo tensor_quant_info = BIQuantizationInfo(),
                               BITarget tensor_target = BITarget::UNSPECIFIED):
                             shape(tensor_shape),
                             data_type(tensor_data_type),
                             quant_info(tensor_quant_info),
                             target(tensor_target){

            }

            /**
             * @brief 设置Tensor描述的形状
             * @param tensor_shape
             * @return
             */
            BITensorDescriptor &set_shape(BITensorShape &tensor_shape) {
                shape = tensor_shape;
                return *this;
            }

            BITensorDescriptor &set_data_type(BIDataType tensor_data_type) {
                data_type = tensor_data_type;
                return *this;
            }

            BITensorDescriptor &set_quantization_info(BIQuantizationInfo tensor_quant_info) {
                quant_info = tensor_quant_info;
                return *this;
            }

            std::unique_ptr<BITensorDescriptor> clone() const override {
                return std::make_unique<BITensorDescriptor>(*this);
            }


            /**
             * @brief 张量的形状
             */
            BITensorShape shape{};
            /**
             * @brief 数据类型
             */
            BIDataType data_type{BIDataType::UNKNOWN};

            /**
             * @brief 量化信息
             */
            BIQuantizationInfo quant_info{};

            /**
             * @brief 设备信息
             */
            BITarget target{BITarget::UNSPECIFIED};
        };
    }
}

#endif //BATMANINFER_TENSOR_DESCRIPTION_HPP
