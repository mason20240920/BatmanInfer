//
// Created by Mason on 2025/1/11.
//

#ifndef BATMANINFER_LEGACY_SUPPORT_HPP
#define BATMANINFER_LEGACY_SUPPORT_HPP

#include <bcl.h>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <function_info/bi_activationLayerInfo.h>

namespace BatmanInfer {
    namespace detail {
        /**
         * 将描述符转换为旧格式。
         * @param desc
         *
         * @return
         */
        BITensorInfo convert_to_legacy_tensor_info(const BclTensorDescriptor &desc);

        /**
         * 将旧版张量元数据转换为描述符
         * @param info
         * @return
         */
        BclTensorDescriptor convert_to_descriptor(const BITensorInfo &info);

        /**
         * 将AclActivation描述符转换为内部描述符
         * @param desc
         * @return
         */
        BIActivationLayerInfo convert_to_activation_info(const BclActivationDescriptor &desc);
    }
}

#endif //BATMANINFER_LEGACY_SUPPORT_HPP
