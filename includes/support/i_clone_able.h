//
// Created by Mason on 2024/12/25.
//

#ifndef BATMANINFER_I_CLONE_ABLE_H
#define BATMANINFER_I_CLONE_ABLE_H

#include <memory>

namespace BatmanInfer {
    namespace misc {
        /**
         * @brief 克隆接口类
         */
        template <class T>
        class ICloneable {
        public:
            /**
             * @brief 默认虚克隆
             */
            virtual ~ICloneable() = default;

            /**
             * @brief 提供类 T 当前对象的克隆
             * @return 克隆类T的对象
             */
            virtual std::unique_ptr<T> clone() const;
        };
    }
}

#endif //BATMANINFER_I_CLONE_ABLE_H
