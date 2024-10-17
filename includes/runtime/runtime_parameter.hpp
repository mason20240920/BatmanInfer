//
// Created by Mason on 2024/10/16.
//

#ifndef BATMAN_INFER_RUNTIME_PARAMETER_HPP
#define BATMAN_INFER_RUNTIME_PARAMETER_HPP
#include "status_code.hpp"
#include <string>
#include <vector>

namespace BatmanInfer {
    /**
     * 计算节点`Operator`中的参数信息
     * 参数分成下面几类
     * 计算节点中的参数信息
     *
     * 1. int
     * 2. float
     * 3. string
     * 4. bool
     * 5. int array
     * 6. string array
     * 7. float array
     */
    struct RuntimeParameter {
        virtual ~RuntimeParameter() = default;

        explicit RuntimeParameter(RuntimeParameterType type = RuntimeParameterType::bParameterUnknown) : type(type) {

        }

        RuntimeParameterType type = RuntimeParameterType::bParameterUnknown;
    };

    struct RuntimeParameterInt: public RuntimeParameter {
        RuntimeParameterInt(): RuntimeParameter(RuntimeParameterType::bParameterInt) {

        }
        int value = 0;
    };

    struct RuntimeParameterFloat: public RuntimeParameter {
        RuntimeParameterFloat() : RuntimeParameter(RuntimeParameterType::bParameterFloat) {

        }

        float value = 0.f;
    };

    struct RuntimeParameterString: public RuntimeParameter {
        RuntimeParameterString(): RuntimeParameter(RuntimeParameterType::bParameterString) {

        }
        std::string value;
    };

    struct RuntimeParameterIntArray: public RuntimeParameter {
        RuntimeParameterIntArray(): RuntimeParameter(RuntimeParameterType::bParameterIntArray) {

        }
        std::vector<int> value;
    };

    struct RuntimeParameterFloatArray: public RuntimeParameter {
        RuntimeParameterFloatArray(): RuntimeParameter(RuntimeParameterType::bParameterFloatArray) {

        }
        std::vector<float> value;
    };

    struct RuntimeParameterStringArray : public RuntimeParameter {
        RuntimeParameterStringArray() : RuntimeParameter(RuntimeParameterType::bParameterStringArray) {

        }
        std::vector<std::string> value;
    };

    struct RuntimeParameterBool : public RuntimeParameter {
        RuntimeParameterBool() : RuntimeParameter(RuntimeParameterType::bParameterBool) {

        }
        bool value = false;
    };
}

#endif //BATMAN_INFER_RUNTIME_PARAMETER_HPP
