//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_BI_LUT_MANAGER_HPP
#define BATMANINFER_BI_LUT_MANAGER_HPP

#include <data/core/core_types.hpp>
#include <data/core/quantization_info.hpp>
#include <function_info/bi_activationLayerInfo.h>

#include <map>
#include <memory>

namespace BatmanInfer {
#ifdef __aarch64__
    using LookupTable256 = std::array<float, 256>; // 256项的浮点查找表
    using LookupTable65536 = std::array<float16_t, 65536>; // 65536项的半精度浮点查找表
#endif // __aarch64__

    enum class LUTType {
        Activation,  // Determined by activation type
        Exponential, // e^(beta * x)
    };

    struct BILUTInfo {
        // For exponential lookup
        BILUTInfo(LUTType lut, float b, BIDataType type, BIUniformQuantizationInfo info)
                : act(), alpha(1.0f), beta(b), dt(type), qinfo(info), type(lut) {
        }

        // For activation functions
        BILUTInfo(BIActivationFunction func, float a, float b, BIDataType type, BIUniformQuantizationInfo info)
                : act(func), alpha(a), beta(b), dt(type), qinfo(info), type(LUTType::Activation) {
        }

        // Operators enable use of map with Lutinfo as key
        friend bool operator<(const BILUTInfo &l, const BILUTInfo &r) {
            const auto l_tup = std::make_tuple(l.type, l.act, l.alpha, l.beta, l.dt, l.qinfo.scale, l.qinfo.offset);
            const auto r_tup = std::make_tuple(r.type, r.act, r.alpha, r.beta, r.dt, r.qinfo.scale, r.qinfo.offset);

            return l_tup < r_tup;
        }

        bool operator==(const BILUTInfo &l) const {
            return this->type == l.type && this->act == l.act && this->alpha == l.alpha && this->beta == l.beta &&
                   this->dt == l.dt && this->qinfo == l.qinfo;
        }

        BIActivationLayerInfo::ActivationFunction act;   // 激活函数类型
        float alpha;                                   // 激活函数参数
        float beta;                                    // 激活函数参数
        BIDataType dt;                                   // 索引数据类型
        BIUniformQuantizationInfo qinfo;                // 量化信息
        LUTType type;                                 // 查找表类型
    };

    class BILUTManager {
    public:
        BILUTManager() = default;

        static BILUTManager &get_instance();

#ifdef __aarch64__

        template<typename T>
        std::shared_ptr<T> get_lut_table(BILUTInfo info);

    private:
        template<typename T>
        inline std::map<BILUTInfo, std::weak_ptr<T>> &get_map();

        // 存储查找表的映射
        std::map<BILUTInfo, std::weak_ptr<LookupTable256>> map_fp32{};
        std::map<BILUTInfo, std::weak_ptr<LookupTable65536>> map_fp16{};
#endif // __aarch64__
    };
}

#endif //BATMANINFER_BI_LUT_MANAGER_HPP
