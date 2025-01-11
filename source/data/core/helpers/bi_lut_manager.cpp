//
// Created by Mason on 2025/1/11.
//

#include <data/core/helpers/bi_lut_manager.hpp>

#include <common/utils/bi_validate.hpp>
#include <support/b_float16.hpp>

namespace BatmanInfer {
#ifdef __aarch64__
    namespace {

        union Element {
            uint16_t i = 0;
            float16_t fp;
        };

        inline float16_t activation(float16_t x, const BILUTInfo &info) {
            float16_t out = 0.f;
            switch (info.act) {
                case BIActivationLayerInfo::ActivationFunction::LOGISTIC:
                    out = 1.f / (1.f + std::exp(-x));
                    break;
                case BIActivationLayerInfo::ActivationFunction::TANH: {
                    out = static_cast<float16_t>(info.alpha * std::tanh(info.beta * x));
                    break;
                }
                default:
                    BI_COMPUTE_ERROR("Unsupported Activation for 16-bit LUT table");
                    break;
            }
            return out;
        }

        inline float exponential(float fp, const BILUTInfo &info) {
            return std::exp(fp * info.beta);
        }

// Read bf16 value as u16, convert to fp32.
// Calculate exp in fp32, return as bf16
        inline uint16_t exponential_bf16(uint16_t x, const BILUTInfo &info) {
            float fp = bf16_to_float(x);
            fp = exponential(fp, info);
            return float_to_bf16(fp);
        }

        void init_lut(LookupTable256 &lut, const BILUTInfo &info) {
            // assert lut is valid config.
            BI_COMPUTE_ASSERT((info.type == LUTType::Exponential && info.dt == BIDataType::QASYMM8) ||
                              (info.type == LUTType::Exponential && info.dt == BIDataType::QASYMM8_SIGNED));

            for (int i = 0; i < 256; ++i) {
                const float deq = info.dt == BIDataType::QASYMM8 ? dequantize_qasymm8(i, info.qinfo)
                                                                 : dequantize_qasymm8_signed(i - 128, info.qinfo);
                lut[i] = exponential(deq, info);
            }
        }

        void init_lut(LookupTable65536 &lut, const BILUTInfo &info) {
            // assert lut is valid config.
            BI_COMPUTE_ASSERT((info.type == LUTType::Activation && info.dt == BIDataType::F16) ||
                              (info.type == LUTType::Exponential && info.dt == BIDataType::BFLOAT16));

            Element item = {0}; // Fill lut by iterating over all 16 bit values using the union.
            Element bf16 = {0}; // Temporary object used to store bf16 values as fp16 in lut
            while (true) {
                switch (info.type) {
                    case LUTType::Activation: {
                        lut[item.i] = activation(item.fp, info);
                        break;
                    }
                    case LUTType::Exponential: {
                        bf16.i = exponential_bf16(item.i, info);
                        lut[item.i] = bf16.fp;
                        break;
                    }
                    default:
                        BI_COMPUTE_ERROR("Unsupported Activation for 16-bit LUT table");
                        break;
                }
                if (item.i == 65535)
                    break;
                item.i++;
            }
        }

    } // namespace

    template<>
    inline std::map<BILUTInfo, std::weak_ptr<LookupTable256>> &BILUTManager::get_map<LookupTable256>() {
        return map_fp32;
    }

    template<>
    inline std::map<BILUTInfo, std::weak_ptr<LookupTable65536>> &BILUTManager::get_map<LookupTable65536>() {
        return map_fp16;
    }

    template<typename T>
    std::shared_ptr<T> BILUTManager::get_lut_table(BILUTInfo info) {
        auto &map = get_map<T>();
        const auto itr = map.find(info);
        auto s_ptr = (itr != map.end()) ? itr->second.lock() : nullptr; // nullptr if invalid or not found.
        if (s_ptr != nullptr) {
            // Found and valid
            return s_ptr; // Return weak ptr as shared ptr
        } else {
            // Not found, or pointer not valid
            // We do not use make_shared to prevent the weak_ptr keeping the control block alive
            std::shared_ptr<T> ptr(new T);
            init_lut(*ptr, info);
            map[info] = ptr;
            return ptr;
        }
    }

    template std::shared_ptr<LookupTable256> BILUTManager::get_lut_table<LookupTable256>(BILUTInfo info);

    template std::shared_ptr<LookupTable65536> BILUTManager::get_lut_table<LookupTable65536>(BILUTInfo info);

#endif // __aarch64__

    // Static function to get LutManager instance
    BILUTManager &BILUTManager::get_instance() {
        static auto inst_ = std::make_unique<BILUTManager>(); // The one, single instance.
        return *inst_;
    }
}