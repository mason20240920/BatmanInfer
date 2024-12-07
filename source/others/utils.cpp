//
// Created by Mason on 2024/10/13.
//
#include <others/utils.hpp>

namespace BatmanInfer {
    std::string ShapeStr(const std::vector<int> &shapes) {
        std::ostringstream ss;
        for (int i = 0; i < shapes.size(); ++i) {
            ss << shapes.at(i);
            if (i != shapes.size() - 1) {
                ss << " x ";
            }
        }
        return ss.str();
    }

    RuntimeDataType convert_runtime_data(const int onnx_type) {
        switch (onnx_type) {
            case 11:
                return RuntimeDataType::kTypeFloat32;
            default:
                return RuntimeDataType::kTypeUnknown;
        }
    }

    std::vector<uint32_t> convert_to_uint32(const std::vector<float>& input) {
        std::vector<uint32_t> output(input.size());
        std::transform(input.begin(), input.end(), output.begin(),
                       [](float value) { return static_cast<uint32_t>(value); });
        return output;
    }

    std::shared_ptr<RuntimeAttribute> find_keys_with_substring(const std::map<std::string, std::shared_ptr<RuntimeAttribute>>& attributes,
                                                               const std::string& substring) {

        for (const auto& pair : attributes) {
            // 检查键是否包含子串
            if (pair.first.find(substring) != std::string::npos) {
                return pair.second;
            }
        }

        return nullptr;
    }

    std::vector<float> convert_int64_to_float(const std::vector<int64_t>& input) {
        std::vector<float> output(input.size());
        std::transform(input.begin(), input.end(), output.begin(), [](int64_t value) {
            return static_cast<float>(value);
        });
        return output;
    }
}