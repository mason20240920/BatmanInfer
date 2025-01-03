//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_CPU_MODEL_HPP
#define BATMANINFER_CPU_MODEL_HPP

#include <data/core/cpp/cpp_types.hpp>

#include <cstdint>
#include <string>

namespace BatmanInfer {
    namespace cpu_info {
        using CpuModel = BatmanInfer::CPUModel;

        /**
         * @brief 转换一个CPU模型到一个字符串
         * @param model CpuModel值被转换完成
         * @return 字符串代表符合的CpuModel
         */
        std::string cpu_model_to_string(CpuModel model);

        /**
         * @brief 从MIDR值中提取模型类型
         * @param midr MIDR信息(主ID寄存器)
         * @return CpuModel对应
         */
        CpuModel midr_to_model(uint32_t midr);

        /**
         * @brief 检查模型是否支持半精度浮点运算。
         *
         * @note 在某些旧内核配置中使用此项，因为某些功能未被暴露。
         *
         * @param model
         * @return
         */
        bool model_supports_fp16(CpuModel model);

        /**
         * @brief 检查模型是否支持点积
         * @param model
         * @return
         */
        bool model_supports_dot(CpuModel model);
    }
}

#endif //BATMANINFER_CPU_MODEL_HPP
