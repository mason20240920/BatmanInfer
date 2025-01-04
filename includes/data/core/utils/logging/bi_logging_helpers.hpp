//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_LOGGING_HELPERS_HPP
#define BATMANINFER_BI_LOGGING_HELPERS_HPP

#include <data/core/utils/logging/bi_logging_types.hpp>

#include <support/bi_toolchain_support.hpp>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>

namespace BatmanInfer {
    namespace logging {
        /**
         * @brief  Create a string given a format
         * @tparam Ts
         * @param fmt String format
         * @param args Arguments
         * @return
         */
        template<typename... Ts>
        inline std::string string_with_format(const std::string &fmt,
                                              Ts &&...args) {
            size_t size     = support::cpp11::snprintf(nullptr, 0, fmt.c_str(), args...) + 1;
            auto   char_str = std::make_unique<char[]>(size);
            support::cpp11::snprintf(char_str.get(), size, fmt.c_str(), args...);
            return std::string(char_str.get(), char_str.get() + size - 1);
        }

        /**
         * @brief Wraps a value with angles and returns the string
         * @tparam T
         * @param val
         * @return
         */
        template<typename T>
        inline std::string angle_wrap_value(const T &val) {
            std::ostringstream ss;
            ss << "[" << val << "]";
            return ss.str();
        }

        /**
         * @brief Translates a given log level to a string.
         * @param log_level @ref BILogLevel to be translated to string.
         * @return
         */
        const std::string &string_from_log_level(BILogLevel log_level);
    }
}

#endif //BATMANINFER_BI_LOGGING_HELPERS_HPP
