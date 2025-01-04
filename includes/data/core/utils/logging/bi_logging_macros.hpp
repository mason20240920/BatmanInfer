//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_LOGGING_MACROS_HPP
#define BATMANINFER_BI_LOGGING_MACROS_HPP

#include <data/core/utils/logging/bi_logging_logger_registry.hpp>

#include <sstream>

#ifdef BI_COMPUTE_LOGGING_ENABLED

#ifdef __GNUC__

inline std::string signature_name(const std::string &pretty_func) {
    const auto scope_op = pretty_func.find("::");
    const auto begin    = pretty_func.substr(0, scope_op).rfind(" ") + 1;
    const auto end      = pretty_func.rfind("(") - begin;

    return pretty_func.substr(begin, end) + "()";
}

#define BI_COMPUTE_SIGNATURE_NAME signature_name(__PRETTY_FUNCTION__)
#else /* __GNUC__ */
#define BI_COMPUTE_SIGNATURE_NAME (__func__)
#endif /* __GNUC__ */

#define BI_COMPUTE_LOG_MSG(logger_name, log_level, msg)                                 \
    do                                                                                   \
    {                                                                                    \
        auto __logger = BatmanInfer::logging::BILoggerRegistry::get().logger(logger_name); \
        if (__logger != nullptr)                                                         \
        {                                                                                \
            __logger->log(log_level, msg);                                               \
        }                                                                                \
    } while (false)

#define BI_COMPUTE_LOG_MSG_WITH_FUNCNAME(logger_name, log_level, msg)                   \
    do                                                                                   \
    {                                                                                    \
        auto __logger = BatmanInfer::logging::BILoggerRegistry::get().logger(logger_name); \
        if (__logger != nullptr)                                                         \
        {                                                                                \
            std::ostringstream s;                                                        \
            s << BI_COMPUTE_SIGNATURE_NAME << " : " << msg;                             \
            __logger->log(log_level, s.str());                                           \
        }                                                                                \
    } while (false)

#define BI_COMPUTE_LOG_MSG_WITH_FORMAT(logger_name, log_level, fmt, ...)                     \
    do                                                                                        \
    {                                                                                         \
        auto __logger = BatmanInfer::logging::BILoggerRegistry::get().logger(logger_name);      \
        if (__logger != nullptr)                                                              \
        {                                                                                     \
            size_t size     = ::snprintf(nullptr, 0, fmt, __VA_ARGS__) + 1;                   \
            auto   char_str = std::make_unique<char[]>(size);                                 \
            ::snprintf(char_str.get(), size, fmt, __VA_ARGS__);                               \
            __logger->log(log_level, std::string(char_str.get(), char_str.get() + size - 1)); \
        }                                                                                     \
    } while (false)

#define BI_COMPUTE_LOG_STREAM(logger_name, log_level, stream)                           \
    do                                                                                   \
    {                                                                                    \
        auto __logger = BatmanInfer::logging::BILoggerRegistry::get().logger(logger_name); \
        if (__logger != nullptr)                                                         \
        {                                                                                \
            std::ostringstream s;                                                        \
            s << stream;                                                                 \
            __logger->log(log_level, s.str());                                           \
        }                                                                                \
    } while (false)

#else /* BI_COMPUTE_LOGGING_ENABLED */

#define BI_COMPUTE_LOG_MSG(logger_name, log_level, msg)
#define BI_COMPUTE_LOG_MSG_WITH_FUNCNAME(logger_name, log_level, msg)
#define BI_COMPUTE_LOG_MSG_WITH_FORMAT(logger_name, log_level, fmt, ...)
#define BI_COMPUTE_LOG_STREAM(logger_name, log_level, stream)

#endif

#endif //BATMANINFER_BI_LOGGING_MACROS_HPP
