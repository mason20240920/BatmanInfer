//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_LOGGING_LOG_HPP
#define BATMANINFER_BI_LOGGING_LOG_HPP

#include <data/core/utils/logging/bi_logging_macros.hpp>

#ifdef BI_COMPUTE_LOGGING_ENABLED
/** Create a default core logger
 *
 * @note It will eventually create all default loggers in don't exist
 */
#define BI_COMPUTE_CREATE_DEFAULT_CORE_LOGGER()                                   \
    do                                                                             \
    {                                                                              \
        if (BatmanInfer::logging::BILoggerRegistry::get().logger("CORE") == nullptr) \
        {                                                                          \
            BatmanInfer::logging::BILoggerRegistry::get().create_reserved_loggers(); \
        }                                                                          \
    } while (false)
#else /* BI_COMPUTE_LOGGING_ENABLED */
#define BI_COMPUTE_CREATE_DEFAULT_CORE_LOGGER()
#endif /* BI_COMPUTE_LOGGING_ENABLED */

/** Log a message to the core system logger
 *
 * @param[in] log_level Logging level
 * @param[in] msg       Message to log
 */
#define BI_COMPUTE_LOG_MSG_CORE(log_level, msg)     \
    do                                               \
    {                                                \
        BI_COMPUTE_CREATE_DEFAULT_CORE_LOGGER();    \
        BI_COMPUTE_LOG_MSG("CORE", log_level, msg); \
    } while (false)

/** Log a message with format to the core system logger
 *
 * @param[in] log_level Logging level
 * @param[in] fmt       String format (printf style)
 * @param[in] ...       Message arguments
 */
#define BI_COMPUTE_LOG_MSG_WITH_FORMAT_CORE(log_level, fmt, ...)             \
    do                                                                        \
    {                                                                         \
        BI_COMPUTE_CREATE_DEFAULT_CORE_LOGGER();                             \
        BI_COMPUTE_LOG_MSG_WITH_FORMAT("CORE", log_level, fmt, __VA_ARGS__); \
    } while (false)

/** Log a stream to the core system logger
 *
 * @param[in] log_level Logging level
 * @param[in] ss        Stream to log
 */
#define BI_COMPUTE_LOG_STREAM_CORE(log_level, ss)     \
    do                                                 \
    {                                                  \
        BI_COMPUTE_CREATE_DEFAULT_CORE_LOGGER();      \
        BI_COMPUTE_LOG_STREAM("CORE", log_level, ss); \
    } while (false)

/** Log information level message to the core system logger
 *
 * @param[in] msg Stream to log
 */
#define BI_COMPUTE_LOG_INFO_MSG_CORE(msg)                                   \
    do                                                                       \
    {                                                                        \
        BI_COMPUTE_CREATE_DEFAULT_CORE_LOGGER();                            \
        BI_COMPUTE_LOG_MSG_CORE(BatmanInfer::logging::BILogLevel::INFO, msg); \
    } while (false)

/** Log information level formatted message to the core system logger
 *
 * @param[in] fmt String format (printf style)
 * @param[in] ... Message arguments
 */
#define BI_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE(fmt, ...)                                            \
    do                                                                                                 \
    {                                                                                                  \
        BI_COMPUTE_CREATE_DEFAULT_CORE_LOGGER();                                                      \
        BI_COMPUTE_LOG_MSG_WITH_FORMAT_CORE(BatmanInfer::logging::BILogLevel::INFO, #fmt, __VA_ARGS__); \
    } while (false)

/** Log information level stream to the core system logger
 *
 * @param[in] ss Message to log
 */
#define BI_COMPUTE_LOG_INFO_STREAM_CORE(ss)                                   \
    do                                                                         \
    {                                                                          \
        BI_COMPUTE_CREATE_DEFAULT_CORE_LOGGER();                              \
        BI_COMPUTE_LOG_STREAM_CORE(BatmanInfer::logging::BILogLevel::INFO, ss); \
    } while (false)

#endif //BATMANINFER_BI_LOGGING_LOG_HPP
