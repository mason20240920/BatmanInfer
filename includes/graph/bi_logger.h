//
// Created by holynova on 2025/1/10.
//

#ifndef BATMANINFER_GRAPH_BI_LOGGER_H
#define BATMANINFER_GRAPH_BI_LOGGER_H

#include "data/core/utils/logging/bi_logging_macros.hpp"

#ifdef BI_COMPUTE_LOGGING_ENABLED
/** Create a default core logger
 *
 * @note It will eventually create all default loggers in don't exist
 */
#define BI_COMPUTE_CREATE_DEFAULT_GRAPH_LOGGER()                                      \
    do                                                                                \
    {                                                                                 \
        if (BatmanInfer::logging::BILoggerRegistry::get().logger("GRAPH") == nullptr) \
        {                                                                             \
            BatmanInfer::logging::BILoggerRegistry::get().create_reserved_loggers();  \
        }                                                                             \
    } while (false)
#else /* BI_COMPUTE_LOGGING_ENABLED */
#define BI_COMPUTE_CREATE_DEFAULT_GRAPH_LOGGER()
#endif /* BI_COMPUTE_LOGGING_ENABLED */

#define BI_COMPUTE_LOG_GRAPH(log_level, x)    \
    BI_COMPUTE_CREATE_DEFAULT_GRAPH_LOGGER(); \
    BI_COMPUTE_LOG_STREAM("GRAPH", log_level, x)

#define BI_COMPUTE_LOG_GRAPH_VERBOSE(x)       \
    BI_COMPUTE_CREATE_DEFAULT_GRAPH_LOGGER(); \
    BI_COMPUTE_LOG_STREAM("GRAPH", BatmanInfer::logging::BILogLevel::VERBOSE, x)

#define BI_COMPUTE_LOG_GRAPH_INFO(x)          \
    BI_COMPUTE_CREATE_DEFAULT_GRAPH_LOGGER(); \
    BI_COMPUTE_LOG_STREAM("GRAPH", BatmanInfer::logging::BILogLevel::INFO, x)

#define BI_COMPUTE_LOG_GRAPH_WARNING(x)       \
    BI_COMPUTE_CREATE_DEFAULT_GRAPH_LOGGER(); \
    BI_COMPUTE_LOG_STREAM("GRAPH", BatmanInfer::logging::BILogLevel::WARN, x)

#define BI_COMPUTE_LOG_GRAPH_ERROR(x)         \
    BI_COMPUTE_CREATE_DEFAULT_GRAPH_LOGGER(); \
    BI_COMPUTE_LOG_STREAM("GRAPH", BatmanInfer::logging::BILogLevel::ERROR, x)

#endif //BATMANINFER_GRAPH_BI_LOGGER_H
