//
// Created by Mason on 2025/1/7.
//

#ifndef BATMANINFER_SRC_COMMON_LOG_H
#define BATMANINFER_SRC_COMMON_LOG_H

#ifndef BI_COMPUTE_LOGGING_ENABLED

#define BI_COMPUTE_CREATE_ACL_LOGGER()
#define BI_COMPUTE_LOG_MSG_ACL(log_level, msg)
#define BI_COMPUTE_LOG_MSG_WITH_FORMAT_ACL(log_level, fmt, ...)
#define BI_COMPUTE_LOG_ERROR_ACL(msg)
#define BI_COMPUTE_LOG_ERROR_WITH_FUNCNAME_ACL(msg)
#define BI_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL(msg)
#define BI_COMPUTE_LOG_PARAMS(...)

#else /* BI_COMPUTE_LOGGING_ENABLED */

#include <data/core/bi_error.h>
#include <data/core/utils/logging/bi_logging_macros.hpp>
#include <utils/bi_type_printer.hpp>



/** Create a logger
 *
 * @note It will eventually create all default loggers in don't exist
 */
#define BI_COMPUTE_CREATE_ACL_LOGGER()                                                                      \
    do                                                                                                       \
    {                                                                                                        \
        if (BatmanInfer::logging::BILoggerRegistry::get().logger("ComputeLibrary") == nullptr)                 \
        {                                                                                                    \
           BatmanInfer::logging::BILoggerRegistry::get().create_logger("ComputeLibrary",                      \
                                                                       BatmanInfer::logging::BILogLevel::INFO); \
        }                                                                                                    \
    } while (false)

/** Log a message to the logger
 *
 * @param[in] log_level Logging level
 * @param[in] msg       Message to log
 */
#define BI_COMPUTE_LOG_MSG_ACL(log_level, msg)                \
    do                                                         \
    {                                                          \
        BI_COMPUTE_CREATE_ACL_LOGGER();                       \
        BI_COMPUTE_LOG_MSG("ComputeLibrary", log_level, msg); \
    } while (false)

/** Log a message with format to the logger
 *
 * @param[in] log_level Logging level
 * @param[in] fmt       String format (printf style)
 * @param[in] ...       Message arguments
 */
#define BI_COMPUTE_LOG_MSG_WITH_FORMAT_ACL(log_level, fmt, ...)                        \
    do                                                                                  \
    {                                                                                   \
        BI_COMPUTE_CREATE_ACL_LOGGER();                                                \
        BI_COMPUTE_LOG_MSG_WITH_FORMAT("ComputeLibrary", log_level, fmt, __VA_ARGS__); \
    } while (false)

/** Log an error message to the logger
 *
 * @param[in] msg Message to log
 */
#define BI_COMPUTE_LOG_ERROR_ACL(msg)                                                     \
    do                                                                                     \
    {                                                                                      \
        BI_COMPUTE_CREATE_ACL_LOGGER();                                                   \
        BI_COMPUTE_LOG_MSG("ComputeLibrary", BatmanInfer::logging::BILogLevel::ERROR, msg); \
    } while (false)

/** Log an error message to the logger with function name before the message
 *
 * @param[in] msg Message to log
 */
#define BI_COMPUTE_LOG_ERROR_WITH_FUNCNAME_ACL(msg)                                                     \
    do                                                                                                   \
    {                                                                                                    \
        BI_COMPUTE_CREATE_ACL_LOGGER();                                                                 \
        BI_COMPUTE_LOG_MSG_WITH_FUNCNAME("ComputeLibrary", BatmanInfer::logging::BILogLevel::ERROR, msg); \
    } while (false)

/** Log an information message to the logger with function name before the message
 *
 * @param[in] msg Message to log
 */
#define BI_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL(msg)                                                     \
    do                                                                                                  \
    {                                                                                                   \
        BI_COMPUTE_CREATE_ACL_LOGGER();                                                                \
        BI_COMPUTE_LOG_MSG_WITH_FUNCNAME("ComputeLibrary", BatmanInfer::logging::BILogLevel::INFO, msg); \
    } while (false)

/** Function template specialization for the out of bound element at index = tuple_size
 *
 * @param[in,out] data_registry   Reference to the input parameters data in a string format
 * @param[in]     in_params_tuple Tuple of different input data types
 */
template<std::size_t Index, typename... Tp>
inline typename std::enable_if<Index == sizeof...(Tp), void>::type
logParamsImpl(std::vector<std::string> &data_registry, const std::tuple<Tp...> &in_params_tuple) {
    // Because it is out of bound index so do nothing
    BI_COMPUTE_UNUSED(data_registry);
    BI_COMPUTE_UNUSED(in_params_tuple);
}

/** Function template to iterate over all input parameters tuple at compile time:
 *
 * @param[in,out] data_registry   Reference to a vector of input parameters data in a string format
 * @param[in]     in_params_tuple Constant reference to a tuple of different input data types
 */
template<std::size_t Index, typename... Tp>
inline typename std::enable_if<
        Index < sizeof...(Tp), void>::type logParamsImpl(std::vector<std::string> &data_registry,
                                                         const std::tuple<Tp...> &in_params_tuple) {
    data_registry.push_back(BatmanInfer::to_string(std::get<Index>(in_params_tuple)));
    // Unfold the next tuple element
    logParamsImpl<Index + 1, Tp...>(data_registry, in_params_tuple);
}

/** Function Template with variable number of inputs to collect all the passed parameters from
 *  the logging macro BI_COMPUTE_LOG_PARAMS(...)
 *
 * @param[in] ...ins The input parameters in the variadic template, taken by universal references Ts.. &&, (not by value)
 *                   to avoid detecting T as an abstract data type when passing any of these parameters as an L-value
 *                   reference to an abstract type.
 *
 * @return  Vector of the parameters' data in a string format
 */
template<typename... Ts>
const std::vector<std::string> logParams(Ts &&...ins) {
    std::vector<std::string> data_registry{};
    std::tuple<Ts...> in_params_tuple{ins...};

    // Start logging the tuple elements, starting from 0 to tuple_size-1
    logParamsImpl<0>(data_registry, in_params_tuple);
    return data_registry;
}

/** Inline function to parse the input parameters string passed from strignizing of the variadic macro input
 *  #__VA_ARGS__.
 *  It is Inline to avoid the redefinition of this function each time this header is included
 *
 * @param[in] in_params_str Constant reference to a string consists of the names of the input parameters provided
 *                          as:BI_COMPUTE_LOG_PARAMS(src0, src1) the params_names = "src0, src1"
 *
 * @return  Vector of strings containing all the names of the input parameters
 */
inline const std::vector<std::string> getParamsNames(const std::string &in_params_str) {
    std::stringstream ss(in_params_str);

    // Vector containing all the names of the input parameters
    std::vector<std::string> names;
    std::string temp;

    // Usually the input parameters string would be name of parameters separated
    // by ',' e.g. "src0, src1, policy"
    while (std::getline(ss, temp, ',')) {
        names.push_back(temp);
    }
    for (auto &name: names) {
        // Totally get rid of white space characters
        name.erase(std::remove(name.begin(), name.end(), ' '), name.end());
    }
    return names;
}

/** It constructs the log message to be displayed by the logger by writing each parameter name and its
 *  corresponding data info string.
 *
 * @param[in] params_names  Constant reference to a string consists of the the input parameters' names
 *                          provided e.g.: BI_COMPUTE_LOG_PARAMS(src0, src1) then params_names = "src0, src1"
 * @param[in] data_registry Constant reference to a registry of all parameters' data in string format,
 *                          stringnized by BatmanInfer::to_string()
 *
 * @return  Log message string to be displayed
 */
inline const std::string constructDataLog(const std::vector<std::string> &params_names,
                                          const std::vector<std::string> &data_registry) {
    std::string dataLog = "\n ";
    BI_COMPUTE_ERROR_ON(params_names.size() != data_registry.size());
    for (uint8_t i = 0; i < params_names.size(); ++i) {
        dataLog += params_names[i] + ": " + data_registry.at(i) + "\n ";
    }

    return dataLog;
}

/** Macro for logging input Parameters from any function.
 *  It detects the input parameters names, and their corresponding values before stringizing them using
 *  the overloaded BI_COMPUTE::to_string() type printer. Finally, displayed using the printer configured
 *  in the logger.
 *
 * @param[in] ... Input parameters
 */
#define BI_COMPUTE_LOG_PARAMS(...)                                                  \
    do                                                                               \
    {                                                                                \
        BI_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL(                                      \
            constructDataLog(getParamsNames(#__VA_ARGS__), logParams(__VA_ARGS__))); \
    } while (false)
#endif /* BI_COMPUTE_LOGGING_ENABLED */

#endif //BATMANINFER_SRC_COMMON_LOG_H
