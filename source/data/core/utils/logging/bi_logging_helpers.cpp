//
// Created by Mason on 2025/1/4.
//

#include "data/core/utils/logging/bi_logging_helpers.hpp"

#include <map>
#include <string>

using namespace BatmanInfer::logging;

const std::string &BatmanInfer::logging::string_from_log_level(BatmanInfer::logging::BILogLevel log_level) {
    static std::map<BILogLevel, const std::string> log_level_map = {
            {BILogLevel::VERBOSE, "VERBOSE"},
            {BILogLevel::INFO,    "INFO"},
            {BILogLevel::WARN,    "WARN"},
            {BILogLevel::OFF,     "OFF"},
    };

    return log_level_map[log_level];
}