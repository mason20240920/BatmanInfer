//
// Created by Mason on 2025/1/4.
//

#include <data/core/utils/logging/bi_logging_logger_registry.hpp>

#include <data/core/bi_error.h>

#include <support/mutex.hpp>

using namespace BatmanInfer::logging;

/** Reserved logger used by the library */
std::set<std::string> BILoggerRegistry::_reserved_loggers = {"CORE", "RUNTIME", "GRAPH"};

BILoggerRegistry::BILoggerRegistry() : _mtx(), _loggers() {
}

BILoggerRegistry &BILoggerRegistry::get() {
    static BILoggerRegistry _instance;
    return _instance;
}

void BILoggerRegistry::create_logger(const std::string &name,
                                     BILogLevel log_level,
                                     const std::vector<std::shared_ptr<BIPrinter>> &printers) {
    BatmanInfer::lock_guard<BatmanInfer::Mutex> lock(_mtx);
    if ((_loggers.find(name) == _loggers.end()) && (_reserved_loggers.find(name) == _reserved_loggers.end())) {
        _loggers[name] = std::make_shared<BILogger>(name, log_level, printers);
    }
}

void BILoggerRegistry::remove_logger(const std::string &name) {
    BatmanInfer::lock_guard<BatmanInfer::Mutex> lock(_mtx);
    if (_loggers.find(name) != _loggers.end()) {
        _loggers.erase(name);
    }
}

std::shared_ptr<BILogger> BILoggerRegistry::logger(const std::string &name) {
    BatmanInfer::lock_guard<BatmanInfer::Mutex> lock(_mtx);
    return (_loggers.find(name) != _loggers.end()) ? _loggers[name] : nullptr;
}

void
BILoggerRegistry::create_reserved_loggers(BILogLevel log_level,
                                          const std::vector<std::shared_ptr<BIPrinter>> &printers) {
    BatmanInfer::lock_guard<BatmanInfer::Mutex> lock(_mtx);
    for (const auto                             &r: _reserved_loggers) {
        if (_loggers.find(r) == _loggers.end()) {
            _loggers[r] = std::make_shared<BILogger>(r, log_level, printers);
        }
    }
}