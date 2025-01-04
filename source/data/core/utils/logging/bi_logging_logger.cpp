//
// Created by Mason on 2025/1/4.
//

#include <data/core/utils/logging/bi_logging_logger.hpp>

#include <data/core/bi_error.h>

#include <memory>

using namespace BatmanInfer::logging;

BILogger::BILogger(std::string name, BILogLevel log_level, std::shared_ptr<BIPrinter> printer) : _name(std::move(name)),
                                                                                                 _log_level(log_level),
                                                                                                 _printers({std::move(
                                                                                                         printer)}),
                                                                                                 _decorators() {
    BI_COMPUTE_ERROR_ON(printer == nullptr);

    set_default_decorators();
}

BILogger::BILogger(std::string name, BatmanInfer::logging::BILogLevel log_level,
                   std::vector<std::shared_ptr<BIPrinter>> printers) : _name(std::move(name)), _log_level(log_level),
                                                                       _printers(std::move(printers)), _decorators() {
    // Check printers
    for (const auto &p: _printers) {
        BI_COMPUTE_UNUSED(p);
        BI_COMPUTE_ERROR_ON(p == nullptr);
    }
    // Set default message decorators
    set_default_decorators();
}

BILogger::BILogger(std::string name, BatmanInfer::logging::BILogLevel log_level,
                   std::vector<std::shared_ptr<BIPrinter>> printers,
                   std::vector<std::unique_ptr<BIIDecorator>> decorators) : _name(std::move(name)),
                                                                            _log_level(log_level),
                                                                            _printers(std::move(printers)),
                                                                            _decorators(std::move(decorators)) {
    // Check printers
    for (const auto &p: _printers) {
        BI_COMPUTE_UNUSED(p);
        BI_COMPUTE_ERROR_ON(p == nullptr);
    }
    // Check decorators
    for (const auto &d: _decorators) {
        BI_COMPUTE_UNUSED(d);
        BI_COMPUTE_ERROR_ON(d == nullptr);
    }
}

void BILogger::log(BILogLevel log_level, const std::string &msg) {
    // Return if message shouldn't be logged
    // i.e. if log level does not match the logger's
    if (!is_loggable(log_level)) {
        return;
    }

    // Print message to all printers
    print_all(create_log_msg(msg, log_level));
}

void BILogger::set_log_level(BatmanInfer::logging::BILogLevel log_level) {
    _log_level = log_level;
}

BILogLevel BILogger::log_level() const {
    return _log_level;
}

std::string BILogger::name() const {
    return _name;
}

void BILogger::add_printer(std::shared_ptr<BIPrinter> printer) {
    BI_COMPUTE_ERROR_ON(printer == nullptr);
    _printers.push_back(std::move(printer));
}

void BILogger::add_decorator(std::unique_ptr<BIIDecorator> decorator) {
    BI_COMPUTE_ERROR_ON(decorator == nullptr);
    _decorators.push_back(std::move(decorator));
}

void BILogger::set_default_decorators() {
    _decorators.emplace_back(std::make_unique<BIStringDecorator>(_name));
    _decorators.emplace_back(std::make_unique<BIDateDecorator>());
    _decorators.emplace_back(std::make_unique<BILogLevelDecorator>());
}

bool BILogger::is_loggable(BatmanInfer::logging::BILogLevel log_level) {
    return (log_level >= _log_level);
}

void BILogger::decorate_log_msg(BatmanInfer::logging::BILogMsg &msg) {
    for (const auto &d: _decorators) {
        d->decorate(msg);
    }
    msg.raw_ += std::string(" ");
}

std::string BILogger::create_log_msg(const std::string &str, BatmanInfer::logging::BILogLevel log_level) {
    // Adding space string to avoid Android failures
    BILogMsg log_msg(" ", log_level);
    decorate_log_msg(log_msg);
    std::ostringstream ss;
    ss << log_msg.raw_ << " " << str;
    return ss.str();
}

void BILogger::print_all(const std::string &msg) {
    for (auto &p: _printers) {
        p->print(msg);
    }
}








