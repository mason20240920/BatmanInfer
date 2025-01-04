//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_LOGGING_LOGGER_HPP
#define BATMANINFER_BI_LOGGING_LOGGER_HPP

#include <data/core/utils/logging/bi_logging_helpers.hpp>
#include <data/core/utils/logging/bi_logging_i_printer.hpp>
#include <data/core/utils/logging/bi_logging_log_msg_decorators.hpp>
#include <data/core/utils/logging/bi_logging_types.hpp>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace BatmanInfer {
    namespace logging {
        /**
         * @brief 日志类
         */
        class BILogger {
        public:
            /** Default Constructor
     *
     * @param[in] name      Name of the logger
     * @param[in] log_level Logger log level
     * @param[in] printer   Printer to push the messages
     */
            BILogger(std::string name, BILogLevel log_level, std::shared_ptr<BIPrinter> printer);

            /** Default Constructor
             *
             * @param[in] name      Name of the logger
             * @param[in] log_level Logger log level
             * @param[in] printers  Printers to push the messages
             */
            BILogger(std::string name, BILogLevel log_level, std::vector<std::shared_ptr<BIPrinter>> printers = {});

            /** Default Constructor
             *
             * @param[in] name       Name of the logger
             * @param[in] log_level  Logger log level
             * @param[in] printers   Printers to push the messages
             * @param[in] decorators Message decorators, which append information in the logged message
             */
            BILogger(std::string name,
                     BILogLevel log_level,
                     std::vector<std::shared_ptr<BIPrinter>> printers,
                     std::vector<std::unique_ptr<BIIDecorator>> decorators);

            /** Allow instances of this class to be moved */
            BILogger(BILogger &&) = default;

            /** Prevent instances of this class from being copied (As this class contains pointers) */
            BILogger(const BILogger &) = delete;

            /** Prevent instances of this class from being copied (As this class contains pointers) */
            BILogger &operator=(const BILogger &) = delete;

            /** Allow instances of this class to be moved */
            BILogger &operator=(BILogger &&) = default;

            /** Logs a message
             *
             * @param[in] log_level Log level of the message
             * @param[in] msg       Message to log
             */
            void log(BILogLevel log_level, const std::string &msg);

            /** Logs a formatted message
             *
             * @param[in] log_level Log level of the message
             * @param[in] fmt       Message format
             * @param[in] args      Message arguments
             */
            template<typename... Ts>
            void log(BILogLevel log_level, const std::string &fmt, Ts &&...args);

            /** Sets log level of the logger
             *
             * @warning Not thread-safe
             *
             * @param[in] log_level Log level to set
             */
            void set_log_level(BILogLevel log_level);

            /** Returns logger's log level
             *
             * @return Logger's log level
             */
            BILogLevel log_level() const;

            /** Returns logger's name
             *
             * @return Logger's name
             */
            std::string name() const;

            /** Adds a printer to the logger
             *
             * @warning Not thread-safe
             *
             * @param[in] printer
             */
            void add_printer(std::shared_ptr<BIPrinter> printer);

            /** Adds a log message decorator to the logger
             *
             * @warning Not thread-safe
             *
             * @param[in] decorator
             */
            void add_decorator(std::unique_ptr<BIIDecorator> decorator);

        private:
            /** Set default message decorators */
            void set_default_decorators();

            /** Checks if a message should be logged depending
             *  on the message log level and the loggers one
             *
             * @param[in] log_level Log level
             *
             * @return True if message should be logged else false
             */
            bool is_loggable(BILogLevel log_level);

            /** Decorate log message
             *
             * @param[in] Log message to decorate
             */
            void decorate_log_msg(BILogMsg &msg);

            /** Creates final log message by creating the prefix
             *
             * @param[in] str       Log message
             * @param[in] log_level Message's log level
             *
             * @return Final log message to print
             */
            std::string create_log_msg(const std::string &str, BILogLevel log_level);

            /** Prints the message to all the printers
             *
             * @param[in] msg Message to print
             */
            void print_all(const std::string &msg);

        private:
            std::string                                _name;
            BILogLevel                                 _log_level;
            std::vector<std::shared_ptr<BIPrinter>>    _printers;
            std::vector<std::unique_ptr<BIIDecorator>> _decorators;
        };

        template<typename... Ts>
        inline void BILogger::log(BILogLevel log_level, const std::string &fmt, Ts &&...args) {
            // Return if message shouldn't be logged
            // i.e. if log level does not match the logger's
            if (!is_loggable(log_level)) {
                return;
            }

            // Print message to all printers
            print_all(create_log_msg(string_with_format(fmt, args...), log_level));
        }
    }
}

#endif //BATMANINFER_BI_LOGGING_LOGGER_HPP
