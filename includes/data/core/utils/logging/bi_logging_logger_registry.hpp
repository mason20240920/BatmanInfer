//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_LOGGING_LOGGER_REGISTRY_HPP
#define BATMANINFER_BI_LOGGING_LOGGER_REGISTRY_HPP

#include <data/core/utils/logging/bi_logging_logger.hpp>
#include <data/core/utils/logging/bi_i_printers.hpp>
#include <data/core/utils/logging/bi_logging_types.hpp>

#include <support/mutex.hpp>

#include <memory>
#include <set>
#include <unordered_map>

namespace BatmanInfer {
    namespace logging {
        /**
         * @brief 注册表类保存所有已实例化的日志记录器。
         */
        class BILoggerRegistry final {
        public:
            /**
             * @brief 获取注册表实例
             * @return
             */
            static BILoggerRegistry &get();

            /** Creates a logger
    *
    * @note Some names are reserved e.g. [CORE, RUNTIME, GRAPH]
    *
    * @param[in] name      Logger's name
    * @param[in] log_level Logger's log level. Defaults to INFO
    * @param[in] printers  Printers to attach to the system loggers. Defaults with a @ref StdPrinter.
    */
            void create_logger(const std::string &name,
                               BILogLevel log_level = BILogLevel::INFO,
                               const std::vector<std::shared_ptr<BIPrinter>> &printers = {
                                       std::make_shared<BIStdPrinter>()});

            /** Remove a logger
             *
             * @param name Logger's name
             */
            void remove_logger(const std::string &name);

            /** Returns a logger instance
             *
             * @param[in] name Logger to return
             *
             * @return Logger
             */
            std::shared_ptr<BILogger> logger(const std::string &name);

            /** Creates reserved library loggers
             *
             * @param[in] log_level (Optional) Logger's log level. Defaults to INFO
             * @param[in] printers  (Optional) Printers to attach to the system loggers. Defaults with a @ref StdPrinter.
             */
            void create_reserved_loggers(BILogLevel log_level = BILogLevel::INFO,
                                         const std::vector<std::shared_ptr<BIPrinter>> &printers = {
                                                 std::make_shared<BIStdPrinter>()});

        private:
            /**
             * @brief 默认构造函数
             */
            BILoggerRegistry();

        private:
            BatmanInfer::Mutex                                         _mtx;
            std::unordered_map<std::string, std::shared_ptr<BILogger>> _loggers;
            static std::set<std::string>                               _reserved_loggers;
        };
    }
}

#endif //BATMANINFER_BI_LOGGING_LOGGER_REGISTRY_HPP
