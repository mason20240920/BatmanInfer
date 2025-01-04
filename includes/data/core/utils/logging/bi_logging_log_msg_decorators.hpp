//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_LOGGING_LOG_MSG_DECORATORS_HPP
#define BATMANINFER_BI_LOGGING_LOG_MSG_DECORATORS_HPP

#include <data/core/bi_error.h>
#include <data/core/utils/logging/bi_logging_helpers.hpp>
#include <data/core/utils/logging/bi_logging_types.hpp>

#include <chrono>
#include <ctime>
#include <string>

#ifndef NO_MULTI_THREADING

#include <thread>

#endif

namespace BatmanInfer {
    namespace logging {
        /**
         * @brief 日志信息装饰器接口
         */
        class BIIDecorator {
        public:
            virtual ~BIIDecorator() = default;

            /**
             * @brief 修饰日志信息
             * @param log_msg
             */
            virtual void decorate(BILogMsg &log_msg) = 0;
        };

        /**
         * @brief 字符串装饰器
         *
         * 增加用户定义的字符串到日志信息里
         */
        class BIStringDecorator : public BIIDecorator {
        public:
            BIStringDecorator(const std::string &str) : _str(str) {
                _str = angle_wrap_value(str);
            }

            void decorate(BatmanInfer::logging::BILogMsg &log_msg) override {
                log_msg.raw_ += _str;
            }

        private:
            std::string _str;
        };

        /**
         * @brief 日期装饰器
         *
         * 在日志信息里面新增日期
         */
        class BIDateDecorator : public BIIDecorator {
        public:
            void decorate(BatmanInfer::logging::BILogMsg &log_msg) override {
                log_msg.raw_ += angle_wrap_value(get_time());
            }

        private:
            /**
             * @brief 获取当前系统时间
             * @return
             */
            std::string get_time() {
                auto now  = std::chrono::system_clock::now();
                auto time = std::chrono::system_clock::to_time_t(now);

                // TODO: use put_time for gcc > 4.9
                char buf[100] = {0};
                std::strftime(buf, sizeof(buf), "%d-%m-%Y %I:%M:%S", std::localtime(&time));
                return buf;
            }
        };

        class BIThreadIdDecorator : public BIIDecorator {
        public:
            // Inherited methods overridden:
            void decorate(BILogMsg &log_msg) override {
#ifndef NO_MULTI_THREADING
                log_msg.raw_ += angle_wrap_value(std::this_thread::get_id());
#else  /* NO_MULTI_THREADING */
                BI_COMPUTE_UNUSED(log_msg);
#endif /* NO_MULTI_THREADING */
            }
        };

        class BILogLevelDecorator : public BIIDecorator {
        public:
            void decorate(BatmanInfer::logging::BILogMsg &log_msg) override {
                log_msg.raw_ += angle_wrap_value(string_from_log_level(log_msg.log_level_));
            }
        };
    }
}

#endif //BATMANINFER_BI_LOGGING_LOG_MSG_DECORATORS_HPP
