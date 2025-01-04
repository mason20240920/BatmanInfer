//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_LOGGING_TYPES_HPP
#define BATMANINFER_BI_LOGGING_TYPES_HPP

#include <string>

namespace BatmanInfer {
    namespace logging {
        enum class BILogLevel {
            /**
             * @brief 所有日志信息
             */
            VERBOSE,
            /**
             * @brief 信息日志层级
             */
            INFO,
            /**
             * @brief 提醒日志
             */
            WARN,
            ERROR,
            OFF
        };

        /**
         * @brief 日志信息
         */
        struct BILogMsg {
            BILogMsg() : raw_(), log_level_(BILogLevel::OFF) {

            }

            /**
             * @brief 构造函数
             * @param msg
             * @param log_level
             */
            explicit BILogMsg(std::string msg, BILogLevel log_level = BILogLevel::OFF) : raw_(msg),
                                                                                         log_level_(log_level) {

            }


            /**
             * @brief 日志信息
             */
            std::string raw_;

            /**
             * @brief 日志级别，默认关闭
             */
            BILogLevel log_level_;
        };
    }
}

#endif //BATMANINFER_BI_LOGGING_TYPES_HPP
