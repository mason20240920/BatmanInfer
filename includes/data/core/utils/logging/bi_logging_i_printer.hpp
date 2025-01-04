//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_LOGGING_I_PRINTER_HPP
#define BATMANINFER_BI_LOGGING_I_PRINTER_HPP

#include <support/mutex.hpp>

namespace BatmanInfer {
    namespace logging {
        // 基础打印机类，用于被其他打印机类继承
        class BIPrinter {
        public:
            BIPrinter() noexcept: _mtx() {

            }

            BIPrinter(const BIPrinter &) = delete;

            BIPrinter &operator=(const BIPrinter &) = delete;

            BIPrinter(BIPrinter &&) = delete;

            BIPrinter &operator=(BIPrinter &&) = delete;

            virtual ~BIPrinter() = default;

            inline void print(const std::string &msg) {
                BatmanInfer::lock_guard<BatmanInfer::Mutex> lock(_mtx);
                print_internal(msg);
            }

        private:
            /**
             * @brief 被子类实现的接口: 打印信息
             * @param msg 信息被打印
             */
            virtual void print_internal(const std::string &msg) = 0;

        private:
            BatmanInfer::Mutex _mtx;
        };
    }
}

#endif //BATMANINFER_BI_LOGGING_I_PRINTER_HPP
