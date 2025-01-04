//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_I_STD_PRINTER_HPP
#define BATMANINFER_BI_I_STD_PRINTER_HPP

#include <data/core/utils/logging/bi_logging_i_printer.hpp>

#include <iostream>

namespace BatmanInfer {
    namespace logging {
        class BIStdPrinter final : public BIPrinter {
        private:
            void print_internal(const std::string &msg) override {
                std::cout << msg << std::endl;
            }
        };
    }
}

#endif //BATMANINFER_BI_I_STD_PRINTER_HPP
