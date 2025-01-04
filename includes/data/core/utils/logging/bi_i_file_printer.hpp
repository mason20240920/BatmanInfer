//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_I_FILE_PRINTER_HPP
#define BATMANINFER_BI_I_FILE_PRINTER_HPP

#include <data/core/utils/io/file_handler.hpp>
#include <data/core/utils/logging/bi_logging_i_printer.hpp>

namespace BatmanInfer {
    namespace logging {
        class BIFilePrinter final : public BIPrinter {
        public:
            explicit BIFilePrinter(const std::string &filename);

        private:
            void print_internal(const std::string &msg) override;

        private:
            io::BIFileHandler _handler;
        };
    }
}


#endif //BATMANINFER_BI_I_FILE_PRINTER_HPP
