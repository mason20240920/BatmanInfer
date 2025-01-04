//
// Created by Mason on 2025/1/4.
//

#include <data/core/utils/logging/bi_i_file_printer.hpp>

using namespace BatmanInfer::logging;

BIFilePrinter::BIFilePrinter(const std::string &filename) {
    _handler.open(filename, std::fstream::out | std::fstream::trunc);
}

void BIFilePrinter::print_internal(const std::string &msg) {
    _handler.stream() << msg << std::endl;
}