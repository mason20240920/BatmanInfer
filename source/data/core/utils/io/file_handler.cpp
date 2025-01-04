//
// Created by Mason on 2025/1/4.
//

#include <data/core/utils/io/file_handler.hpp>

#include <data/core/bi_error.h>

#include <string>

using namespace BatmanInfer::io;

BIFileHandler::BIFileHandler() : _filestream(), _filename(" "), _mode() {
}

BIFileHandler::~BIFileHandler() {
    close();
}

void BIFileHandler::open(const std::string &filename, std::ios_base::openmode mode) {
    close();
    _filestream.open(filename, mode);
    BI_COMPUTE_ERROR_ON(!_filestream.good());
    _filename = filename;
    _mode     = mode;
}

void BIFileHandler::close() {
    _filestream.close();
}

std::fstream &BIFileHandler::stream() {
    return _filestream;
}

std::string BIFileHandler::filename() const {
    return _filename;
}
