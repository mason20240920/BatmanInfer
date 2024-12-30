//
// Created by holynova on 2024/12/30.
//
#include "data/core/bi_error.h"

#include <array>
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <stdexcept>

using namespace BatmanInfer;

BIStatus BatmanInfer::create_error(BIErrorCode error_code, std::string msg)
{
    return BIStatus(error_code, msg);
}

BIStatus
BatmanInfer::create_error_msg(BIErrorCode error_code, const char *func, const char *file, int line, const char *msg)
{
    std::array<char, 512> out{0};
    snprintf(out.data(), out.size(), "in %s %s:%d: %s", func, file, line, msg);
    return BIStatus(error_code, std::string(out.data()));
}

void BatmanInfer::throw_error(BIStatus err)
{
    BI_COMPUTE_THROW(std::runtime_error(err.error_description()));
}

void BIStatus::internal_throw_on_error() const
{
    BI_COMPUTE_THROW(std::runtime_error(_error_description));
}


