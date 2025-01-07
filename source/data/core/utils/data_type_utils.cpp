//
// Created by Mason on 2025/1/7.
//

#include <data/core/utils/data_type_utils.hpp>

namespace BatmanInfer {
    const std::string &string_from_data_type(BIDataType dt) {
        static std::map<BIDataType, const std::string> dt_map = {
                {BIDataType::UNKNOWN,            "UNKNOWN"},
                {BIDataType::S8,                 "S8"},
                {BIDataType::U8,                 "U8"},
                {BIDataType::S16,                "S16"},
                {BIDataType::U16,                "U16"},
                {BIDataType::S32,                "S32"},
                {BIDataType::U32,                "U32"},
                {BIDataType::S64,                "S64"},
                {BIDataType::U64,                "U64"},
                {BIDataType::F16,                "F16"},
                {BIDataType::F32,                "F32"},
                {BIDataType::F64,                "F64"},
                {BIDataType::SIZET,              "SIZET"},
                {BIDataType::QSYMM8,             "QSYMM8"},
                {BIDataType::QSYMM8_PER_CHANNEL, "QSYMM8_PER_CHANNEL"},
                {BIDataType::QASYMM8,            "QASYMM8"},
                {BIDataType::QASYMM8_SIGNED,     "QASYMM8_SIGNED"},
                {BIDataType::QSYMM16,            "QSYMM16"},
                {BIDataType::QASYMM16,           "QASYMM16"},
        };

        return dt_map[dt];
    }
}