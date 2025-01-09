//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_BI_CPP_VALIDATE_HPP
#define BATMANINFER_BI_CPP_VALIDATE_HPP

#include <data/core/bi_vlidate.hpp>
#include <data/core/cpp/cpp_types.hpp>

namespace BatmanInfer {

    /**
     * 如果传递的张量信息的类型是 FP16，并且未编译 FP16 支持，则返回错误。
     * @param function
     * @param file
     * @param line
     * @param tensor_info
     * @return
     */
    inline BIStatus error_on_unsupported_cpu_fp16(const char *function,
                                                  const char *file,
                                                  const int line,
                                                  const BIITensorInfo *tensor_info) {
        bool fp16_kernels_enabled = false;
#if defined(BI_COMPUTE_ENABLE_FP16) && defined(ENABLE_FP16_KERNELS)
        fp16_kernels_enabled = true;
#endif /* defined(ARM_COMPUTE_ENABLE_FP16) && defined(ENABLE_FP16_KERNELS) */

        BI_COMPUTE_RETURN_ERROR_ON_LOC(tensor_info == nullptr, function, file, line);
        BI_COMPUTE_RETURN_ERROR_ON_LOC_MSG(
                (tensor_info->data_type() == BIDataType::F16) &&
                (!CPUInfo::get().has_fp16() || !fp16_kernels_enabled),
                function,
                file, line, "This CPU architecture does not support F16 data type, you need v8.2 or above");
        return BIStatus{};
    }

#define BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(tensor) \
    BI_COMPUTE_RETURN_ON_ERROR(::BatmanInfer::error_on_unsupported_cpu_fp16(__func__, __FILE__, __LINE__, tensor))
}

#endif //BATMANINFER_BI_CPP_VALIDATE_HPP
