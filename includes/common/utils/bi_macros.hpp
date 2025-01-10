//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BI_COMMON_MACROS_HPP
#define BATMANINFER_BI_COMMON_MACROS_HPP

#include <common/utils/bi_utils.hpp>


#define BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status) \
   {                                                               \
        if (status != BatmanInfer::StatusCode::Success)             \
        {                                                           \
            return BatmanInfer::utils::as_cenum<BclStatus>(status); \
        }                                                           \
    }
#endif //BATMANINFER_BI_COMMON_MACROS_HPP
