//
// Created by holynova on 2024/12/31.
//

#ifndef BATMANINFER_BI_SCATTERINFO_H
#define BATMANINFER_BI_SCATTERINFO_H

#include "data/core/bi_error.h"

namespace BatmanInfer {

    /** Scatter Function */
    enum class BIScatterFunction
    {
        Update = 0,
        Add    = 1,
        Sub    = 2,
        Max    = 3,
        Min    = 4
    };
    /** Scatter operator information */
    struct BIScatterInfo
    {
        BIScatterInfo(BIScatterFunction f, bool zero) : func(f), zero_initialization(zero)
        {
            BI_COMPUTE_ERROR_ON_MSG(f != ScatterFunction::Add && zero,
                                     "Zero initialisation is only supported with Add Scatter Function.");
        }
        BIScatterFunction func;            /**< Type of scatter function to use with scatter operator*/
        bool zero_initialization{false};   /**< Fill output tensors with 0. Only available with add scatter function. */
    };

} // namespace BatmanInfer

#endif //BATMANINFER_BI_SCATTERINFO_H
