//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BI_COMMON_VALIDATE_HPP
#define BATMANINFER_BI_COMMON_VALIDATE_HPP

#if defined(BI_COMPUTE_ASSERTS_ENABLED)

#include <cassert>

#define BI_COMPUTE_ASSERT(cond)            assert(cond)
#define BI_COMPUTE_ASSERT_NOT_NULLPTR(ptr) assert((ptr) != nullptr)

#else /* defined(ARM_COMPUTE_ASSERTS_ENABLED) */

#define BI_COMPUTE_ASSERT(cond)
#define BI_COMPUTE_ASSERT_NOT_NULLPTR(ptr)

#endif /* defined(ARM_COMPUTE_ASSERTS_ENABLED) */

#endif //BATMANINFER_BI_COMMON_VALIDATE_HPP
