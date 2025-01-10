//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BCL_UTILS_HPP
#define BATMANINFER_BCL_UTILS_HPP

#include <bcl_types.hpp>

#ifdef __cplusplus
extern "C"
{
#endif /** __cplusplus */

/** Get the size of the existing tensor in byte
*
* @note The size isn't based on allocated memory, but based on information in its descriptor (dimensions, data type, etc.).
*
* @param[in]  tensor A tensor in interest
* @param[out] size   The size of the tensor
*
* @return Status code
*
*  - @ref BclSuccess if function was completed successfully
*  - @ref BclInvalidArgument if a given argument is invalid
*/

BclStatus BclGetTensorSize(BclTensor tensor, uint64_t *size);

/**
 * 获取张量信息描述
 * @param tensor
 * @param desc
 * @return Status code
 *
 *  - @ref BclSuccess if function was completed successfully
 *  - @ref BclInvalidArgument if a given argument is invalid
 */
BclStatus BclGetTensorDescriptor(BclTensor tensor, BclTensorDescriptor *desc);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif //BATMANINFER_BCL_UTILS_HPP
