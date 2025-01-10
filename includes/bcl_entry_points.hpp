//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BCL_ENTRY_POINTS_HPP
#define BATMANINFER_BCL_ENTRY_POINTS_HPP

#include "bcl_types.hpp"

#ifdef __cplusplus
extern "C"
{
#endif /** __cplusplus */

/**
 * 创建上下文对象
 *
 * 上下文负责保存内部信息，并作为服务机制的聚合。
 *
 * @param ctx 如果操作成功，将返回一个有效的非零上下文对象
 * @param target 要为其创建上下文的目标
 * @param options 用于该上下文下创建的所有内核的上下文选项
 * @return 状态码
 *
 * 返回值
 *  - @ref BclSuccess 如果函数成功完成。
 *  - @ref BclOutOfMemory 如果分配内存资源失败。
 *  - @ref BclUnsupportedTarget 如果请求的目标不受支持。
 *  - @ref BclInvalidArgument 如果提供的参数无效
 */
BclStatus BclCreateContext(BclContext *ctx, BclTarget target, const BclContextOptions *options);

/**
 * 销毁一个给定的上下文对象
 *
 * @param ctx 合法的上下文对象
 * @return 返回状态
 *
 * 返回值
 *  - @ref BclSuccess 如果函数成功完成。
 *  - @ref BclInvalidArgument
 */
BclStatus BclDestroyContext(BclContext ctx);

/**
 * 创建操作队列
 *
 * 队列负责调度相关的活动。
 *
 * @param queue 如果操作成功，将返回一个有效的非零队列对象。
 * @param ctx 要使用的上下文。
 * @param options 用于队列操作的选项。
 * @return 状态码
 *
 * 返回值：
 *  - @ref BclSuccess 如果函数成功完成。
 *  - @ref BclOutOfMemory 如果分配内存资源失败。
 *  - @ref BclUnsupportedTarget 如果请求的目标不受支持。
 *  - @ref BclInvalidArgument 如果提供的参数无效。
 */
BclStatus BclCreateQueue(BclQueue *queue, BclContext ctx, const BclQueueOptions *options);

/** Wait until all elements on the queue have been completed
 *
 * @param[in] queue Queue to wait on completion
 *
 * @return Status code
 *
 * Returns:
 *  - @ref BclSuccess if functions was completed successfully
 *  - @ref BclInvalidArgument if the provided queue is invalid
 *  - @ref BclRuntimeError on any other runtime related error
 */
BclStatus BclQueueFinish(BclQueue queue);

/** Destroy a given queue object
*
* @param[in] queue A valid context object to destroy
*
* @return Status code
*
* Returns:
*  - @ref BclSuccess if functions was completed successfully
*  - @ref BclInvalidArgument if the provided context is invalid
*/
BclStatus BclDestroyQueue(BclQueue queue);

/** Create a Tensor object
*
* Tensor is a generalized matrix construct that can represent up to ND dimensionality (where N = 6 for Compute Library)
* The object holds a backing memory along-side to operate on
*
* @param[in, out] tensor   A valid non-zero tensor object if no failures occur
* @param[in]      ctx      Context to be used
* @param[in]      desc     Tensor representation meta-data
* @param[in]      allocate Instructs allocation of the tensor objects
*
* Returns:
*  - @ref BclSuccess if function was completed successfully
*  - @ref BclOutOfMemory if there was a failure allocating memory resources
*  - @ref BclUnsupportedTarget if the requested target is unsupported
*  - @ref BclInvalidArgument if a given argument is invalid
*/
BclStatus BclCreateTensor(BclTensor *tensor, BclContext ctx, const BclTensorDescriptor *desc, bool allocate);

/** Map a tensor's backing memory to the host
*
* @param[in]      tensor Tensor to be mapped
* @param[in, out] handle A handle to the underlying backing memory
*
* @return Status code
*
* Returns:
*  - @ref BclSuccess if function was completed successfully
*  - @ref BclInvalidArgument if a given argument is invalid
*/
BclStatus BclMapTensor(BclTensor tensor, void **handle);

/** Unmap the tensor's backing memory
*
* @param[in] tensor tensor to unmap memory from
* @param[in] handle Backing memory to be unmapped
*
* @return Status code
*
* Returns:
*  - @ref BclSuccess if function was completed successfully
*  - @ref BclInvalidArgument if a given argument is invalid
*/
BclStatus BclUnmapTensor(BclTensor tensor, void *handle);

/** Import external memory to a given tensor object
*
* @param[in, out] tensor Tensor to import memory to
* @param[in]      handle Backing memory to be imported
* @param[in]      type   Type of the imported memory
*
* Returns:
*  - @ref BclSuccess if function was completed successfully
*  - @ref BclInvalidArgument if a given argument is invalid
*/
BclStatus BclTensorImport(BclTensor tensor, void *handle, BclImportMemoryType type);

/** Destroy a given tensor object
*
* @param[in,out] tensor A valid tensor object to be destroyed
*
* @return Status code
*
* Returns:
*  - @ref BclSuccess if function was completed successfully
*  - @ref BclInvalidArgument if the provided tensor is invalid
*/
BclStatus BclDestroyTensor(BclTensor tensor);

/** Creates a tensor pack
*
* Tensor packs are used to create a collection of tensors that can be passed around for operator execution
*
* @param[in,out] pack A valid non-zero tensor pack object if no failures occur
* @param[in]     ctx  Context to be used
*
* @return Status code
*
* Returns:
*  - @ref BclSuccess if function was completed successfully
*  - @ref BclOutOfMemory if there was a failure allocating memory resources
*  - @ref BclInvalidArgument if a given argument is invalid
*/
BclStatus BclCreateTensorPack(BclTensorPack *pack, BclContext ctx);

/** Add a tensor to a tensor pack
*
* @param[in,out] pack    Pack to append a tensor to
* @param[in]     tensor  Tensor to pack
* @param[in]     slot_id Slot of the operator that the tensors corresponds to
*
* @return Status code
*
* Returns:
*  - @ref BclSuccess if function was completed successfully
*  - @ref BclOutOfMemory if there was a failure allocating memory resources
*  - @ref BclInvalidArgument if a given argument is invalid
*/
BclStatus BclPackTensor(BclTensorPack pack, BclTensor tensor, int32_t slot_id);

/** A list of tensors to a tensor pack
*
* @param[in,out] pack        Pack to append the tensors to
* @param[in]     tensors     Tensors to append to the pack
* @param[in]     slot_ids    Slot IDs of each tensors to the operators
* @param[in]     num_tensors Number of tensors that are passed
*
* @return Status code
*
* Returns:
*  - @ref BclSuccess if function was completed successfully
*  - @ref BclOutOfMemory if there was a failure allocating memory resources
*  - @ref BclInvalidArgument if a given argument is invalid
*/
BclStatus BclPackTensors(BclTensorPack pack, BclTensor *tensors, int32_t *slot_ids, size_t num_tensors);

/** Destroy a given tensor pack object
*
* @param[in,out] pack A valid tensor pack object to destroy
*
* @return Status code
*
* Returns:
*  - @ref BclSuccess if functions was completed successfully
*  - @ref BclInvalidArgument if the provided context is invalid
*/
BclStatus BclDestroyTensorPack(BclTensorPack pack);

/** Eager execution of a given operator on a list of inputs and outputs
*
* @param[in]     op      Operator to execute
* @param[in]     queue   Queue to schedule the operator on
* @param[in,out] tensors A list of input and outputs tensors to execute the operator on
*
* @return Status Code
*
* Returns:
*  - @ref BclSuccess if function was completed successfully
*  - @ref BclOutOfMemory if there was a failure allocating memory resources
*  - @ref BclUnsupportedTarget if the requested target is unsupported
*  - @ref BclInvalidArgument if a given argument is invalid
*  - @ref BclRuntimeError on any other runtime related error
*/
BclStatus BclRunOperator(BclOperator op, BclQueue queue, BclTensorPack tensors);

/** Destroy a given operator object
*
* @param[in,out] op A valid operator object to destroy
*
* @return Status code
*
* Returns:
*  - @ref BclSuccess if functions was completed successfully
*  - @ref BclInvalidArgument if the provided context is invalid
*/
BclStatus BclDestroyOperator(BclOperator op);

#ifdef __cplusplus
}
#endif /** __cplusplus */

#endif //BATMANINFER_BCL_ENTRY_POINTS_HPP
