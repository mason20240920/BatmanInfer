//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BCL_TYPES_HPP
#define BATMANINFER_BCL_TYPES_HPP

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

/**< Opaque Context object */
typedef struct BclContext_ *BclContext;
/**< Opaque Queue object */
typedef struct BclQueue_ *BclQueue;
/**< Opaque Tensor object */
typedef struct BclTensor_ *BclTensor;
/**< Opaque Tensor pack object */
typedef struct BclTensorPack_ *BclTensorPack;
/**< Opaque Operator object */
typedef struct BclOperator_ *BclOperator;

// Capabilities bitfield (Note: if multiple are enabled ComputeLibrary will pick the best possible)
typedef uint64_t BclTargetCapabilities;

/**< Error codes returned by the public entry-points */
typedef enum BclStatus : int32_t {
    BclSuccess = 0, /**< Call succeeded, leading to valid state for all involved objects/data */
    BclRuntimeError = 1, /**< Call failed during execution */
    BclOutOfMemory = 2, /**< Call failed due to failure to allocate resources */
    BclUnimplemented = 3, /**< Call failed as requested capability is not implemented */
    BclUnsupportedTarget = 4, /**< Call failed as an invalid backend was requested */
    BclInvalidTarget = 5, /**< Call failed as invalid argument was passed */
    BclInvalidArgument = 6, /**< Call failed as invalid argument was passed */
    BclUnsupportedConfig = 7, /**< Call failed as configuration is unsupported */
    BclInvalidObjectState = 8, /**< Call failed as an object has invalid state */
} BclStatus;

/**< Supported CPU targets */
typedef enum BclTarget {
    BclCpu = 0, /**< Cpu target that uses SIMD extensions */
    BclGpuOcl = 1, /**< OpenCL target for GPU */
} BclTarget;

/** Execution mode types */
typedef enum BclExecutionMode {
    BclPreferFastRerun = 0, /**< Prioritize performance when multiple iterations are performed */
    BclPreferFastStart = 1, /**< Prioritize performance when a single iterations is expected to be performed */
} BclExecutionMode;

/** Available CPU capabilities */
typedef enum BclCpuCapabilities {
    BclCpuCapabilitiesAuto = 0, /**< Automatic discovery of capabilities */

    BclCpuCapabilitiesNeon = (1 << 0), /**< Enable NEON optimized paths */
    BclCpuCapabilitiesSve = (1 << 1), /**< Enable SVE optimized paths */
    BclCpuCapabilitiesSve2 = (1 << 2), /**< Enable SVE2 optimized paths */
    // Reserve 3, 4, 5, 6

    BclCpuCapabilitiesFp16 = (1 << 7), /**< Enable float16 data-type support */
    BclCpuCapabilitiesBf16 = (1 << 8), /**< Enable bfloat16 data-type support */
    // Reserve 9, 10, 11, 12

    BclCpuCapabilitiesDot = (1 << 13), /**< Enable paths that use the udot/sdot instructions */
    BclCpuCapabilitiesMmlaInt8 = (1 << 14), /**< Enable paths that use the mmla integer instructions */
    BclCpuCapabilitiesMmlaFp = (1 << 15), /**< Enable paths that use the mmla float instructions */

    BclCpuCapabilitiesAll = ~0 /**< Enable all paths */
} BclCpuCapabilities;

/**< Allocator interface that can be passed to a context */
typedef struct BclAllocator {
    /** Allocate a block of size bytes of memory.
 *
 * @param[in] user_data User provided data that can be used by the allocator
 * @param[in] size      Size of the allocation
 *
 * @return A pointer to the allocated block if successfull else NULL
 */
    void *(*alloc)(void *user_data, size_t size);

    /** Release a block of size bytes of memory.
 *
 * @param[in] user_data User provided data that can be used by the allocator
 * @param[in] size      Size of the allocation
 */
    void (*free)(void *user_data, void *ptr);

    /** Allocate a block of size bytes of memory.
 *
 * @param[in] user_data User provided data that can be used by the allocator
 * @param[in] size      Size of the allocation
 *
 * @return A pointer to the allocated block if successfull else NULL
 */
    void *(*aligned_alloc)(void *user_data, size_t size, size_t alignment);

    /** Allocate a block of size bytes of memory.
 *
 * @param[in] user_data User provided data that can be used by the allocator
 * @param[in] size      Size of the allocation
 */
    void (*aligned_free)(void *user_data, void *ptr);

    /**< User provided information */
    void *user_data;
} BclAllocator;

/**< Context options */
typedef struct BclContextOptions {
    BclExecutionMode mode;               /**< Execution mode to use */
    BclTargetCapabilities capabilities;       /**< Target capabilities */
    bool enable_fast_math;   /**< Allow precision loss */
    const char *kernel_config_file; /**< Kernel cofiguration file */
    int32_t max_compute_units; /**< Max compute units that can be used by a queue created from the context.
                                                   If <=0 the system will use the hw concurency insted */
    BclAllocator *allocator;         /**< Allocator to be used by all the memory internally */
} BclContextOptions;

/**< Supported tuning modes */
typedef enum {
    BclTuningModeNone = 0, /**< No tuning */
    BclRapid = 1, /**< Fast tuning mode, testing a small portion of the tuning space */
    BclNormal = 2, /**< Normal tuning mode, gives a good balance between tuning mode and performance */
    BclExhaustive = 3, /**< Exhaustive tuning mode, increased tuning time but with best results */
} BclTuningMode;

/**< Queue options */
typedef struct {
    BclTuningMode mode;          /**< Tuning mode */
    int32_t compute_units; /**< Compute Units that the queue will deploy */
} BclQueueOptions;

/**< Supported data types */
typedef enum BclDataType {
    BclDataTypeUnknown = 0, /**< Unknown data type */
    BclUInt8 = 1, /**< 8-bit unsigned integer */
    BclInt8 = 2, /**< 8-bit signed integer */
    BclUInt16 = 3, /**< 16-bit unsigned integer */
    BclInt16 = 4, /**< 16-bit signed integer */
    BclUint32 = 5, /**< 32-bit unsigned integer */
    BclInt32 = 6, /**< 32-bit signed integer */
    BclFloat16 = 7, /**< 16-bit floating point */
    BclBFloat16 = 8, /**< 16-bit brain floating point */
    BclFloat32 = 9, /**< 32-bit floating point */
} BclDataType;

/**< Supported data layouts for operations */
typedef enum BclDataLayout {
    BclDataLayoutUnknown = 0, /**< Unknown data layout */
    BclNhwc = 1, /**< Native, performant, Compute Library data layout */
    BclNchw = 2, /**< Data layout where width is the fastest changing dimension */
} BclDataLayout;

/** Type of memory to be imported */
typedef enum BclImportMemoryType {
    BclHostPtr = 0 /**< Host allocated memory */
} BclImportMemoryType;

/**< Tensor Descriptor */
typedef struct BclTensorDescriptor {
    int32_t ndims;     /**< Number or dimensions */
    int32_t *shape;     /**< Tensor Shape */
    BclDataType data_type; /**< Tensor Data type */
    int64_t *strides;   /**< Strides on each dimension. Linear memory is assumed if nullptr */
    int64_t boffset;   /**< Offset in terms of bytes for the first element */
} BclTensorDescriptor;

/**< Slot type of a tensor */
typedef enum {
    BclSlotUnknown = -1,
    BclSrc = 0,
    BclSrc0 = 0,
    BclSrc1 = 1,
    BclDst = 30,
    BclSrcVec = 256,
} BclTensorSlot;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif //BATMANINFER_BCL_TYPES_HPP
