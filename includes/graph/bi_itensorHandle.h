//
// Created by holynova on 2025/1/6.
//

#ifndef BATMANINFER_GRAPH_BI_ITENSORHANDLE_H
#define BATMANINFER_GRAPH_BI_ITENSORHANDLE_H

#include "data/core/bi_i_tensor.hpp"
#include "graph/bi_types.h"

namespace BatmanInfer {

// Forward declarations
class BIIMemoryGroup;

namespace graph {

    /** Tensor handle interface object */
    class BIITensorHandle
    {
    public:
        /** Default virtual destructor */
        virtual ~BIITensorHandle() = default;
        /** Allocates backend memory for the handle */
        virtual void allocate() = 0;
        /** Allocates backend memory for the handle */
        virtual void free() = 0;
        /** Set backend tensor to be managed by a memory group
         *
         * @param[in] mg Memory group
         */
        virtual void manage(BIIMemoryGroup *mg) = 0;
        /** Maps backend tensor object
         *
         * @param[in] blocking Flags if the mapping operations should be blocking
         */
        virtual void map(bool blocking) = 0;
        /** Un-maps a backend tensor object */
        virtual void unmap() = 0;
        /** Releases backend tensor if is marked as unused
         *
         *
         * @note This has no effect on sub-tensors
         * @warning Parent tensors don't keep track of sub-tensors,
         *          thus if a parent is set as unused then all sub-tensors will be invalidated,
         *          on the other hand if a sub-tensor is marked as unused then the parent tensor won't be released
         */
        virtual void release_if_unused() = 0;
        /** Backend tensor object accessor */
        virtual BatmanInfer::BIITensor &tensor() = 0;
        /** Backend tensor object const accessor */
        virtual const BatmanInfer::BIITensor &tensor() const = 0;
        /** Return the parent tensor handle if is a subtensor else this
         *
         * @return Parent tensor handle
         */
        virtual BIITensorHandle *parent_handle() = 0;
        /** Checks if a backing tensor is a sub-tensor object or not
         *
         * @return True if the backend tensor is a sub-tensor else false
         */
        virtual bool is_subtensor() const = 0;
        /** Returns target type
         *
         * @return Target type
         */
        virtual BITarget target() const = 0;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_ITENSORHANDLE_H
