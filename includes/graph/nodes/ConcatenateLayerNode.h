//
// Created by holynova on 2025/1/3.
//

#ifndef CONCATENATELAYERNODE_H
#define CONCATENATELAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Concatenation Layer node */
    class ConcatenateLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] total_nodes       Number of nodes that will get concatenated
         * @param[in] concat_descriptor Concatenate Layer Descriptor
         */
        ConcatenateLayerNode(unsigned int total_nodes, descriptors::ConcatLayerDescriptor concat_descriptor);
        /** Computes concatenations output descriptor
         *
         * @param[in] input_descriptors Input descriptors
         * @param[in] axis              Concatenation axis
         *
         * @return Expected output descriptor
         */
        static BITensorDescriptor compute_output_descriptor(const std::vector<BITensorDescriptor> &input_descriptors,
                                                            BIDataLayoutDimension                  axis);
        /** Disables or not the depth concatenate node
         *
         * @warning This is used when concatenate is performed using sub-tensors, where this node is used as a placeholder.
         *
         * @param[in] is_enabled If true a backend function is created to perform the concatenation (involves copying),
         *                       while if false, no function is created and we assume that sub-tensors are properly set to simulate
         *                       a zero copy operation.
         */
        void set_enabled(bool is_enabled);
        /** Enabled parameter accessor
         *
         * @return True if a backend function is to be created else false
         */
        bool is_enabled() const;
        /** Concatenation axis parameter accessor
         *
         * @return Concatenation axis
         */
        BIDataLayoutDimension concatenation_axis() const;

        /** Concatenation output quantization info accessor
         *
         * @return Output quantization info
         */
        BIQuantizationInfo output_quantization_info() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        unsigned int                         _total_nodes;
        descriptors::ConcatLayerDescriptor _concat_descriptor;
        bool                                 _is_enabled;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //CONCATENATELAYERNODE_H
