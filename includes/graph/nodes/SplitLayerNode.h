//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_SPLITLAYERNODE_H
#define BATMANINFER_GRAPH_SPLITLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Split Layer node */
class SplitLayerNode final : public BIINode
{
public:
    /** Default Constructor
     *
     * @param[in] num_splits  Number of splits
     * @param[in] axis        (Optional) Axis to split on. Defaults to 0
     * @param[in] size_splits (Optional) The sizes of each output tensor along the split dimension.
     *                        Must sum to the dimension of value along split_dim.
     *                        Can contain one -1 indicating that dimension is to be inferred.
     */
    SplitLayerNode(unsigned int num_splits, int axis = 0, std::vector<int> size_splits = std::vector<int>());
    /** Computes split layer output descriptor
     *
     * @param[in] input_descriptor Descriptor of the input tensor
     * @param[in] num_splits       Number of splits
     * @param[in] axis             Axis to perform the split on
     * @param[in] idx              Index of the split
     *
     * @return  A pair with the descriptor of the split and the starting coordinates
     */
    std::pair<BITensorDescriptor, BICoordinates> compute_output_descriptor(const BITensorDescriptor &input_descriptor,
                                                                           unsigned int            num_splits,
                                                                           int                     axis,
                                                                           unsigned int            idx);
    /** Number of splits accessor
     *
     * @return Number of splits
     */
    unsigned int num_splits() const;
    /** Split axis accessor
     *
     * @return Split axis
     */
    unsigned int axis() const;

    // Inherited overridden methods:
    BIStatus           validate() const override;
    BINodeType         type() const override;
    bool               forward_descriptors() override;
    BITensorDescriptor configure_output(size_t idx) const override;
    void               accept(BIINodeVisitor &v) override;

private:
    unsigned int     _num_splits;
    int              _axis;
    std::vector<int> _size_splits;
};

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_SPLITLAYERNODE_H
