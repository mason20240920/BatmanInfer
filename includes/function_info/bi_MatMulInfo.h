//
// Created by holynova on 2024/12/31.
//

#ifndef BATMANINFER_BI_MATMULINFO_H
#define BATMANINFER_BI_MATMULINFO_H

namespace BatmanInfer {

    /** Class for holding information related to matrix multiplication function
     */
    class BIMatMulInfo
    {
    public:
        /* Get Adjoint LHS flag value */
        bool adj_lhs() const
        {
            return _adj_lhs;
        }
        /* Get Adjoint RHS flag value */
        bool adj_rhs() const
        {
            return _adj_rhs;
        }
        /* Set Adjoint LHS flag */
        BIMatMulInfo &adj_lhs(bool adj_lhs)
        {
            _adj_lhs = adj_lhs;
            return *this;
        }
        /* Set Adjoint RHS flag */
        BIMatMulInfo &adj_rhs(bool adj_rhs)
        {
            _adj_rhs = adj_rhs;
            return *this;
        }

    private:
        bool _adj_lhs{false};
        bool _adj_rhs{false};
    };

} // namespace BatmanInfer

#endif //BATMANINFER_BI_MATMULINFO_H
