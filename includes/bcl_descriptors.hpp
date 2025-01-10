//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BCL_DESCRIPTORS_HPP
#define BATMANINFER_BCL_DESCRIPTORS_HPP

#ifdef __cplusplus
extern "C"
{
#endif

/**< 支持的激活类型 */
typedef enum {
    BclActivationTypeNone = 0,  /**< No activation */
    BclIdentity = 1,  /**< Identity */
    BclLogistic = 2,  /**< Logistic */
    BclTanh = 3,  /**< Hyperbolic tangent */
    BclRelu = 4,  /**< Rectifier */
    BclBoundedRelu = 5,  /**< Upper Bounded Rectifier */
    BclLuBoundedRelu = 6,  /**< Lower and Upper Bounded Rectifier */
    BclLeakyRelu = 7,  /**< Leaky Rectifier */
    BclSoftRelu = 8,  /**< Soft Rectifier */
    BclElu = 9,  /**< Exponential Linear Unit */
    BclAbs = 10, /**< Absolute */
    BclSquare = 11, /**< Square */
    BclSqrt = 12, /**< Square root */
    BclLinear = 13, /**< Linear */
    BclHardSwish = 14, /**< Hard-swish */
} BclActivationType;

typedef struct {
    BclActivationType type;    /**< Activation type */
    float a;       /**< Factor &alpha used by some activations */
    float b;       /**< Factor &beta used by some activations */
    bool inplace; /**< Hint that src and dst tensors will be the same */
} BclActivationDescriptor;

#ifdef __cplusplus
};
#endif

#endif //BATMANINFER_BCL_DESCRIPTORS_HPP
