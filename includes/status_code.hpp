//
// Created by Mason on 2024/10/13.
//

#ifndef BATMANINFER_STATUS_CODE_HPP
#define BATMANINFER_STATUS_CODE_HPP

namespace BatmanInfer {
    enum class RuntimeParameterType {
        bParameterUnknown = 0,
        bParameterBool = 1,
        bParameterInt = 2,

        bParameterFloat = 3,
        bParameterString = 4,
        bParameterIntArray = 5,
        bParameterFloatArray = 6,
        bParameterStringArray = 7,
    };

    enum class InferStatus {
        bInferUnknown = -1,
        bInferSuccess = 0,

        bInferFailedInputEmpty = 1,
        bInferFailedWeightParameterError = 2,
        bInferFailedBiasParameterError = 3,
        bInferFailedStrideParameterError = 4,
        bInferFailedDimensionParameterError = 5,
        bInferFailedInputOutSzeMatchError = 6,

        bInferFailedOutputSizeError = 7,
        bInferFailedShapeParameterError = 9,
        bInferFailedChannelParameterError = 10,
        bInferFailedOutputEmpty = 11,
    };

    enum class ParseParameterAttrStatus {
        bParameterMissingUnknown = -1,
        bParameterMissingStride = 1,
        bParameterMissingPadding = 2,
        bParameterMissingKernel = 3,
        bParameterMissingUseBias = 4,
        bParameterMissingInChannel = 5,
        bParameterMissingOutChannel = 6,

        bParameterMissingEps = 7,
        bParameterMissingNumFeatures = 8,
        bParameterMissingDim = 9,
        bParameterMissingExpr = 10,
        bParameterMissingOutHW = 11,
        bParameterMissingShape = 12,
        bParameterMissingGroups = 13,
        bParameterMissingScale = 14,
        bParameterMissingResizeMode = 15,
        bParameterMissingDilation = 16,
        bParameterMissingPaddingMode = 16,

        bAttrMissingBias = 21,
        bAttrMissingWeight = 22,
        bAttrMissingRunningMean = 23,
        bAttrMissingRunningVar = 24,
        bAttrMissingOutFeatures = 25,
        bAttrMissingYoloStrides = 26,
        bAttrMissingYoloAnchorGrides = 27,
        bAttrMissingYoloGrides = 28,

        bParameterAttrParseSuccess = 0
    };
}

#endif //BATMANINFER_STATUS_CODE_HPP
