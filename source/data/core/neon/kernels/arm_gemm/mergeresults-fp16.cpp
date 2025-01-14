//
// Created by Mason on 2025/1/14.
//
#include <algorithm>

#include <neon/neon_defines.h>

#include <cpu/kernels/assembly/bi_arm_gemm.hpp>
#include <data/core/neon/kernels/arm_gemm/asmlib.hpp>
#include <data/core/neon/kernels/arm_gemm/utils.hpp>

#include <data/core/neon/kernels/arm_gemm/mergeresults.hpp>

namespace BatmanGemm {

#include <data/core/neon/kernels/arm_gemm/merges/list-fp16.hpp>

}

