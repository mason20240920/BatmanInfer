//
// Created by Mason on 2025/4/10.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/neon/bi_ne_functions.h>
#include <runtime/bi_tensor.hpp>

TEST(MemAlloc, TensorAlloc) {
    using namespace BatmanInfer;
    BITensor tensor;
    tensor.allocator()->init(BITensorInfo(BITensorShape(2, 2), 1, BIDataType::F16));
    tensor.allocator()->allocate();
}
