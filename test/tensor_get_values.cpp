//
// Created by Mason on 2024/10/11.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/bi_tensor.hpp>
#include <runtime/bi_blob_lifetime_manager.hpp>
#include <runtime/bi_pool_manager.hpp>
#include <data/core/bi_tensor_info.hpp>

TEST(test_tensor_values, tensor_values1) {
    using namespace BatmanInfer;

    // 创建内存管理器组件
    // 我们需要两个内存管理器:
    // 1. 管理函数内的张量 (mm_layers)
    //    mm_layers：管理函数内部张量
    //* 职责：
    //      负责分配和释放函数内部的临时张量内存。
    //      这些张量的生命周期较短，通常在函数执行完成后即可释放。
    //* 优化目标：
    //      减少函数内部的内存分配和释放开销。
    //      通过复用内存，避免频繁的动态分配操作，提升性能。
    // 2. 管理函数的输入和输出张量 (mm_transitions)
    // * 职责：
    //      负责分配和管理函数之间传递的张量内存。
    //      这些张量需要在多个函数之间共享，因此其生命周期较长。
    // * 优化目标：
    //      确保输入和输出张量的内存可以在不同函数之间高效传递。
    //      避免不必要的内存拷贝和重新分配。

    // 创建生命周期管理器
    auto lifetime_mgr0 = std::make_shared<BIBlobLifetimeManager>();
    auto lifetime_mgr1 = std::make_shared<BIBlobLifetimeManager>();

    // 创建池化管理器
    auto pool_mgr0 = std::make_shared<BIPoolManager>();
    auto pool_mgr1 = std::make_shared<BIPoolManager>();

    // 创建内存管理器
    auto mm_layers = std::make_shared<BIMemoryManagerOnDemand>(lifetime_mgr0, pool_mgr0);
    auto mm_transitions = std::make_shared<BIMemoryManagerOnDemand>(lifetime_mgr1, pool_mgr1);

    // Initialize src tensor
    constexpr unsigned int width_src_image  = 32;
    constexpr unsigned int height_src_image = 32;
    constexpr unsigned int ifm_src_img      = 1;

    const BITensorShape src_shape(width_src_image, height_src_image, ifm_src_img);

    BITensor weights0{};
    BITensor weights1{};

    // 初始化conv0的张量信息
    constexpr unsigned int kernel_x_conv0 = 5;
    constexpr unsigned int kernel_y_conv0 = 5;
    constexpr unsigned int ofm_conv0 = 8;

    const BITensorShape weights_shape_conv0(kernel_x_conv0, kernel_y_conv0, src_shape.z(), ofm_conv0);

    weights0.allocator()->init(BITensorInfo(weights_shape_conv0, 1, BIDataType::F32));
    weights0.print(std::cout);

    std::cout << "Hello" << std::endl;
}

//TEST(test_fill_reshape, fill1) {
//    // 行主序的填充方式
//    using namespace BatmanInfer;
//    Tensor<float> f1(2, 3, 4);
//    std::vector<float> values(2 * 3 * 4);
//    // 将1到24填充到values中
//    for (int i = 0; i < 24; ++i)
//        values.at(i) = float(i + 1);
//    f1.Fill(values);
//    f1.Show();
//}
//
//float MinusOne(float value) {
//    return value - 1.f;
//}
//
//float DoubleRet(float value) {
//    return value * 2;
//}
//
//TEST(test_tranform, transform1) {
//    using namespace BatmanInfer;
//    Tensor<float> f1(2, 3, 4);
//    f1.Rand();
//    f1.Show();
//    f1.Transform(MinusOne);
//    f1.Show();
//}
//
//TEST(test_transform, transform2) {
//    using namespace BatmanInfer;
//    Tensor<float> f2(2, 3, 4);
//    f2.Ones();
//    f2.Show();
//    f2.Transform(DoubleRet);
//    f2.Show();
//}
//
//TEST(test_fill_shape, reshape1) {
//    using namespace BatmanInfer;
//    // Channel, Rows, Cols
//    Tensor<float> f1(2, 3, 4);
//    std::vector<float> values(2 * 3 * 4);
//    // 将1到12填充到values中
//    for (int i = 0; i < 24; ++i)
//        values.at(i) = float(i + 1);
//    f1.Fill(values);
//    f1.Show();
//    f1.Reshape({4, 3, 2});
//    f1.Show();
//}
//
//TEST(test_homework, homework1_flatten1) {
//    using namespace BatmanInfer;
//    Tensor<float> f1(2, 3, 4);
//    LOG(INFO) << "-------------------before Flatten-------------------";
//    f1.Show();
//    f1.Flatten(true);
//    LOG(INFO) << "-------------------after Flatten-------------------";
//    f1.Show();
//    ASSERT_EQ(f1.raw_shapes().size(), 1);
//    ASSERT_EQ(f1.raw_shapes().at(0), 24);
//}
//
//TEST(test_homework, homework1_flatten2) {
//    using namespace BatmanInfer;
//    Tensor<float> f1(12, 24);
//    LOG(INFO) << "-------------------before Flatten-------------------";
//    f1.Show();
//    f1.Flatten(true);
//    LOG(INFO) << "-------------------after Flatten-------------------";
//    f1.Show();
//    ASSERT_EQ(f1.raw_shapes().size(), 1);
//    ASSERT_EQ(f1.raw_shapes().at(0), 24 * 12);
//}
//
//TEST(test_homework, homework2_padding1) {
//    using namespace BatmanInfer;
//    Tensor<float> tensor(3, 4, 5);
//    // channels, rows, cols
//    ASSERT_EQ(tensor.channels(), 3);
//    ASSERT_EQ(tensor.rows(), 4);
//    ASSERT_EQ(tensor.cols(), 5);
//
//    tensor.Fill(1.f);
//    LOG(INFO) << "-------------------before padding-------------------";
//    tensor.Show();
//    tensor.Padding({1, 2, 3, 4}, 0);
//    LOG(INFO) << "-------------------after padding-------------------";
//    tensor.Show();
//    ASSERT_EQ(tensor.rows(), 7);
//    ASSERT_EQ(tensor.cols(), 12);
//
//}
