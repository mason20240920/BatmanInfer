//
// Created by Mason on 2024/10/11.
//

#include <data/tensor.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_tensor_values, tensor_values1) {
    using namespace BatmanInfer;
    Tensor<float> f1(2, 3, 4);
    f1.Rand();
    f1.Show();

    // 返回第一个通道(channel)中的数据
    LOG(INFO) << "Data in the first channel: " << f1.slice(0);
    // 返回(1, 1, 1)位置的数据
    LOG(INFO) << "Data in the (1, 1, 1): " << f1.at(1, 1, 1);
}

TEST(test_fill_reshape, fill1) {
    // 行主序的填充方式
    using namespace BatmanInfer;
    Tensor<float> f1(2, 3, 4);
    std::vector<float> values(2 * 3 * 4);
    // 将1到24填充到values中
    for (int i = 0; i < 24; ++i)
        values.at(i) = float(i + 1);
    f1.Fill(values);
    f1.Show();
}

float MinusOne(float value) {
    return value - 1.f;
}

float DoubleRet(float value) {
    return value * 2;
}

TEST(test_tranform, transform1) {
    using namespace BatmanInfer;
    Tensor<float> f1(2, 3, 4);
    f1.Rand();
    f1.Show();
    f1.Transform(MinusOne);
    f1.Show();
}

TEST(test_transform, transform2) {
    using namespace BatmanInfer;
    Tensor<float> f2(2, 3, 4);
    f2.Ones();
    f2.Show();
    f2.Transform(DoubleRet);
    f2.Show();
}

TEST(test_fill_shape, reshape1) {
    using namespace BatmanInfer;
    // Channel, Rows, Cols
    Tensor<float> f1(2, 3, 4);
    std::vector<float> values(2 * 3 * 4);
    // 将1到12填充到values中
    for (int i = 0; i < 24; ++i)
        values.at(i) = float(i + 1);
    f1.Fill(values);
    f1.Show();
    f1.Reshape({4, 3, 2});
    f1.Show();
}

TEST(test_homework, homework1_flatten1) {
    using namespace BatmanInfer;
    Tensor<float> f1(2, 3, 4);
    LOG(INFO) << "-------------------before Flatten-------------------";
    f1.Show();
    f1.Flatten(true);
    LOG(INFO) << "-------------------after Flatten-------------------";
    f1.Show();
    ASSERT_EQ(f1.raw_shapes().size(), 1);
    ASSERT_EQ(f1.raw_shapes().at(0), 24);
}

TEST(test_homework, homework1_flatten2) {
    using namespace BatmanInfer;
    Tensor<float> f1(12, 24);
    LOG(INFO) << "-------------------before Flatten-------------------";
    f1.Show();
    f1.Flatten(true);
    LOG(INFO) << "-------------------after Flatten-------------------";
    f1.Show();
    ASSERT_EQ(f1.raw_shapes().size(), 1);
    ASSERT_EQ(f1.raw_shapes().at(0), 24 * 12);
}

TEST(test_homework, homework2_padding1) {
    using namespace BatmanInfer;
    Tensor<float> tensor(3, 4, 5);
    // channels, rows, cols
    ASSERT_EQ(tensor.channels(), 3);
    ASSERT_EQ(tensor.rows(), 4);
    ASSERT_EQ(tensor.cols(), 5);

    tensor.Fill(1.f);
    LOG(INFO) << "-------------------before padding-------------------";
    tensor.Show();
    tensor.Padding({1, 2, 3, 4}, 0);
    LOG(INFO) << "-------------------after padding-------------------";
    tensor.Show();
    ASSERT_EQ(tensor.rows(), 7);
    ASSERT_EQ(tensor.cols(), 12);

}
