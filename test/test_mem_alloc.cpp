//
// Created by Mason on 2025/4/10.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/neon/bi_ne_functions.h>
#include <runtime/bi_tensor.hpp>

#include "utils/utils.hpp"

namespace MemAllocTest {
    void print_tensor(const BatmanInfer::BITensor &tensor,
                      const std::string &name = "temp",
                      const BatmanInfer::BIIOFormatInfo::PrintRegion region =
                              BatmanInfer::BIIOFormatInfo::PrintRegion::Full) {
        std::cout << name << std::endl;
        BatmanInfer::BIIOFormatInfo format;
        format.element_delim = ", "; // 元素之间用逗号分隔
        format.row_delim = "\n"; // 每行换行
        format.align_columns = 1; // 对齐列
        format.print_region = region;

        tensor.print(std::cout, format);
    }

    template<typename T>
    void fill_tensor_val_with_arr(const BatmanInfer::BITensor &tensor, const std::vector<T> val) {
        auto tensor_ptr = reinterpret_cast<T *>(tensor.buffer());
        const size_t num_elements = tensor.info()->tensor_shape().total_size(); // 获取元素数量
        for (size_t i = 0; i < num_elements; i++) {
            tensor_ptr[i] = val[i];
        }
    }
}

TEST(MemAlloc, TensorAlloc) {
    using namespace BatmanInfer;
    BITensor tensor;
    tensor.allocator()->init(BITensorInfo(BITensorShape(16, 16), 1, BIDataType::F16));
    tensor.allocator()->allocate();

    BITensorShape sub_shape(4, 4); // 要提取 64x64 的子区域
    BITensorInfo sub_info(sub_shape, 1, BIDataType::F16);
    sub_info.set_format(Format::F16);

    auto input_ptr = reinterpret_cast<float16_t *>(tensor.buffer());
    for (int i = 0; i < 256; i++) {
        input_ptr[i] = static_cast<float16_t>(i);
    }

    // 4. 创建子张量
    BITensor sub_tensor;
    sub_tensor.allocator()->init(*tensor.allocator(), sub_info);
    BIIOFormatInfo format;
    format.element_delim = ", "; // 元素之间用逗号分隔
    format.row_delim = "\n"; // 每行换行
    format.align_columns = 1; // 对齐列

    sub_tensor.print(std::cout, format);
}

TEST(MemAllocGPT2, GPTAllocDynamic) {
    using namespace BatmanInfer;
    int batch_size = 1;
    int seq_len = 1;
    // 1. 初始化一个最大input算子
    BITensor original_input_tensor;
    BITensorShape original_input_tensor_shape(16, 20);
    original_input_tensor.allocator()->init(BITensorInfo(original_input_tensor_shape, 1, BIDataType::U32));
    original_input_tensor.allocator()->allocate();

    // 1.1 初始化一个小型算子
    BITensor input_tensor;
    BITensorShape input_tensor_shape(seq_len, batch_size);
    BITensorInfo input_info(input_tensor_shape, 1, BIDataType::U32);
    input_info.set_format(Format::U32);
    input_tensor.allocator()->init(*original_input_tensor.allocator(), input_info);
    std::vector<uint32_t> indices_data{0};
    MemAllocTest::fill_tensor_val_with_arr(input_tensor, indices_data);


    // 2. Gather的权重
    BITensorShape gather_weight_shape(768, 6003);
    const std::string &weight_path =
            "./input_res/transformer_wte_weight.npy";
    BITensor weight = utils::create_type_tensor(
        weight_path, gather_weight_shape,
        BIDataType::F16);
    // MemAllocTest::print_tensor(weight, "weight");

    // 3. 输出原始矩阵
    BITensor original_gather_output_tensor;
    BITensorShape original_gather_output_tensor_shape(768, 16, 20);
    original_gather_output_tensor.allocator()->init(
        BITensorInfo(original_gather_output_tensor_shape, 1, BIDataType::F16));
    original_gather_output_tensor.allocator()->allocate();

    // 3.1 输出矩阵的子矩阵
    BITensor gather_output_tensor;
    BITensorShape gather_output_tensor_shape(768, seq_len, batch_size);
    BITensorInfo gather_output_info(gather_output_tensor_shape, 1, BIDataType::F16);
    gather_output_info.set_format(Format::F16);
    gather_output_tensor.allocator()->init(*original_gather_output_tensor.allocator(), gather_output_info);


    // 2. 进行NEGather筛选
    BINEGather gather_layer;
    gather_layer.configure(&weight, &input_tensor, &gather_output_tensor, 1);
    gather_layer.run();

    // 3. Add权重的获取
    BITensorShape add_wte_weight_shape(768, 16);
    const std::string &add_wte_weight_path = "./input_res/add_wte_weights.npy";
    BITensor add_wte_weight = utils::create_type_tensor(
        add_wte_weight_path, add_wte_weight_shape,
        BIDataType::F16);

    // 临时的数据
    BITensor sub_add_weight;
    BITensorShape sub_add_weight_shape(768, seq_len);
    BITensorInfo sub_add_weight_info(sub_add_weight_shape, 1, BIDataType::F16);
    sub_add_weight_info.set_format(Format::F16);
    sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);
    MemAllocTest::print_tensor(sub_add_weight, "add");

    // 4. Add输出的原始最大值
    BITensor original_add_output_tensor;
    original_add_output_tensor.allocator()->init(
        BITensorInfo(original_gather_output_tensor_shape, 1, BIDataType::F16));
    original_add_output_tensor.allocator()->allocate();

    // 4.1 Add输出的新的数据格式
    BITensor add_output_tensor;
    add_output_tensor.allocator()->init(*original_add_output_tensor.allocator(), gather_output_info);

    BINEArithmeticAddition add_layer;
    add_layer.configure(&gather_output_tensor, &sub_add_weight, &add_output_tensor, BIConvertPolicy::SATURATE);
    add_layer.run();

    MemAllocTest::print_tensor(add_output_tensor, "add_output");

    // 再次进行运行(动态)
    batch_size = 2;
    seq_len = 4;
    input_tensor_shape = BITensorShape(seq_len, batch_size);
    input_info.set_tensor_shape(input_tensor_shape);
    input_tensor.allocator()->init(*original_input_tensor.allocator(), input_info);
    indices_data = {0, 1, 2, 3, 0, 1, 2, 3};
    MemAllocTest::fill_tensor_val_with_arr(input_tensor, indices_data);

    gather_output_tensor_shape = BITensorShape(768, seq_len, batch_size);
    gather_output_info.set_tensor_shape(gather_output_tensor_shape);
    gather_output_tensor.allocator()->init(*original_gather_output_tensor.allocator(), gather_output_info);

    add_output_tensor.allocator()->init(*original_add_output_tensor.allocator(), gather_output_info);

    sub_add_weight_shape = BITensorShape(768, seq_len);
    sub_add_weight_info.set_tensor_shape(sub_add_weight_shape);
    sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);

    gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
    add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
    gather_layer.run();
    add_layer.run();
    MemAllocTest::print_tensor(add_output_tensor, "add_output");
}
