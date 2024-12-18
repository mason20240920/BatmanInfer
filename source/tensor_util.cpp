//
// Created by Mason on 2024/10/18.
//

#include <glog/logging.h>
#include <data/Tensor.hpp>
#include <data/tensor_util.hpp>
#include <omp.h>
#include <cblas.h>
#include <Halide.h>

namespace BatmanInfer {
    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t batch_size,
                                                uint32_t channels,
                                                uint32_t rows,
                                                uint32_t cols) {
        return std::make_shared<Tensor<float>>(batch_size, channels, rows, cols);
    }

    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels,
                                                uint32_t rows,
                                                uint32_t cols) {
        return std::make_shared<Tensor<float>>(0, channels, rows, cols);
    }

    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t rows,
                                                uint32_t cols) {
        return std::make_shared<Tensor<float>>(0, 0, rows, cols);
    }

    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t size) {
        return std::make_shared<Tensor<float>>(0, 0, 0, size);
    }

    std::shared_ptr<Tensor<float>> TensorClone(const std::shared_ptr<Tensor<float>> &tensor) {
        return std::make_shared<Tensor<float>>(*tensor);
    }

    std::shared_ptr<Tensor<float>> MatrixMultiply(const std::shared_ptr<Tensor<float>> &tensor1,
                                                  const std::shared_ptr<Tensor<float>> &tensor2) {
        CHECK(!tensor1->empty() && !tensor2->empty());

        // Ensure the number of columns in the first matrix equals the number of rows in the second matrix
        CHECK_EQ(tensor1->cols(), tensor2->rows()) << "Incompatible dimensions for matrix multiplication";

        // Perform matrix multiplication for each channel
        uint32_t channels = tensor1->channels();
        CHECK_EQ(channels, tensor2->channels()) << "Channel mismatch between tensors";

        // 计算结果张量的维度
        std::vector<uint32_t> result_shapes = tensor1->shapes();
        result_shapes[result_shapes.size() - 2] = tensor1->rows();  // 更新行数
        result_shapes[result_shapes.size() - 1] = tensor2->cols();  // 更新列数

        // 使用 OpenMP 并行化通道的乘法
        auto result = std::make_shared<Tensor<float>>(result_shapes);

        // 使用 OpenMP 并行化通道的乘法
        for (uint32_t c = 0; c < channels; ++c) {
            // 获取每个通道的矩阵数据指针
            const float *data1 = tensor1->raw_ptr() + c * tensor1->rows() * tensor1->cols();
            const float *data2 = tensor2->raw_ptr() + c * tensor2->rows() * tensor2->cols();
            float *result_data = result->raw_ptr() +
                                 c * result_shapes[result_shapes.size() - 2] * result_shapes[result_shapes.size() - 1];

            // 使用 OpenBLAS 的 cblas_sgemm 进行矩阵乘法
            // C = alpha * A * B + beta * C
            // A: tensor1 矩阵 (MxK), B: tensor2 矩阵 (KxN), C: result 矩阵 (MxN)
            // alpha = 1.0, beta = 0.0 表示不考虑原始 C 的值
            cblas_sgemm(CblasRowMajor,          // 数据存储格式为行优先
                        CblasNoTrans,           // A 不进行转置
                        CblasNoTrans,           // B 不进行转置
                        static_cast<int>(tensor1->rows()),        // M: A 的行数
                        static_cast<int>(tensor2->cols()),        // N: B 的列数
                        static_cast<int>(tensor1->cols()),        // K: A 的列数（同时也是 B 的行数）
                        1.0f,                   // alpha: 缩放因子
                        data1,                  // A 的数据指针
                        static_cast<int>(tensor1->cols()),        // lda: A 的列数（主维度跨度）
                        data2,                  // B 的数据指针
                        static_cast<int>(tensor2->cols()),        // ldb: B 的列数（主维度跨度）
                        0.0f,                   // beta: 缩放因子
                        result_data,            // C 的数据指针
                        static_cast<int>(tensor2->cols()));       // ldc: C 的列数（主维度跨度）
        }
        return result;
    }

    std::shared_ptr<Tensor<float>> MultiplyElement(const std::shared_ptr<Tensor<float>> &tensor1,
                                                   const std::shared_ptr<Tensor<float>> &tensor2) {
        CHECK(!tensor1->empty() && !tensor2->empty());

        CHECK_EQ(tensor1->cols(), tensor2->cols()) << "Incompatible dimensions for elements multiplication";
        CHECK_EQ(tensor1->rows(), tensor2->rows()) << "Incompatible dimensions for elements multiplication";
        CHECK_EQ(tensor1->channels(), tensor2->channels()) << "Incompatible dimensions for elements multiplication";

        sftensor result = TensorClone(tensor1);

        result->Mul(tensor2);

        return result;
    }

    std::shared_ptr<Tensor<float>> Concat(const std::vector<std::shared_ptr<Tensor<float>>> &tensors,
                                          int axis) {
        // 验证输入的张量数组是否是空的
        CHECK(!tensors.empty());
        // 第一个张量的结构
        const auto &first_shape = tensors[0]->shapes();
        // 轴所在的范围(>= 0 或者 < 张量的shapes), 一般都是3维，这块可以改为 axis < 3
        CHECK(axis >= 0 && axis < first_shape.size());

        // 计算输出张量形状, 目前是3维不变
        std::vector<uint32_t> output_shape = first_shape;
        uint32_t concat_dim_size = 0;

        // 验证输入张量的形状，并计算合并维度的总大小
        for (const auto &tensor: tensors) {
            const auto &shape = tensor->shapes();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i != static_cast<size_t>(axis)) {
                    CHECK(shape[i] == first_shape[i]) << "Shape mismatch on non-concat axis.";
                }
            }
            concat_dim_size += shape[axis];
        }
        output_shape[axis] = concat_dim_size;

        // 创建输出张量
        auto output_tensor = std::make_shared<Tensor<float>>(output_shape);

        // 获取合并维度的跨度
        uint32_t offset = 0;
        for (const auto &tensor: tensors) {
            const auto &shape = tensor->shapes();
            uint32_t current_size = shape[axis];

            // 获取源张量和目标张量的底层数据指针
            const float *src_data = tensor->raw_ptr();
            float *dst_data = output_tensor->raw_ptr();

            // 计算张量在目标张量中的偏移量
            uint32_t outer_size = 1; // 合并轴之前的元素数量
            for (int i = 0; i < axis; ++i) {
                outer_size *= shape[i];
            }
            uint32_t inner_size = 1; // 合并轴之后的元素数量
            for (int i = axis + 1; i < shape.size(); ++i) {
                inner_size *= shape[i];
            }

            // 使用 OpenMP 并行化
#pragma omp parallel for collapse(2)
            for (uint32_t outer = 0; outer < outer_size; ++outer) {
                for (uint32_t inner = 0; inner < inner_size; ++inner) {
                    // 计算源和目标的内存偏移量
                    uint32_t src_offset = outer * current_size * inner_size + inner;
                    uint32_t dst_offset = outer * output_shape[axis] * inner_size + offset * inner_size + inner;

                    // 批量复制数据
                    memcpy(dst_data + dst_offset, src_data + src_offset, current_size * sizeof(float));
                }
            }

            // 更新偏移量
            offset += current_size;
        }

        return output_tensor;
    }


    void merge_tensors(const std::vector<sftensor> &tensors,
                       std::vector<sftensor> &merge_tensor) {
        merge_tensor.clear();
        merge_tensor.reserve(tensors.size());
        merge_tensor.insert(merge_tensor.end(), tensors.begin(), tensors.end());
    }

    std::shared_ptr<Tensor<float>> Trilu(const std::shared_ptr<Tensor<float>> &tensor,
                                         int upper) {
        CHECK(tensor != nullptr) << "Input tensor is null!";
        CHECK(!tensor->empty()) << "Tensor is empty!";
        CHECK(upper == 0 || upper == 1) << "Parameter 'upper' must be 0 (lower triangle) or 1 (upper triangle)!";

        const auto &shape = tensor->shapes();
        uint32_t rows = shape[shape.size() - 2];
        uint32_t cols = shape[shape.size() - 1];
        uint32_t channels = tensor->channels();

        auto result = std::make_shared<Tensor<float>>(shape);

        // 定义 Halide 的变量
        Halide::Var x("x"), y("y"), c("c");

        // 定义输入 Halide 函数
        Halide::Buffer<float> input(tensor->raw_ptr(), cols, rows, channels);
        Halide::Func trilu("trilu");

        // 定义三角矩阵提取逻辑
        trilu(x, y, c) = select(
                (upper == 1 && x >= y) || (upper == 0 && x <= y),
                input(x, y, c),
                0.0f
        );

        // 调度优化
        trilu.parallel(c).vectorize(x, 8);

        // 输出结果到 Buffer
        Halide::Buffer<float> output(result->raw_ptr(), cols, rows, channels);
        trilu.realize(output);

        return result;
    }

    /**
     * @brief 把int数组转化为float类型
     * @param runtime_parameter_int
     * @return
     */
    std::vector<float> convert_to_int_vector(const RuntimeParameterIntArray *runtime_parameter_int) {
        std::vector<float> float_vec;
        float_vec.reserve(runtime_parameter_int->value.size());
        std::transform(runtime_parameter_int->value.begin(),
                       runtime_parameter_int->value.end(),
                       std::back_inserter(float_vec),
                       [](int val) { return static_cast<float>(val); });
        return float_vec;
    }

    void Gemm(const sftensor &tensor1,
              const sftensor &tensor2,
              sftensor &result,
              float bias) {
        CHECK(tensor1->dimensions() == 2 && tensor2->dimensions()  == 2 && result->dimensions() == 2) << "Dimensions not right in gemm";
        // 获取矩阵的维度信息
        int tensor_1_rows = tensor1->rows();
        int tensor_1_cols = tensor1->cols();
        int tensor_2_cols = tensor2->cols();


        // 检查维度是否匹配
        if (tensor2->rows() != tensor_1_cols || result->cols() != tensor_1_rows || result->rows() != tensor_2_cols) {
            LOG(ERROR) << "Error: Matrix dimensions do not match for multiplication.";
            return;
        }

        // 获取矩阵的步长（Halide 的 stride 表示每一维的内存跨度）
        int lda = tensor1->cols();  // 矩阵 A 的列主存储间距
        int ldb = tensor2->cols();  // 矩阵 B 的列主存储间距
        int ldc = result->cols();  // 矩阵 C 的列主存储间距

        // 获取底层数据指针
        const float *tensor1_data = reinterpret_cast<const float *>(tensor1->matrix_host());
        const float *tensor2_data = reinterpret_cast<const float *>(tensor2->matrix_host());
        float *result_data = reinterpret_cast<float *>(result->matrix_host());

        // 调用 OpenBLAS 的 cblas_sgemm
        cblas_sgemm(CblasRowMajor,      // 数据存储方式：行主序
                    CblasNoTrans,       // 矩阵 A 不转置
                    CblasNoTrans,       // 矩阵 B 不转置
                    tensor_1_rows,                  // 矩阵 A 的行数
                    tensor_2_cols,                  // 矩阵 B 的列数
                    tensor_1_cols,                  // 矩阵 A 的列数 / 矩阵 B 的行数
                    1.0f,              // 缩放因子 alpha
                    tensor1_data, lda,        // 矩阵 A 和其列主存储间距
                    tensor2_data, ldb,        // 矩阵 B 和其列主存储间距
                    bias,               // 缩放因子 beta
                    result_data, ldc);       // 矩阵 C 和其列主存储间距


//        if (bias != 0.0f) {
//            Halide::Buffer<float> input(result->data());
//
//            // Step 1: 定义 Halide 变量
//            Halide::Var x, y;
//
//            // Step 2: 定义 Halide Func
//            Halide::Func add_scalar;
//            add_scalar(x, y) = input(x, y) + bias;
//
//            // Step 3: 调度优化（可选）
//            add_scalar.parallel(y).vectorize(x, 8, Halide::TailStrategy::GuardWithIf);
//
//            // Step 4: 实现计算并写入输出 buffer
//            add_scalar.realize(input);
//        }
    }
}