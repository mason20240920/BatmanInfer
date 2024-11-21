//
// Created by Mason on 2024/10/18.
//

#include <glog/logging.h>
#include <data/Tensor.hpp>
#include <data/tensor_util.hpp>
#include <omp.h>

namespace BatmanInfer {
    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels,
                                                uint32_t rows,
                                                uint32_t cols) {
        return std::make_shared<Tensor<float>>(channels, rows, cols);
    }

    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t rows,
                                                uint32_t cols) {
        return std::make_shared<Tensor<float>>(1, rows, cols);
    }

    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t size) {
        return std::make_shared<Tensor<float>>(1, 1, size);
    }

    std::shared_ptr<Tensor<float>> TensorClone(std::shared_ptr<Tensor<float>> tensor) {
        return std::make_shared<Tensor<float>>(*tensor);
    }

    void TensorElementMultiply(const std::shared_ptr<Tensor<float>>& tensor1,
                               const std::shared_ptr<Tensor<float>>& tensor2,
                               const std::shared_ptr<Tensor<float>>& output_tensor) {
        CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
        if (tensor1->shapes() == tensor2->shapes()) {
            CHECK(tensor1->shapes() == output_tensor->shapes());
            output_tensor->set_data(tensor1->data() % tensor2->data());
        } else {
            CHECK(tensor1->channels() == tensor2->channels())
                            << "Tensors shape are not adapting";
            const auto& [input_tensor1, input_tensor2] =
                    TensorBroadcast(tensor1, tensor2);
            CHECK(output_tensor->shapes() == input_tensor1->shapes() &&
                  output_tensor->shapes() == input_tensor2->shapes());
            output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
        }
    }

    std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor& tensor1,
                                                   const sftensor& tensor2) {
        CHECK(tensor1 != nullptr && tensor2 != nullptr);
        if (tensor1->shapes() == tensor2->shapes()) {
            return {tensor1, tensor2};
        } else {
            CHECK(tensor1->channels() == tensor2->channels());
            if (tensor2->rows() == 1 && tensor2->cols() == 1) {
                sftensor new_tensor =
                        TensorCreate(tensor2->channels(), tensor1->rows(), tensor1->cols());
                CHECK(tensor2->size() == tensor2->channels());
                for (uint32_t c = 0; c < tensor2->channels(); ++c) {
                    new_tensor->slice(c).fill(tensor2->index(c));
                }
                return {tensor1, new_tensor};
            } else if (tensor1->rows() == 1 && tensor1->cols() == 1) {
                sftensor new_tensor =
                        TensorCreate(tensor1->channels(), tensor2->rows(), tensor2->cols());
                CHECK(tensor1->size() == tensor1->channels());
                for (uint32_t c = 0; c < tensor1->channels(); ++c) {
                    new_tensor->slice(c).fill(tensor1->index(c));
                }
                return {new_tensor, tensor2};
            } else {
                LOG(FATAL) << "Broadcast shape is not adapting!";
                return {tensor1, tensor2};
            }
        }
    }

    std::shared_ptr<Tensor<float>> MatrixMultiply(const std::shared_ptr<Tensor<float>> &tensor1,
                                                  const std::shared_ptr<Tensor<float>> &tensor2) {
        CHECK(!tensor1->empty() && !tensor2->empty());

        // Ensure the number of columns in the first matrix equals the number of rows in the second matrix
        CHECK_EQ(tensor1->cols(), tensor2->rows()) << "Incompatible dimensions for matrix multiplication";

        // Perform matrix multiplication for each channel
        uint32_t channels = tensor1->channels();
        CHECK_EQ(channels, tensor2->channels()) << "Channel mismatch between tensors";

        // Prepare the result tensor
        sftensor result = TensorCreate(channels, tensor1->rows(), tensor2->cols());

        // 使用 OpenMP 并行化通道的乘法
#pragma omp parallel for
        for (uint32_t c = 0; c < channels; ++c) {
            // Multiply the slices (2D matrices) of each channel
            result->slice(c) = tensor1->slice(c) * tensor2->slice(c);
        }

        return result;
    }

    std::shared_ptr<Tensor<float>> Concat(const std::vector<std::shared_ptr<Tensor<float>>>& tensors,
                                          int axis) {
        CHECK(!tensors.empty());

        const auto& first_shape = tensors[0]->shapes();

        CHECK(axis >= 0 && axis < first_shape.size());

        // 计算输出张量形状
        std::vector<uint32_t> output_shape = first_shape;
        uint32_t concat_dim_size = 0;

        for (const auto& tensor : tensors) {
            const auto& shape = tensor->shapes();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i != static_cast<size_t>(axis)) {
                    CHECK(shape[i] == first_shape[i]);
                }
            }
            concat_dim_size += shape[axis];
        }
        output_shape[axis] = concat_dim_size;

        // 创建输出张量
        auto output_tensor = std::make_shared<Tensor<T>>(output_shape);

        // 拼接张量
        uint32_t offset = 0;
        for (const auto& tensor : tensors) {
            const auto& shape = tensor->shapes();
            uint32_t current_size = shape[axis];

            // 使用 OpenMP 并行化拼接过程
#pragma omp parallel for
            for (uint32_t c = 0; c < tensor->channels(); ++c) {
                for (uint32_t r = 0; r < tensor->rows(); ++r) {
                    for (uint32_t col = 0; col < tensor->cols(); ++col) {
                        std::vector<uint32_t> src_idx = {c, r, col};
                        std::vector<uint32_t> dst_idx = src_idx;
                        dst_idx[axis] += offset;

                        output_tensor->at(dst_idx) = tensor->at(src_idx);
                    }
                }
            }
            offset += current_size;
        }

        return output_tensor;
    }
}