//
// Created by Mason on 2024/10/11.
//
#include <data/Tensor.hpp>
#include <glog/logging.h>
#include <memory>
#include <set>
#include <omp.h>

namespace BatmanInfer {
    Tensor<float>::Tensor(uint32_t size) {
        // 传入参数依次是, rows cols channels
        data_ = arma::fcube(1, size, 1);
        this->raw_shapes_ = std::vector<uint32_t>{size};
    }

    Tensor<bool>::Tensor(uint32_t size) {
        // 传入参数一次是, rows, cols channels, 初始化全 0
        data_ = arma::Cube<arma::u8>(1, size, 1, arma::fill::zeros);
        this->raw_shapes_ = std::vector<uint32_t>{size};
    }

    Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
        // 传入参数 rows, cols, channels
        data_ = arma::fcube(rows, cols, 1);
        this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
    }

    Tensor<bool>::Tensor(uint32_t rows, uint32_t cols) {
        data_ = arma::Cube<arma::u8>(rows, cols, 1, arma::fill::zeros);
        this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
    }

    Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
        data_ = arma::fcube(rows, cols, channels);
        if (channels == 1 && rows == 1)
            // 当channel和rows同时等于1, raw_shapes的长度也会是1，表示此时Tensor是一维
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        else if (channels == 1)
            // 当channel等于1时，raw_shapes长度等于2, 表示Tensor是二维
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        else
            // 创建3维张量，则raw_shapes的长度为3，表示此时Tensor是三维的
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
    }

    Tensor<bool>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
        data_ = arma::Cube<arma::u8>(rows, cols, channels, arma::fill::zeros);
        if (channels == 1 && rows == 1)
            // 当channel和rows同时等于1, raw_shapes的长度也会是1，表示此时Tensor是一维
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        else if (channels == 1)
            // 当channel等于1时，raw_shapes长度等于2, 表示Tensor是二维
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        else
            // 创建3维张量，则raw_shapes的长度为3，表示此时Tensor是三维的
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
    }

    Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) {
        CHECK(!shapes.empty() && shapes.size() <= 3);

        uint32_t remaining = 3 - shapes.size();
        std::vector<uint32_t> shapes_(3, 1);
        std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

        uint32_t channels = shapes_.at(0);
        uint32_t rows = shapes_.at(1);
        uint32_t cols = shapes_.at(2);

        data_ = arma::fcube(rows, cols, channels);
        if (channels == 1 && rows == 1) {
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        } else if (channels == 1) {
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        } else {
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
    }

    Tensor<bool>::Tensor(const std::vector<uint32_t>& shapes) {
        CHECK(!shapes.empty() && shapes.size() <= 3);

        uint32_t remaining = 3 - shapes.size();
        std::vector<uint32_t> shapes_(3, 1);
        std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

        uint32_t channels = shapes_.at(0);
        uint32_t rows = shapes_.at(1);
        uint32_t cols = shapes_.at(2);

        data_ = arma::Cube<arma::u8>(rows, cols, channels, arma::fill::zeros);
        if (channels == 1 && rows == 1) {
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        } else if (channels == 1) {
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        } else {
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
    }

    uint32_t Tensor<float>::rows() const {
        CHECK(!this->data_.empty());
        return this->data_.n_rows;
    }

    uint32_t Tensor<bool>::rows() const {
        CHECK(!this->data_.empty());
        return this->data_.n_rows;
    }

    uint32_t Tensor<float>::cols() const {
        CHECK(!this->data_.empty());
        return this->data_.n_cols;
    }

    uint32_t Tensor<bool>::cols() const {
        CHECK(!this->data_.empty());
        return this->data_.n_cols;
    }

    uint32_t Tensor<float>::channels() const {
        CHECK(!this->data_.empty());
        return this->data_.n_slices;
    }

    uint32_t Tensor<bool>::channels() const {
        CHECK(!this->data_.empty());
        return this->data_.n_slices;
    }

    uint32_t Tensor<float>::size() const {
        CHECK(!this->data_.empty());
        return this->data_.size();
    }

    uint32_t Tensor<bool>::size() const {
        CHECK(!this->data_.empty());
        return this->data_.size();
    }

    void Tensor<float>::Ones() {
        CHECK(!this->data_.empty());
        this->data_.fill(1);
    }

    void Tensor<bool>::Ones() {
        CHECK(!this->data_.empty());
        this->data_.fill(1);
    }

    void Tensor<float>::Fill(float value) {
        CHECK(!this->data_.empty());
        this->data_.fill(value);
    }

    void Tensor<bool>::Fill(arma::u8 value) {
        CHECK(!this->data_.empty());
        this->data_.fill(value);
    }



    void Tensor<float>::Show() {
        for (uint32_t i = 0; i < this->channels(); ++i) {
            LOG(INFO) << "Channels: " << i;
            LOG(INFO) << "\n" << this->data_.slice(i);
        }
    }

    void Tensor<bool>::Show() {
        for (uint32_t i = 0; i < this->channels(); ++i) {
            LOG(INFO) << "Channels: " << i;
            LOG(INFO) << "\n" << this->data_.slice(i);
        }
    }

    void Tensor<float>::Rand() {
        CHECK(!this->data_.empty());
        this->data_.randn();
    }

    void Tensor<bool>::Rand() {
        CHECK(!this->data_.empty());
        this->data_.randn();
    }

    const arma::fmat& Tensor<float>::slice(uint32_t channel) const {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    const arma::Mat<arma::u8>& Tensor<bool>::slice(uint32_t channel) const {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    arma::fmat& Tensor<float>::slice(uint32_t channel) {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    arma::Mat<arma::u8>& Tensor<bool>::slice(uint32_t channel) {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_.at(row, col, channel);
    }

    arma::u8 Tensor<bool>::at(uint32_t channel, uint32_t row, uint32_t col) const {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_.at(row, col, channel);
    }

    float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_.at(row, col, channel);
    }

    arma::u8& Tensor<bool>::at(uint32_t channel, uint32_t row, uint32_t col) {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_.at(row, col, channel);
    }

    void Tensor<float>::Fill(const std::vector<float> &values, bool row_major) {
        CHECK(!this->data_.empty());
        const uint32_t total_elems = this->data_.size();
        CHECK_EQ(values.size(), total_elems);
        if (row_major) {
            const uint32_t rows = this->rows();
            const uint32_t cols = this->cols();
            const uint32_t planes = rows * cols;
            const uint32_t channels = this->data_.n_slices;
            for (uint32_t i = 0; i < channels; ++i) {
                // 获取第i个通道的矩阵
                auto& channel_data = this->data_.slice(i);
                // 对矩阵赋值, 一个矩阵的长度
                const arma::fmat& channel_data_t = arma::fmat(values.data() + i * planes, this->cols(), this->rows());
                // 转置，从列添加到行添加
                channel_data = channel_data_t.t();
            }
        } else
            std::copy(values.begin(), values.end(), this->data_.memptr());
    }

    void Tensor<bool>::Fill(const std::vector<arma::u8> &values, bool row_major) {
        CHECK(!this->data_.empty());
        const uint32_t total_elems = this->data_.size();
        CHECK_EQ(values.size(), total_elems);
        if (row_major) {
            const uint32_t rows = this->rows();
            const uint32_t cols = this->cols();
            const uint32_t planes = rows * cols;
            const uint32_t channels = this->data_.n_slices;
            for (uint32_t i = 0; i < channels; ++i) {
                // 获取第i个通道的矩阵
                auto& channel_data = this->data_.slice(i);
                // 对矩阵赋值，一个矩阵的长度
                const auto& channel_data_t = arma::Mat<arma::u8>(values.data() + i * planes, this->cols(), this->rows());
                // 转置
                channel_data = channel_data_t.t();
            }
        } else
            std::copy(values.begin(), values.end(), this->data_.memptr());
    }

    // 接收一个float类型参数，返回一个float类型参数
    void Tensor<float>::Transform(const std::function<float(float)> &filter) {
        CHECK(!this->data_.empty());
        this->data_.transform(filter);
    }

    void Tensor<bool>::Transform(const std::function<arma::u8(arma::u8)> &filter) {
        CHECK(!this->data_.empty());
        this->data_.transform(filter);
    }

    std::vector<float> Tensor<float>::values(bool row_major) {
        CHECK_EQ(this->data_.empty(), false);
        std::vector<float> values(this->data_.size());

        if (!row_major)
            std::copy(this->data_.mem, this->data_.mem + this->data_.size(), values.begin());
        else {
            uint32_t index = 0;
            for (uint32_t c = 0; c < this->data_.n_slices; ++c) {
                const arma::fmat& channel = this->data_.slice(c).t();
                std::copy(channel.begin(), channel.end(), values.begin() + index);
                index += channel.size();
            }
            CHECK_EQ(index, values.size());
        }
        return values;
    }

    void Tensor<float>::Reshape(const std::vector<uint32_t> &shapes, bool row_major) {
        CHECK(!this->data_.empty());
        CHECK(!shapes.empty());

        const uint32_t origin_size = this->size();
        const uint32_t current_size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
        CHECK(shapes.size() <= 3);
        CHECK(current_size == origin_size);

        std::vector<float> values;
        // 行主序
        values = this->values(row_major);

        if (shapes.size() == 3) {
            data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
            raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
        } else if (shapes.size() == 2) {
            // 这是二维张量
            data_.reshape(shapes.at(0), shapes.at(1), 1);
            raw_shapes_ = {shapes.at(0), shapes.at(1)};
        } else {
            data_.reshape(1, shapes.at(0), 1);
            raw_shapes_ = {shapes.at(0)};
        }

        if (row_major) {
            this->Fill(values, true);
        }
    }

    float* Tensor<float>::raw_ptr() {
        CHECK(!this->data_.empty());
        return this->data_.memptr();
    }

    bool Tensor<float>::empty() const {
        return this->data_.empty();
    }

    const std::vector<uint32_t >& Tensor<float>::raw_shapes() const {
        CHECK(!this->raw_shapes_.empty());
        CHECK_LE(this->raw_shapes_.size(), 3);
        CHECK_GE(this->raw_shapes_.size(), 1);
        return this->raw_shapes_;
    }

    void Tensor<float>::Flatten(bool row_major) {
        CHECK(!this->data_.empty());
        if (this->raw_shapes_.size() == 1)
            return;
        // 获取原始的size
        uint32_t vec_size = this->data_.size();
        Reshape({vec_size}, row_major);
    }

    void Tensor<float>::Padding(const std::vector<uint32_t> &pads, float padding_value) {
        CHECK(!this->data_.empty());
        CHECK_EQ(pads.size(), 4);
        // 四周填充的维度
        uint32_t pad_rows1 = pads.at(0); // up
        uint32_t pad_rows2 = pads.at(1); // bottom
        uint32_t pad_cols1 = pads.at(2); // left
        uint32_t pad_cols2 = pads.at(3); // right

        // Original dimensions
        uint32_t original_rows = this->rows();
        uint32_t original_cols = this->cols();
        uint32_t channels = this->channels();

        // New dimensions after padding
        uint32_t new_rows = original_rows + pad_rows1 + pad_rows2;
        uint32_t new_cols = original_cols + pad_cols1 + pad_cols2;

        // Create a new data cube with padded dimensions
        arma::fcube new_data(new_rows,
                             new_cols,
                             channels,
                             arma::fill::value(padding_value));

        // Copy original data into the center of the new data cube
        new_data.subcube(pad_rows1,
                         pad_cols1,
                         0,
                         new_data.n_rows - pad_rows2 - 1,
                         new_data.n_cols - pad_cols2 - 1,
                         new_data.n_slices - 1) = this->data_;

        // Replace the old data with the new padded data
        this->data_ = std::move(new_data);

        // Update the raw shapes to reflect the new dimensions
        this->raw_shapes_ = {channels, new_rows, new_cols};
    }

    std::vector<uint32_t> Tensor<float>::shapes() const {
        CHECK(!this->data_.empty());
        return {this->channels(), this->rows(), this->cols()};
    }

    void Tensor<float>::Transpose() {
        CHECK(!this->data_.empty());

        // 获取当前的形状信息
        uint32_t channels = this->channels();
        uint32_t rows = this->rows();
        uint32_t cols = this->cols();

        // 创建一个新的数据容器，用于存储转置后的数据
        arma::fcube transposed_data(cols, rows, channels);

        // 使用 OpenMP 并行化通道的转置
#pragma omp parallel for
        for (uint32_t i = 0; i < channels; ++i) {
            // 获取当前通道的矩阵
            arma::fmat current_slice = this->slice(i);
            // 对矩阵进行转置
            transposed_data.slice(i) = current_slice.t();
        }

        // 对数据先进行转置
        this->data_.reshape(cols, rows, channels);

        // 用转置后的数据替换原始数据
        this->set_data(transposed_data);

        // 更新 raw_shapes_ 信息
        this->raw_shapes_ = {channels, cols, rows};
    }

    float &Tensor<float>::index(uint32_t offset) {
        CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
        return this->data_.at(offset);
    }

    arma::fcube &Tensor<float>::data() {
        return this->data_;
    }

    const arma::fcube &Tensor<float>::data() const {
        return this->data_;
    }

    std::vector<uint32_t> Tensor<float>::unravel_index(uint32_t flat_index, const std::vector<uint32_t>& shape) const {
        std::vector<uint32_t> indices(shape.size(), 0);
        uint32_t remain = flat_index;
        for (int i = shape.size() - 1; i >= 0; --i) {
            indices[i] = remain % shape[i];
            remain /= shape[i];
        }
        return indices;
    }

    float Tensor<float>::at(const std::vector<uint32_t> &indices) const {
        CHECK(indices.size() == this->raw_shapes_.size());
        uint32_t channel = 0, row = 0, col = 0;

        if (indices.size() == 1) {
            // 一维张量
            col = indices[0];
        } else if (indices.size() == 2) {
            // 二维张量
            row = indices[0];
            col = indices[1];
        } else if (indices.size() == 3) {
            // 三维张量
            channel = indices[0];
            row = indices[1];
            col = indices[2];
        }

        return this->at(channel, row, col);
    }

    float& Tensor<float>::at(const std::vector<uint32_t>& indices) {
        CHECK(indices.size() == this->shapes().size());
        uint32_t channel = 0, row = 0, col = 0;

        if (indices.size() == 1) {
            // 一维张量
            col = indices[0];
        } else if (indices.size() == 2) {
            // 二维张量
            row = indices[0];
            col = indices[1];
        } else if (indices.size() == 3) {
            // 三维张量
            channel = indices[0];
            row = indices[1];
            col = indices[2];
        }

        return this->at(channel, row, col);
    }

    float *Tensor<float>::matrix_raw_ptr(uint32_t index) {
        CHECK_LT(index, this->channels());
        uint32_t offset = index * this->rows() * this -> cols();
        CHECK_LE(offset, this->size());
        float *mem_ptr = this->raw_ptr() + offset;
        return mem_ptr;
    }

    void Tensor<float>::set_data(const arma::fcube &data) {
        CHECK(data.n_rows == this->data_.n_rows)
              << data.n_rows << "!=" << this->data_.n_rows;
        CHECK(data.n_cols == this->data_.n_cols)
              << data.n_cols << "!=" << this->data_.n_cols;
        CHECK(data.n_slices == this->data_.n_slices)
                        << data.n_slices << " != " << this->data_.n_slices;
        this->data_ = data;
    }

    void Tensor<float>::Expand(const std::vector<uint32_t> &shapes) {
        CHECK(!shapes.empty()) << "Target shapes can not be empty";
        CHECK(shapes.size() <= 3) << "Target shapes dimension exceeds 3D!";

        std::vector<uint32_t > current_shapes = this->raw_shapes_;
        uint32_t current_dims = current_shapes.size();
        uint32_t target_dims = shapes.size();

        if (current_dims < target_dims)
            current_shapes.insert(current_shapes.begin(), target_dims - current_dims, 1);

        for (size_t i = 0; i < target_dims; ++i)
            CHECK(current_shapes[i] == shapes[i] || current_shapes[i] == 1)
                 << "Shape mismatch: can not broadcast current shape "
                 << current_shapes[i] << " to target shape " << shapes[i];

        uint32_t target_channels = shapes.size() == 3 ? shapes[0] : 1;
        uint32_t target_rows = shapes.size() >= 2 ? shapes[target_dims - 2] : 1;
        uint32_t target_cols = shapes.back();

        arma::fcube expanded_data(target_rows, target_cols, target_channels);

#pragma omp parallel for
        for (uint32_t c = 0; c < target_channels; ++c) {
            // 如果当前张量的通道数为 1，则重复第一个通道的数据；
            // 否则，直接取当前通道的数据
            uint32_t src_channel = (current_shapes[0] == 1) ? 0 : c;

            // 获取当前通道的二维矩阵数据（`slice`）。
            arma::fmat slice = this->data_.slice(src_channel);

            // 情况 1：如果当前张量的行数和列数都为 1（即标量扩展），
            // 则将当前通道的所有元素填充为原始标量值。
            if (current_shapes[1] == 1 && current_shapes[2] == 1)
                expanded_data.slice(c).fill(slice(0, 0));
            else if (current_shapes[1] == 1)
                // 情况 2：如果当前张量的行数为 1（即一维向量扩展到二维），
                // 则重复第一行的值，扩展为 `target_rows` 行。
                expanded_data.slice(c) = arma::repmat(slice.row(0), target_rows, 1);
            else if (current_shapes[2] == 1)
                // 情况 3：如果当前张量的列数为 1（即一维向量扩展到二维），
                // 则重复第一列的值，扩展为 `target_cols` 列。
                expanded_data.slice(c) = arma::repmat(slice.col(0), 1, target_cols);
            else
                // 情况 4：如果当前张量的行数和列数都与目标形状匹配，
                // 则直接将当前通道的数据复制到目标通道中，无需扩展。
                expanded_data.slice(c) = slice;
        }

        this->data_ = std::move(expanded_data);
        this->raw_shapes_ = shapes;
    }

    void Tensor<float>::Equal(const float &compare_va) {
        // 获取数据指针和总元素数
        // 使用 Armadillo 的 find 函数找到满足条件的索引
        arma::uvec indices_not_equal_1 = arma::find(data_ != compare_va);
        arma::uvec indices_equal_1 = arma::find(data_ == compare_va);

        float* data_ptr = data_.memptr();

#pragma omp parallel for
        for (arma::uword i = 0; i < indices_not_equal_1.n_elem; ++i) {
            data_ptr[indices_not_equal_1[i]] = 1.0f;
        }

        // 并行处理等于 1 的元素
#pragma omp parallel for
        for (arma::uword i = 0; i < indices_equal_1.n_elem; ++i) {
            data_ptr[indices_equal_1[i]] = 0.0f;
        }
    }

    void Tensor<float>::Transpose(const std::vector<uint32_t> &new_order, bool row_major) {
        using index_type = size_t;  // 统一使用 size_t 作为索引类型
        CHECK(!this->data_.empty());
        CHECK(new_order.size() == 3);

        // 原始形状
        const auto temp_shape = shapes();

        // 获取当前数据的值, 预分配并对齐内存
        alignas(32) std::vector<float> values = this->values(row_major);

        // 提前计算stride以避免重复计算
        // stride_1 = rows * cols
        const size_t stride_1 = temp_shape[1] * temp_shape[2];
        // stride_2 = cols;
        const size_t stride_2 = temp_shape[2];

        // 根据新的顺序调整形状
        arma::fcube transposed_data(temp_shape[new_order[1]],
                                    temp_shape[new_order[2]],
                                    temp_shape[new_order[0]]);

        // 填充转置后的数据
        if (row_major) {
#ifdef _OPENMP
            omp_set_num_threads(omp_get_max_threads());
#endif

            // 使用分块策略
            constexpr size_t BLOCK_SIZE = 64;

            // 如果是行主序，需要调整填充顺序
#pragma omp parallel for collapse(3) schedule(guided) proc_bind(close)
            for (size_t i = 0; i < temp_shape[0]; i += BLOCK_SIZE) {
                for (size_t j = 0; j < temp_shape[1]; j += BLOCK_SIZE) {
                    for (size_t k = 0; k < temp_shape[2]; k += BLOCK_SIZE) {
                        // 分块处理
                        for (size_t ii = i; ii < std::min<index_type>(i + BLOCK_SIZE, temp_shape[0]); ++ii) {
                            for (size_t jj = j; jj < std::min<index_type>(j + BLOCK_SIZE, temp_shape[1]); ++jj) {
#pragma omp simd
                                for (size_t kk = k; kk < std::min<index_type>(k + BLOCK_SIZE, temp_shape[2]); ++kk) {
                                    const size_t old_index = ii * stride_1 + jj * stride_2 + kk;

                                    // 如果 new_order 在编译时已知，这里可以用 if constexpr 优化
                                    const size_t new_i = new_order[0] == 0 ? ii : (new_order[0] == 1 ? jj : kk);
                                    const size_t new_j = new_order[1] == 0 ? ii : (new_order[1] == 1 ? jj : kk);
                                    const size_t new_k = new_order[2] == 0 ? ii : (new_order[2] == 1 ? jj : kk);

                                    transposed_data(new_j, new_k, new_i) = values[old_index];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // 列主序的情况
#pragma omp parallel for collapse(3) schedule(static)
            for (size_t i = 0; i < temp_shape[0]; ++i) {
                for (size_t j = 0; j < temp_shape[1]; ++j) {
                    for (size_t k = 0; k < temp_shape[2]; ++k) {
                        transposed_data(j, k, i) = data_(i, j, k);
                    }
                }
            }
        }

        // 使用移动语义避免不必要的拷贝
        data_ = std::move(transposed_data);
        raw_shapes_ = {temp_shape[new_order[0]], temp_shape[new_order[1]], temp_shape[new_order[2]]};
    }


    void Tensor<float>::Where(const float &x,
                              const float &y) {
        // 使用 Armadillo 的 find 函数找到满足条件的索引
        arma::uvec indices_not_equal_1 = arma::find(data_ != 1.0f);
        arma::uvec indices_equal_1 = arma::find(data_ == 1.0f);

        // 获取数据指针
        float* data_ptr = data_.memptr();

        // 并行处理不等于 1 的元素
#pragma omp parallel for
        for (arma::uword i = 0; i < indices_not_equal_1.n_elem; ++i) {
            data_ptr[indices_not_equal_1[i]] = x;
        }

        // 并行处理等于 1 的元素
#pragma omp parallel for
        for (arma::uword i = 0; i < indices_equal_1.n_elem; ++i) {
            data_ptr[indices_equal_1[i]] = y;
        }
    }

    void Tensor<float>::Sqrt() {
        // 获取指向数据的指针
        const size_t n_slices = data_.n_slices;

        // 使用 OpenMP 并行化切片操作
#pragma omp parallel for
        for (size_t i = 0; i < n_slices; ++i) {
            // 对每个切片使用 arma::sqrt 进行逐元素平方根操作
            // Get a non-const reference to the slice
            arma::Mat<float>& slice = data_.slice(i);
            slice = arma::sqrt(slice);
        }
    }


    void Tensor<float>::Div(const float &div_num) {
        if (div_num == 0.0f) {
            throw std::invalid_argument("Division by zero");
        }

        // 获取数据指针和尺寸
        float* data_ptr = data_.memptr();
        const size_t total_elements = data_.n_elem;

        // 使用 OpenMP 并行化
#pragma omp parallel for
        for (size_t i = 0; i < total_elements; ++i) {
            data_ptr[i] /= div_num;
        }
    }

    void Tensor<float>::Mul(const std::shared_ptr<Tensor<float>>& other) {
#pragma omp parallel for collapse(3)
        for (size_t k = 0; k < data_.n_slices; ++k) {
            for (size_t i = 0; i < data_.n_rows; ++i) {
                for (size_t j = 0; j < data_.n_cols; ++j) {
                    data_(i, j, k) = other->data_(i, j, k) * data_(i, j, k);
                }
            }
        }
    }


    std::vector<sftensor> Tensor<float>::Split(const uint32_t &split_axis, const std::vector<uint32_t> &split_lst) {
        CHECK(!this->data_.empty());

        std::vector<sftensor> ret;

        uint32_t total_axis_len = 0;
        // rows: 在行维度进行切分
        if (split_axis == 0) {
            // 获取分割的个数
            auto split_num = split_lst.size();
            ret.reserve(split_num);
            for (int i = 0; i < split_num; ++i) {
                // 提取当前切片的行
                arma::fcube slice_data = data_.rows(i * split_lst.at(i), (i + 1) * split_lst.at(i) - 1);

                auto tensor = std::make_shared<Tensor<float>>(split_lst.at(i), cols(), channels());
                tensor->data_ = slice_data;
                ret.push_back(tensor);
            }
        }
        else if (split_axis == 1) {
            auto split_num = split_lst.size();
            ret.reserve(split_num);

            for (int i = 0; i < split_num; ++i) {
                arma::fcube slice_data = data_.cols(i * split_lst.at(i), (i + 1) * split_lst.at(i) - 1);
                auto tensor = std::make_shared<Tensor<float>>(rows(), split_lst.at(i), channels());
                tensor->data_ = slice_data;
                ret.push_back(tensor);
            }
        } else if (split_axis == 2) {
            total_axis_len = channels();
            auto split_num = split_lst.size();
            ret.reserve(split_num);

            for (int i = 0; i < split_num; ++i) {
                arma::fcube slice_data = data_.slices(i * split_lst.at(i), (i + 1) * split_lst.at(i) - 1);
                auto tensor = std::make_shared<Tensor<float>>(rows(), cols(), split_lst.at(i));
                tensor->data_ = slice_data;
                ret.push_back(tensor);
            }
        }

        return ret;
    }


    // ================= 重构代码，用HalideRuntime重构项目 ===============
    Tensor<float>::Tensor(int32_t batch_size,
                          int32_t channels,
                          int32_t rows,
                          int32_t cols) {
        // 动态存储维度信息
        std::vector<halide_dimension_t> dimensions;
        std::vector<uint32_t> raw_shapes;

        // 根据输入参数动态生成维度信息
        if (batch_size > 0) {
            dimensions.emplace_back(0,
                                    batch_size,
                                    channels * rows * cols,
                                    0);
            raw_shapes.emplace_back(batch_size);
        }
        if (channels > 0) {
            dimensions.emplace_back(0,
                                    channels,
                                    rows * cols,
                                    0);
            raw_shapes.emplace_back(channels);
        }
        if (rows > 0) {
            dimensions.emplace_back(0,
                                    rows,
                                    cols,
                                    0);
            raw_shapes.emplace_back(rows);
        }
        if (cols > 0) {
            dimensions.emplace_back(0,
                                    cols,
                                    1,
                                    0);
            raw_shapes.emplace_back(cols);
        }
        // 设置 h_data_ 的维度信息
        h_data_.dimensions = static_cast<int32_t>(dimensions.size());
        h_data_.dim = new halide_dimension_t[h_data_.dimensions];

        // 将维度信息复制到h_data.dim
        std::copy(dimensions.begin(),
                  dimensions.end(),
                  h_data_.dim);
        std::copy(raw_shapes.begin(),
                  raw_shapes.end(),
                  raw_shapes_.begin());
    }
}