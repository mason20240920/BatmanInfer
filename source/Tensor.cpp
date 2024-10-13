//
// Created by Mason on 2024/10/11.
//
#include <data/Tensor.hpp>
#include <glog/logging.h>
#include <memory>

namespace BatmanInfer {
    Tensor<float>::Tensor(uint32_t size) {
        // 传入参数依次是, rows cols channels
        data_ = arma::fcube(1, size, 1);
        this->raw_shapes_ = std::vector<uint32_t>{size};
    }

    Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
        // 传入参数 rows, cols, channels
        data_ = arma::fcube(rows, cols, 1);
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
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols, channels};
    }

    Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) {
        auto raw_count = shapes.size();
        if (raw_count == 1)
            data_ = arma::fcube(1, shapes.at(0), 1);
        else if (raw_count == 2)
            data_ = arma::fcube(shapes.at(0), shapes.at(1), 1);
        else
            data_ = arma::fcube(shapes.at(0), shapes.at(1), shapes.at(2));
        this->raw_shapes_ = shapes;
    }

    uint32_t Tensor<float>::rows() const {
        CHECK(!this->data_.empty());
        return this->data_.n_rows;
    }

    uint32_t Tensor<float>::cols() const {
        CHECK(!this->data_.empty());
        return this->data_.n_cols;
    }

    uint32_t Tensor<float>::channels() const {
        CHECK(!this->data_.empty());
        return this->data_.n_slices;
    }

    uint32_t Tensor<float>::size() const {
        CHECK(!this->data_.empty());
        return this->data_.size();
    }

    void Tensor<float>::Ones() {
        CHECK(!this->data_.empty());
        this->data_.fill(1);
    }

    void Tensor<float>::Fill(float value) {
        CHECK(!this->data_.empty());
        this->data_.fill(value);
    }

    void Tensor<float>::Show() {
        for (uint32_t i = 0; i < this->channels(); ++i) {
            LOG(INFO) << "Channels: " << i;
            LOG(INFO) << "\n" << this->data_.slice(i);
        }
    }

    void Tensor<float>::Rand() {
        CHECK(!this->data_.empty());
        this->data_.randn();
    }

    const arma::fmat& Tensor<float>::slice(uint32_t channel) const {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    arma::fmat& Tensor<float>::slice(uint32_t channel) {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
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

    // 接收一个float类型参数，返回一个float类型参数
    void Tensor<float>::Transform(const std::function<float(float)> &filter) {
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
        if (row_major)
            values = this->values(true);
        if (shapes.size() == 3) {
            this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
            this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
        } else if (shapes.size() == 2) {
            // 这是二维张量
            this->data_.reshape(shapes.at(0), shapes.at(1), 1);
            this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
        } else {
            this->data_.reshape(1, shapes.at(0), 1);
            this->raw_shapes_ = {shapes.at(0)};
        }

        if (row_major)
            this->Fill(values, true);
    }

    const float* Tensor<float>::raw_ptr() const {
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
}