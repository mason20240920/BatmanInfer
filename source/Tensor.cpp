//
// Created by Mason on 2024/10/11.
//
#include <data/Tensor.hpp>
#include <glog/logging.h>
#include <data/marco.hpp>
#include <memory>
#include <set>
#include <omp.h>
#include <Halide.h>
#include <arm_neon.h>

namespace BatmanInfer {
    Tensor<float>::Tensor(uint32_t size) {
        // 传入参数依次是, rows cols channels
        h_data_.dimensions = 1;
        h_data_.dim = new halide_dimension_t[h_data_.dimensions];
        h_data_.dim[0] = {0,
                          static_cast<int32_t>(size),
                          1,
                          0};
        // 为 host 分配内存
        // 假设每个元素是 float 类型
        h_data_.host = new uint8_t [size * sizeof(float)];
        h_data_.type = halide_type_t(halide_type_float, 32, 1); // 32 位浮点数，标量
        this->raw_shapes_ = std::vector<uint32_t>{size};
    }

    Tensor<bool>::Tensor(uint32_t size) {
        // 传入参数一次是, rows, cols channels, 初始化全 0
        data_ = arma::Cube<arma::u8>(1, size, 1, arma::fill::zeros);
        this->raw_shapes_ = std::vector<uint32_t>{size};
    }

    Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
        // 传入参数 rows, cols, channels
        h_data_ = {0};
        h_data_.dimensions = 2;
        h_data_.dim = new halide_dimension_t[h_data_.dimensions];
        h_data_.dim[0] = {0,
                          static_cast<int32_t>(cols),
                          1,
                          0};
        h_data_.dim[1] = {0,
                          static_cast<int32_t>(rows),
                          static_cast<int32_t>(cols),
                          0};
        // 为 host 分配内存
        h_data_.host = new uint8_t[rows * cols * sizeof(float)]; // 假设每个元素是 float 类型
        h_data_.type = halide_type_t(halide_type_float, 32, 1); // 32 位浮点数，标量
        this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
    }

    Tensor<bool>::Tensor(uint32_t rows, uint32_t cols) {
        data_ = arma::Cube<arma::u8>(rows, cols, 1, arma::fill::zeros);
        this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
    }

    Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
        // 动态存储维度信息
        std::vector<halide_dimension_t> dimensions;
        std::vector<uint32_t> raw_shapes;

        uint32_t vec_length = 1;

        // 根据输入参数动态生成维度信息
        if (channels > 0) {
            dimensions.emplace_back(0,
                                    channels,
                                    rows * cols,
                                    0);
            raw_shapes.emplace_back(channels);
            vec_length *= channels;
        }
        if (rows > 0) {
            dimensions.emplace_back(0,
                                    rows,
                                    cols,
                                    0);
            raw_shapes.emplace_back(rows);
            vec_length *= rows;
        }
        if (cols > 0) {
            dimensions.emplace_back(0,
                                    cols,
                                    1,
                                    0);
            raw_shapes.emplace_back(cols);
            vec_length *= cols;
        }

        // 设置 h_data_ 的维度信息
        h_data_.dimensions = static_cast<int32_t>(dimensions.size());
        h_data_.dim = new halide_dimension_t[h_data_.dimensions];
        h_data_.host = new uint8_t[vec_length * sizeof(float)];
        h_data_.type = halide_type_t(halide_type_float, 32, 1); // 32 位浮点数，标量

        // 进行维度转换
        std::reverse(dimensions.begin(), dimensions.end());
        // 将维度信息复制到h_data.dim
        std::copy(dimensions.begin(),
                  dimensions.end(),
                  h_data_.dim);
        raw_shapes_ = raw_shapes;
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
        CHECK(!shapes.empty() && shapes.size() <= 4);

        uint32_t remaining = 4 - shapes.size();
        std::vector<uint32_t> shapes_(4, 1);
        std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

        uint32_t batch_size = shapes_.at(0);
        uint32_t channels = shapes_.at(1);
        uint32_t rows = shapes_.at(2);
        uint32_t cols = shapes_.at(3);


        // 动态存储维度信息
        std::vector<halide_dimension_t> dimensions;
        std::vector<uint32_t> raw_shapes;
        uint32_t vec_length = 1;

        // 根据输入参数动态生成维度信息
        if (batch_size > 0) {
            dimensions.emplace_back(0,
                                    batch_size,
                                    channels * rows * cols,
                                    0);
            raw_shapes.emplace_back(batch_size);
            vec_length = vec_length * batch_size;
        }
        if (channels > 0) {
            dimensions.emplace_back(0,
                                    channels,
                                    rows * cols,
                                    0);
            raw_shapes.emplace_back(channels);
            vec_length = vec_length * channels;
        }
        if (rows > 0) {
            dimensions.emplace_back(0,
                                    rows,
                                    cols,
                                    0);
            raw_shapes.emplace_back(rows);
            vec_length = vec_length * rows;
        }
        if (cols > 0) {
            dimensions.emplace_back(0,
                                    cols,
                                    1,
                                    0);
            raw_shapes.emplace_back(cols);
            vec_length = vec_length * cols;
        }
        // 设置 h_data_ 的维度信息
        h_data_.dimensions = static_cast<int32_t>(dimensions.size());
        h_data_.dim = new halide_dimension_t[h_data_.dimensions];
        h_data_.host = new uint8_t[vec_length * sizeof(float)];
        h_data_.type = halide_type_t(halide_type_float, 32, 1); // 32 位浮点数，标量

        std::reverse(dimensions.begin(), dimensions.end());
        // 将维度信息复制到h_data.dim
        std::copy(dimensions.begin(),
                  dimensions.end(),
                  h_data_.dim);
        raw_shapes_ = raw_shapes;
    }

    Tensor<float>::Tensor(halide_buffer_t h_data, std::vector<uint32_t> raw_shapes): h_data_(h_data), raw_shapes_(std::move(raw_shapes)) {

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

    uint32_t Tensor<float>::batch_size() const {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        CHECK(h_data_.dimensions == 4); // NOLINT
        return h_data_.dim[3].extent;
    }

    uint32_t Tensor<float>::rows() const {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr);
        CHECK(h_data_.dimensions >= 2);
        return h_data_.dim[1].extent;
    }

    uint32_t Tensor<bool>::rows() const {
        CHECK(!this->data_.empty());
        return this->data_.n_rows;
    }

    uint32_t Tensor<float>::cols() const {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        CHECK(h_data_.dimensions >= 1); // NOLINT
        return h_data_.dim[0].extent;
    }

    uint32_t Tensor<bool>::cols() const {
        CHECK(!this->data_.empty());
        return this->data_.n_cols;
    }

    uint32_t Tensor<float>::channels() const {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        CHECK(h_data_.dimensions >= 3); // NOLINT
        return h_data_.dim[2].extent;
    }

    uint32_t Tensor<bool>::channels() const {
        CHECK(!this->data_.empty());
        return this->data_.n_slices;
    }

    uint32_t Tensor<float>::size() const {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        CHECK(h_data_.dimensions >= 1); // NOLINT

        size_t data_size = h_data_.type.bytes();

        for (int i = 0; i < h_data_.dimensions; ++i) {
            int current_dim_size = h_data_.dim[i].extent;
            data_size *= current_dim_size; // 累乘每一维的 extent
        }
        return data_size / sizeof(float);
    }

    uint32_t Tensor<bool>::size() const {
        CHECK(!this->data_.empty());
        return this->data_.size();
    }

    void Tensor<float>::Ones() {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT

        // 获取维度信息
        int dimensions = h_data_.dimensions;
//        std::vector<int> extents(dimensions);
//        for (int i = 0; i < dimensions; ++i)
//            extents[i] = h_data_.dim[i].extent;

        // 定义 Halide 函数
        Halide::Var x, y, z, w; // 支持最多 4 维
        Halide::Func assign_ones;

        // 调整函数的调度策略
        if (dimensions == 1) {
            assign_ones(x) = Halide::cast<float>(1.0f);
            assign_ones.parallel(x);
        } else if (dimensions == 2) {
            assign_ones(x, y) = Halide::cast<float>(1.0f);
            assign_ones.parallel(y).vectorize(x, 8);   // 使用 SIMD 优化
        }
        else if (dimensions == 3) {
            assign_ones(x, y, z) = Halide::cast<float>(1.0f);
            assign_ones.parallel(z).vectorize(x, 8, Halide::TailStrategy::GuardWithIf);
        }
        else if (dimensions == 4) {
            assign_ones(x, y, z, w) = Halide::cast<float>(1.0f);
//            assign_ones.parallel(w).vectorize(x, 4);
        }

        // 生成 Halide 输出
        Halide::Buffer<float> output(h_data_);
        assign_ones.realize(output);

        // 将结果同步回 halide_buffer_t
        output.copy_to_host();
    }

    void Tensor<bool>::Ones() {
        CHECK(!this->data_.empty());
        this->data_.fill(1);
    }

    void Tensor<float>::Fill(float value) {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT

        // 获取维度信息
        int dimensions = h_data_.dimensions;
        std::vector<int> extents(dimensions);
        for (int i = 0; i < dimensions; ++i)
            extents[i] = h_data_.dim[i].extent;

        // 定义 Halide 函数
        Halide::Var x, y, z, w; // 支持最多 4 维
        Halide::Func assign_ones;
        assign_ones(x, y, z, w) = Halide::cast<float>(value);

        // 调整函数的调度策略
        if (dimensions == 1)
            assign_ones.parallel(x);
        else if (dimensions == 2)
            assign_ones.parallel(y).vectorize(x, 8);   // 使用 SIMD 优化
        else if (dimensions == 3)
            assign_ones.parallel(z).vectorize(x, 8);
        else if (dimensions == 4)
            assign_ones.parallel(w).vectorize(x, 8);

        // 生成 Halide 输出
        Halide::Buffer<float> output(reinterpret_cast<float *>(h_data_.host), extents);
        assign_ones.realize(output);
    }

    void Tensor<bool>::Fill(arma::u8 value) {
        CHECK(!this->data_.empty());
        this->data_.fill(value);
    }



    void Tensor<float>::Show() {
        CHECK(h_data_.host != nullptr && h_data_.dimensions >= 2 && h_data_.dim != nullptr); // NOLINT

        int dims = h_data_.dimensions;

        // Step 2: Get the size of the last two dimensions (rows and cols)
        int rows = h_data_.dim[1].extent; // Second to last dimension
        int cols = h_data_.dim[0].extent; // Last dimension

        // Step 3: Handle higher dimensions (Batch, Channel, etc.)
        int batch_size = raw_shapes_.size() > 3 ? static_cast<int>(raw_shapes_[dims - 4]) : 1;
        int channels = raw_shapes_.size() > 2 ? static_cast<int>(raw_shapes_[dims - 3]) : 1;

        // Step 4: Cast the data pointer to the appropriate type
        // Assuming the buffer contains float data
        const auto *data = (const float *)h_data_.host;

        // Step 5: Calculate strides for accessing data
        int row_stride = h_data_.dim[1].stride;
        int col_stride = h_data_.dim[0].stride;
        int channel_stride = (dims > 2) ? h_data_.dim[2].stride : rows * cols;
        int batch_stride = (dims > 3) ? h_data_.dim[3].stride : channels * rows * cols;

        // Step 6: Iterate over Batch and Channels, then print the matrix
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                std::cout << "[Batch " << b << "] [Channel " << c << "]:\n";
                for (int r = 0; r < rows; r++) {
                    for (int col = 0; col < cols; col++) {
                        // Calculate the linear index in the data array
                        int index = b * batch_stride + c * channel_stride + r * row_stride + col * col_stride;
                        std::cout << data[index] << "\t";
                    }
                    std::cout << "\n"; // End of row
                }
                std::cout << "\n"; // End of channel
            }
        }
    }

    void Tensor<bool>::Show() {
        for (uint32_t i = 0; i < this->channels(); ++i) {
            LOG(INFO) << "Channels: " << i;
            LOG(INFO) << "\n" << this->data_.slice(i);
        }
    }

    void Tensor<float>::Rand() {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT

        // 获取维度信息
        int dimensions = h_data_.dimensions;

        // 定义 Halide 的变量和函数
        std::vector<Halide::Var> vars(dimensions);
        for (int i = 0; i < dimensions; i++) {
            vars[i] = Halide::Var("dim" + std::to_string(i));
        }

        Halide::Func random_fill("random_fill");

        // 定义 Halide 函数
        random_fill(vars) = Halide::random_float() * 2 - 1;

        // 获取rows的数据
        auto col_len = static_cast<int32_t>(raw_shapes_.at(raw_shapes_.size() - 1));

        // 调度：并行化和向量化
        if (dimensions >= 2) {
            CHECK(col_len > 0 && col_len % 8 == 0); // 确保列长度适合向量化
            // 倒数第二维 (行) 并行化，倒数第一维 (列) 向量化
            random_fill.parallel(vars[1]).vectorize(vars[0], col_len);
        } else if (dimensions == 1) {
            random_fill.vectorize(vars[0], col_len);
        }

        CHECK(h_data_.host != nullptr); // 确保 host 指针有效
        for (int i = 0; i < h_data_.dimensions; i++) {
            CHECK(h_data_.dim[i].extent > 0); // 确保每个维度的大小有效
        }

        // 将生成的随机数写入 halide_buffer_t
        Halide::Buffer<float> output(h_data_);


        random_fill.realize(output);


    }

    void Tensor<bool>::Rand() {
        CHECK(!this->data_.empty());
        this->data_.randn();
    }

    const halide_buffer_t* Tensor<float>::slice(uint32_t channel) const {
        CHECK_LT(channel, this->channels());
        const auto channel_dim = h_data_.dimensions - 3;
        // 获取目标维度的信息
        const halide_dimension_t* dims = h_data_.dim;
        uint32_t extent = dims[channel_dim].extent;

        // 动态分配 halide_buffer_t
        auto* sliced_buffer = new halide_buffer_t(h_data_); // 复制原始 buffer

        // 调整目标维度的min 和 extent
        sliced_buffer->dim[channel_dim].min += channel;  // 固定到目标 channel
        sliced_buffer->dim[channel_dim].extent = 1;  // 维度大小设置为1

        // 计算新的host指针
        size_t offset = channel * dims[channel_dim].stride;  // 计算 channel的偏移量
        sliced_buffer->host += offset * h_data_.type.bytes(); // 更新host指针
        return sliced_buffer;
    }

    const arma::Mat<arma::u8>& Tensor<bool>::slice(uint32_t channel) const {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    float* Tensor<float>::slice(uint32_t channel) {
        CHECK(raw_shapes_.size() > 2) << "The dimensions of tensor is less than 3D";
        uint32_t channel_len= h_data_.dim[2].extent;
        CHECK(channel < channel_len) << "Channel index out of range";

        // 计算每个 channel 的大小
        size_t channel_size = rows() * cols();

        // 获取 host 数据指针
        auto* data = reinterpret_cast<float*>(h_data_.host);

        // 计算目标 channel 的 起始地址
        size_t offset = channel * channel_size;
        float* target_channel_ptr = data + offset;

        return target_channel_ptr;
    }

    arma::Mat<arma::u8>& Tensor<bool>::slice(uint32_t channel) {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    float Tensor<float>::at(uint32_t batch_size,
                            uint32_t channel,
                            uint32_t row,
                            uint32_t col) const {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        std::vector<int> indices;

        if (batch_size > 0)
            indices.emplace_back(batch_size);
        if (channel > 0)
            indices.emplace_back(channel);
        if (row > 0)
            indices.emplace_back(row);
        if (col > 0)
            indices.emplace_back(col);

        // 计算偏移量
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            const halide_dimension_t& dim = h_data_.dim[i];
            int index = indices[i];

            // 检查索引是否在范围内
            CHECK(index >= dim.min && index < dim.min + dim.extent) << "Index out of bounds!";

            // 计算当前维度的偏移量
            offset += (index - dim.min) * dim.stride;
        }

        // 访问对应位置的值
        auto* data = reinterpret_cast<float*>(h_data_.host);
        return data[offset];
    }

    arma::u8 Tensor<bool>::at(uint32_t channel, uint32_t row, uint32_t col) const {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_.at(row, col, channel);
    }

    float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
        return at(0, channel, row, col);
    }

    arma::u8& Tensor<bool>::at(uint32_t channel, uint32_t row, uint32_t col) {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_.at(row, col, channel);
    }

    void Tensor<float>::Fill(const std::vector<float> &values, bool row_major) const {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT

        auto total_elements = size();
        CHECK(total_elements == values.size()) << "Size of values does not match buffer dimensions!";

        // 高性能赋值: 直接拷贝
        std::memcpy(h_data_.host, values.data(), values.size() * sizeof(float));
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
    void Tensor<float>::Transform(const std::function<float(float)> &filter) const {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT

        // 获取元素总数
        auto element_count = size();

        auto* data = reinterpret_cast<float *>(h_data_.host);

        int simd_width = 4; // NEON 每次处理 4 个 float
        int i = 0;

        for (; i <= element_count - simd_width; i += simd_width) {
            // 加载 4 个浮点数
            float32x4_t values = vld1q_f32(&data[i]);

            // 示例：假设 filter 是平方操作
            float32x4_t result = vmulq_f32(values, values);

            // 存储结果
            vst1q_f32(&data[i], result);
        }

        // 处理剩余的元素（非 SIMD 部分）
        for (; i < element_count; i++) {
            data[i] = data[i] * data[i];
        }

    }

    void Tensor<bool>::Transform(const std::function<arma::u8(arma::u8)> &filter) {
        CHECK(!this->data_.empty());
        this->data_.transform(filter);
    }

    std::vector<float> Tensor<float>::values(bool row_major) {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT

        // 获取buffer里面所有元素
        int dimensions = h_data_.dimensions;
        auto total_elements = size();

        // 初始化输出数组
        std::vector<float> ret(total_elements);

        // 获取raw buffer data
        const auto* buffer_data = reinterpret_cast<const float*>(h_data_.host);

        // 查看data是否连续存储
        bool is_contiguous = true;
        size_t expected_stride = 1;
        for (int i = 0; i < dimensions; ++i) {
            if (h_data_.dim[i].stride != expected_stride) {
                is_contiguous = false;
                break;
            }
            expected_stride *= h_data_.dim[i].extent;
        }

        if (is_contiguous) {
            // 如果数据是连续的，直接用内存拷贝
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < total_elements; ++i)
                ret[i] = buffer_data[i];
        } else {
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < total_elements; ++i) {
                size_t idx = 0;
                size_t stride = 1;
                for (int d = 0; d < dimensions; ++d) {
                    size_t coord = (i / stride) % h_data_.dim[d].extent;
                    idx += coord * h_data_.dim[d].stride;
                    stride *= h_data_.dim[d].extent;
                }
                ret[i] = buffer_data[idx];
            }
        }
        return ret;
    }

    void Tensor<float>::Reshape(const std::vector<uint32_t> &shapes) {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT

        const auto total_elements = size();

        size_t new_total_elements = 1;
        for (size_t dim: shapes)
            new_total_elements *= dim;

        CHECK(total_elements == new_total_elements) << "The new shapes of Reshape is not match";

        if (shapes.size() > static_cast<size_t>(h_data_.dimensions)) {
            delete[] h_data_.dim; // 释放旧的 dim 数组
            h_data_.dim = new halide_dimension_t[shapes.size()]; // 分配新的 dim 数组
        }

        raw_shapes_ = shapes;

        // 创建新的halide_buffer_t 进行reshape
        int new_dimensions = static_cast<int>(shapes.size());
        h_data_.dimensions = new_dimensions;

        bool is_contiguous = true;
        size_t expected_stride = 1;
        for (int i = h_data_.dimensions - 1; i >= 0; --i) {
            if (h_data_.dim[i].stride != expected_stride) {
                is_contiguous = false;
                break;
            }
            expected_stride *= h_data_.dim[i].extent;
        }

        // 连续内存
        if (is_contiguous) {
            // Update the dim array
            size_t stride = 1;
            for (int i = new_dimensions - 1; i >= 0; --i) {
                h_data_.dim[i].min = 0;                     // Reset min to 0
                h_data_.dim[i].extent = shapes[i];       // Set the new extent
                h_data_.dim[i].stride = stride;            // Update stride
                stride *= shapes[i];                    // Update stride for the next dimension
            }
            return;
        }

        // 非连续内存
        // 分配新的连续内存缓冲区
        size_t element_size = sizeof(float); // 假设数据类型是 float
        auto* new_data = static_cast<uint8_t*>(malloc(total_elements * element_size));
        CHECK(!new_data) << "Failed to allocate memory for reshaped buffer.";

        // 拷贝数据到新的连续内存
        auto* src = reinterpret_cast<float*>(h_data_.host);
        auto* dst = reinterpret_cast<float*>(new_data);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < total_elements; ++i) {
            size_t offset = 0;
            size_t index = i;
            for (int d = h_data_.dimensions - 1; d >= 0; --d) {
                size_t coord = index % h_data_.dim[d].extent;
                offset += coord * h_data_.dim[d].stride;
                index /= h_data_.dim[d].extent;
            }
            dst[i] = src[offset];
        }

        // 更新 halide_buffer_t 的指针和维度信息
        free(h_data_.host); // 释放原始数据
        h_data_.host = new_data;

        h_data_.dimensions = new_dimensions;

        size_t stride = 1;
        for (int i = new_dimensions - 1; i >= 0; --i) {
            h_data_.dim[i].min = 0;                     // Reset min to 0
            h_data_.dim[i].extent = shapes[i];       // Set the new extent
            h_data_.dim[i].stride = stride;            // Update stride
            stride *= shapes[i];                    // Update stride for the next dimension
        }
    }

    float* Tensor<float>::raw_ptr() const {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        return reinterpret_cast<float*>(h_data_.host);
    }

    bool Tensor<float>::empty() const {
        return h_data_.host == nullptr || h_data_.dimensions == 0 || h_data_.dim == nullptr;
    }

    const std::vector<uint32_t >& Tensor<float>::raw_shapes() const {
        CHECK(!this->raw_shapes_.empty());
        CHECK_LE(this->raw_shapes_.size(), 4);
        CHECK_GE(this->raw_shapes_.size(), 1);
        return this->raw_shapes_;
    }

    void Tensor<float>::Flatten(bool row_major) {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        if (this->raw_shapes_.size() == 1)
            return;
        // 获取原始的size
        uint32_t vec_size = size();
        Reshape({vec_size});
    }

    void Tensor<float>::Padding(const std::vector<uint32_t> &pads, float padding_value) {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        CHECK_EQ(pads.size(), 4);

        // 将 halide_buffer_t 包装为 Halide::Buffer
        Halide::Buffer<float> input(h_data_);

        // 四周填充的维度
        uint32_t pad_rows1 = pads.at(0); // up
        uint32_t pad_rows2 = pads.at(1); // bottom
        uint32_t pad_cols1 = pads.at(2); // left
        uint32_t pad_cols2 = pads.at(3); // right

        // 输入的维度信息
        int batch_size = input.dim(0).extent();
        int channels = input.dim(1).extent();
        int rows = input.dim(2).extent();
        int cols = input.dim(3).extent();

        // New dimensions after padding
        uint32_t new_rows = rows + pad_rows1 + pad_rows2;
        uint32_t new_cols = cols + pad_cols1 + pad_cols2;

        // 定义 Halide 的变量
        Halide::Var n("n"), c("c"), x("x"), y("y");

        // 定义 Halide 的函数
        Halide::Func padded("padded");

        Halide::Expr pad_cols1_expr = Halide::Expr(pad_cols1);
        Halide::Expr pad_cols2_expr = Halide::Expr(pad_cols2);
        Halide::Expr pad_rows1_expr = Halide::Expr(pad_rows1);
        Halide::Expr pad_rows2_expr = Halide::Expr(pad_rows2);
        Halide::Expr cols_expr = Halide::Expr(cols);
        Halide::Expr rows_expr = Halide::Expr(rows);

        // 定义填充逻辑
        padded(n, c, x, y) = select(
                x >= pad_cols1_expr && x < (pad_cols1_expr + cols_expr) &&
                y >= pad_rows1_expr && y < (pad_rows1_expr + rows_expr),
                input(n, c, x - pad_cols1_expr, y - pad_rows1_expr), // 原始数据
                padding_value // 填充区域
        );

        // 调度优化
        padded.parallel(n).parallel(c).vectorize(x, 8);

        // 分配输出 Buffer
        Halide::Buffer<float> output(batch_size, channels, new_rows, new_cols);

        // 生成输出
        padded.realize(output);

        // 将结果拷贝回 halide_buffer_t
        output.copy_to_host();
        h_data_ = *output.raw_buffer();
        this->raw_shapes_ = {static_cast<unsigned int>(batch_size),
                             static_cast<unsigned int>(channels),
                             new_rows,
                             new_cols};
    }

    std::vector<uint32_t> Tensor<float>::shapes() const {
//        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        return raw_shapes_;
    }

    void Tensor<float>::Transpose() {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        CHECK_GT(h_data_.dimensions, 2);

        // 将halide_buffer_ 转为 Halide::Buffer
        Halide::Buffer<float> input(h_data_);

        // 获取输入维度信息
        int dim_count = input.dimensions();
        int origin_cols_dim = dim_count - 1;   // cols维度
        int origin_rows_dim = dim_count - 2;   // rows维度

        // 定义 Halide 的变量
        std::vector<Halide::Var> vars(dim_count);
        for (int i = 0; i < dim_count; i++)
            vars[i] = Halide::Var("dim" + std::to_string(i));

        // 定义Halide函数
        Halide::Func transpose("transpose");

        // 转置rows和cols, 使用Halide::Buffer 的索引访问
        transpose(vars) = input(vars[origin_rows_dim], vars[origin_cols_dim]);

        // 调度优化
        transpose.parallel(vars[0]); // 并行处理第一个维度
        transpose.vectorize(vars[origin_cols_dim], 8); // 对倒数第一维向量化

        // 分配输出 Buffer
        std::vector<int> output_extents(dim_count);
        for (int i = 0; i < dim_count; i++) {
            output_extents[i] = input.dim(i).extent();
        }

        // 倒数第一维和倒数第二维的大小需要交换
        output_extents[origin_cols_dim] = input.dim(origin_rows_dim).extent();
        output_extents[origin_rows_dim] = input.dim(origin_cols_dim).extent();

        Halide::Buffer<float> output(output_extents);

        // 生成输出
        transpose.realize(output);

        // 将结果拷贝回 halide_buffer_t
        output.copy_to_host();
        h_data_ = *output.raw_buffer();

        // 更新 raw_shapes_ 信息
        std::swap(raw_shapes_[raw_shapes_.size() - 1], raw_shapes_[raw_shapes_.size() - 2]);
    }

    float &Tensor<float>::index(uint32_t offset) const {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        CHECK(offset < size()) << "Tensor index out of bound!";
        // 获取数据指针并转换为 float*
        auto* data = reinterpret_cast<float*>(h_data_.host);
        return data[offset];
    }

    halide_buffer_t &Tensor<float>::data() {
        return h_data_;
    }

    const halide_buffer_t &Tensor<float>::data() const {
        return h_data_;
    }



//    std::vector<uint32_t> Tensor<float>::unravel_index(uint32_t flat_index, const std::vector<uint32_t>& shape) const {
//        std::vector<uint32_t> indices(shape.size(), 0);
//        uint32_t remain = flat_index;
//        for (int i = shape.size() - 1; i >= 0; --i) {
//            indices[i] = remain % shape[i];
//            remain /= shape[i];
//        }
//        return indices;
//    }

    float Tensor<float>::at(const std::vector<uint32_t> &indices) const {
        CHECK(indices.size() == this->raw_shapes_.size());
        uint32_t batch_size = 0, channel = 0, row = 0, col = 0;

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
        } else if (indices.size() == 4) {
            batch_size = indices[0];
            channel = indices[1];
            row = indices[2];
            col = indices[3];
        }

        return at(batch_size, channel, row, col);
    }

    void Tensor<float>::Expand(const std::vector<uint32_t> &shapes) {
        CHECK(!shapes.empty()) << "Target shapes can not be empty";
        CHECK(shapes.size() <= 4) << "Target shapes dimension exceeds 4D!";

        std::vector<uint32_t > current_shapes = this->raw_shapes_;
        uint32_t current_dims = current_shapes.size();
        uint32_t target_dims = shapes.size();

        // 如果当前形状维度少于目标维度，补充 1
        if (current_dims < target_dims)
            current_shapes.insert(current_shapes.begin(), target_dims - current_dims, 1);

        for (size_t i = 0; i < target_dims; ++i)
            CHECK(current_shapes[i] == shapes[i] || current_shapes[i] == 1)
                 << "Shape mismatch: can not broadcast current shape "
                 << current_shapes[i] << " to target shape " << shapes[i];

        // 构造目标维度描述
        std::vector<halide_dimension_t> target_dim(target_dims);
        uint32_t total_size = 1;
        for (size_t i = 0; i < target_dims; ++i) {
            target_dim[i].min = 0;
            target_dim[i].extent = static_cast<int>(shapes[i]);
            target_dim[i].stride = (i == 0) ? 1 : target_dim[i - 1].stride * target_dim[i - 1].extent;
            total_size *= shapes[i];
        }

        // 分配新的 halide_buffer_t
        halide_buffer_t expanded_data = {
                0, // device
                nullptr, // device_interface
                nullptr, // host
                0,       // flags
                halide_type_t(halide_type_float, 32), // 数据类型
                (int32_t)target_dims, // 维度数
                nullptr, // dimensions
                target_dim.data() // 维度描述
        };

        // 分配 host 内存
        expanded_data.host = (uint8_t *)malloc(total_size * sizeof(float));

        // 遍历所有元素，进行广播
#pragma omp parallel for
        for (uint32_t i = 0; i < total_size; ++i) {
            // 计算目标位置的多维索引
            std::vector<uint32_t> idx(target_dims);
            uint32_t temp = i;
            for (int d = target_dims - 1; d >= 0; --d) {
                idx[d] = temp % shapes[d];
                temp /= shapes[d];
            }

            // 根据多维索引计算源数据的位置
            uint32_t src_offset = 0;
            for (size_t d = 0; d < target_dims; ++d) {
                uint32_t src_idx = (current_shapes[d] == 1) ? 0 : idx[d];
                src_offset += src_idx * ((d < current_dims) ? this->h_data_.dim[d].stride : 0);
            }

            // 复制数据
            auto *src_data = (float *)(h_data_.host);
            auto *dst_data = (float *)(expanded_data.host);
            dst_data[i] = src_data[src_offset];
        }

        if (h_data_.host)
            free(h_data_.host);

        h_data_ = expanded_data;
        raw_shapes_ = shapes;
    }

    void Tensor<float>::Equal(const float &compare_va) const {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT

        // 获取底层数据指针和总元素数
        auto *data_ptr = reinterpret_cast<float *>(h_data_.host);
        const auto total_elements = size();

        // 并行处理数据
#pragma omp parallel for
        for (int i = 0; i < total_elements; ++i) {
            if (data_ptr[i] == compare_va) {
                data_ptr[i] = 0.0f;
            } else {
                data_ptr[i] = 1.0f;
            }
        }
    }

    void Tensor<float>::Transpose(const std::vector<uint32_t> &new_order, bool row_major) {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        using index_type = uint32_t;  // 统一使用 size_t 作为索引类型
        CHECK(new_order.size() == 4);
        CHECK(raw_shapes_.size() == 4);

        // 计算新形状
        std::vector<index_type> new_shape(4);
        for (int i = 0; i < 4; ++i) {
            new_shape[i] = raw_shapes_[new_order[i]];
        }

        // 创建一个新的缓冲区用于存储转置结果
        size_t total_elements = raw_shapes_[0] * raw_shapes_[1] * raw_shapes_[2] * raw_shapes_[3];
        std::vector<float> transposed_data(total_elements);

        // 获取原始数据指针
        auto *data_ptr = reinterpret_cast<float *>(h_data_.host);

        // 计算原始步长（stride）
        std::vector<index_type> old_stride(4);
        old_stride[3] = 1; // 最内层是列
        for (int i = 2; i >= 0; --i) {
            old_stride[i] = old_stride[i + 1] * raw_shapes_[i + 1];
        }

        // 计算新步长（stride）
        std::vector<index_type> new_stride(4);
        new_stride[3] = 1; // 最内层是列
        for (int i = 2; i >= 0; --i)
            new_stride[i] = new_stride[i + 1] * new_shape[i + 1];

        // 并行处理转置
#ifdef _OPENMP
        omp_set_num_threads(omp_get_max_threads());
#endif
#pragma omp parallel for schedule(static)
        for (index_type b = 0; b < raw_shapes_[0]; ++b) {         // batch
            for (index_type c = 0; c < raw_shapes_[1]; ++c) {     // channels
                for (index_type r = 0; r < raw_shapes_[2]; ++r) { // rows
                    for (index_type col = 0; col < raw_shapes_[3]; ++col) { // cols
                        // 计算原始索引
                        index_type old_index = b * old_stride[0] +
                                               c * old_stride[1] +
                                               r * old_stride[2] +
                                               col * old_stride[3];

                        // 计算新索引
                        index_type new_b = (new_order[0] == 0) ? b : (new_order[0] == 1) ? c : (new_order[0] == 2) ? r : col;
                        index_type new_c = (new_order[1] == 0) ? b : (new_order[1] == 1) ? c : (new_order[1] == 2) ? r : col;
                        index_type new_r = (new_order[2] == 0) ? b : (new_order[2] == 1) ? c : (new_order[2] == 2) ? r : col;
                        index_type new_col = (new_order[3] == 0) ? b : (new_order[3] == 1) ? c : (new_order[3] == 2) ? r : col;

                        index_type new_index = new_b * new_stride[0] +
                                               new_c * new_stride[1] +
                                               new_r * new_stride[2] +
                                               new_col * new_stride[3];

                        // 写入转置后的数据
                        transposed_data[new_index] = data_ptr[old_index];
                    }
                }
            }
        }

        // 将转置后的数据复制回 halide_buffer_t
        std::copy(transposed_data.begin(), transposed_data.end(), data_ptr);

        // 更新 halide_buffer_t 的形状
        for (int i = 0; i < 4; ++i) {
            h_data_.dim[i].extent = static_cast<int32_t>(new_shape[i]);
            h_data_.dim[i].stride = static_cast<int32_t>(new_stride[i]);
        }

        raw_shapes_ = new_shape;
    }


    void Tensor<float>::Where(const float &x,
                              const float &y) {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT

        // 获取数据指针
        auto *data_ptr = reinterpret_cast<float *>(h_data_.host);

        // 获取维度信息
        const int dims = h_data_.dimensions;
        std::vector<int> extents(dims);  // 每个维度的大小
        std::vector<int> strides(dims); // 每个维度的步长

        for (int i = 0; i < dims; ++i) {
            extents[i] = h_data_.dim[i].extent;
            strides[i] = h_data_.dim[i].stride;
        }

        // 动态多维循环展开
#pragma omp parallel
        {
            if (dims == 1) {
                // 一维情况
#pragma omp for
                for (int i = 0; i < extents[0]; ++i) {
                    int index = i * strides[0];
                    data_ptr[index] = (data_ptr[index] == 1.0f) ? y : x;
                }
            } else if (dims == 2) {
                // 二维情况
#pragma omp for collapse(2)
                for (int i = 0; i < extents[0]; ++i) {
                    for (int j = 0; j < extents[1]; ++j) {
                        int index = i * strides[0] + j * strides[1];
                        data_ptr[index] = (data_ptr[index] == 1.0f) ? y : x;
                    }
                }
            } else if (dims == 3) {
                // 三维情况
#pragma omp for collapse(3)
                for (int i = 0; i < extents[0]; ++i) {
                    for (int j = 0; j < extents[1]; ++j) {
                        for (int k = 0; k < extents[2]; ++k) {
                            int index = i * strides[0] + j * strides[1] + k * strides[2];
                            data_ptr[index] = (data_ptr[index] == 1.0f) ? y : x;
                        }
                    }
                }
            } else if (dims == 4) {
                // 四维情况
#pragma omp for collapse(4)
                for (int i = 0; i < extents[0]; ++i) {
                    for (int j = 0; j < extents[1]; ++j) {
                        for (int k = 0; k < extents[2]; ++k) {
                            for (int l = 0; l < extents[3]; ++l) {
                                int index = i * strides[0] + j * strides[1] + k * strides[2] + l * strides[3];
                                data_ptr[index] = (data_ptr[index] == 1.0f) ? y : x;
                            }
                        }
                    }
                }
            }
        }
    }

    void Tensor<float>::Sqrt() {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        CHECK(raw_shapes_.empty() || raw_shapes_.size() > 4) << "Unsupported number of dimensions (must be between 1 and 4).";

        // 获取维度信息
        const auto total_elements = size();

        // 获取数据指针
        auto* data_ptr = reinterpret_cast<float*>(h_data_.host);

        // 使用 OpenMP 并行化切片操作
#pragma omp parallel for
        for (size_t idx = 0; idx < total_elements; ++idx) {
            // 直接访问数据并计算平方根
            CHECK(data_ptr[idx] < 0) << "Negative value found in tensor. Sqrt not defined.";
            data_ptr[idx] = std::sqrt(data_ptr[idx]);
        }
    }


    void Tensor<float>::Div(const float &div_num) {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        CHECK(div_num != 0.0f) << "Division by zero";

        // 获取数据指针和尺寸
        auto* data_ptr = reinterpret_cast<float*>(h_data_.host);
        // 获取维度信息
        const auto total_elements = size();

        // 使用 OpenMP 并行化
#pragma omp parallel for
        for (size_t idx = 0; idx < total_elements; ++idx) {
            data_ptr[idx] /= div_num;
        }
    }

    void Tensor<float>::Mul(const std::shared_ptr<Tensor<float>>& other) {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        CHECK(other->h_data_.dimensions == h_data_.dimensions) << "Dimension mismatch between tensors.";


        // 获取数据指针
        auto* data_ptr = reinterpret_cast<float*>(h_data_.host);
        const auto* other_data_ptr = reinterpret_cast<const float*>(other->h_data_.host);

        const auto total_elements = size();


#pragma omp parallel for
        for (size_t idx = 0; idx < total_elements; ++idx) {
            data_ptr[idx] *= other_data_ptr[idx];
        }
    }


    std::vector<sftensor> Tensor<float>::Split(uint32_t &split_axis, const std::vector<uint32_t> &split_lst) {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT
        CHECK(raw_shapes_.empty() || raw_shapes_.size() > 4) << "Unsupported number of dimensions (must be between 1 and 4).";

        // 处理负数轴
        if (split_axis < 0) {
            split_axis = (split_axis + h_data_.dimensions) % h_data_.dimensions;
        }

        CHECK(split_axis >= 0 && split_axis < h_data_.dimensions) << "Unsupported number of dimensions (must be between 1 and 4).";

        // 获取当前轴的长度
        uint32_t axis_extent = raw_shapes_[split_axis];

        // 检查split_lst的总和是否超过当前轴的长度
        uint32_t total_split_len = 0;
        for (auto len : split_lst) {
            total_split_len += len;
        }
        CHECK(total_split_len == axis_extent); // 确保切分总长度等于当前轴长度

        // 提前计算每个切片的起始偏移量
        std::vector<uint32_t> offsets(split_lst.size());
        uint32_t current_offset = h_data_.dim[split_axis].min;
        for (size_t i = 0; i < split_lst.size(); ++i) {
            offsets[i] = current_offset;
            current_offset += split_lst[i];
        }

        // 创建结果容器
        std::vector<std::shared_ptr<Tensor<float>>> ret(split_lst.size());

        // 并行切分
#pragma omp parallel for
        for (size_t i = 0; i < split_lst.size(); ++i) {
            uint32_t split_len = split_lst[i];

            // 创建新的halide_buffer_t
            halide_buffer_t new_buffer = h_data_; // 复制原始buffer
            new_buffer.dim[split_axis].min = offsets[i];
            new_buffer.dim[split_axis].extent = split_len;

            // 修改raw_shapes_以反映新的子张量形状
            std::vector<uint32_t> new_raw_shapes = raw_shapes_;
            new_raw_shapes[split_axis] = split_len;

            // 创建新Tensor并存储到结果中
            ret[i] = std::make_shared<Tensor<float>>(new_buffer, new_raw_shapes);
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
        uint32_t vec_length = 1;

        // 根据输入参数动态生成维度信息
        if (batch_size > 0) {
            dimensions.emplace_back(0,
                                    batch_size,
                                    channels * rows * cols,
                                    0);
            raw_shapes.emplace_back(batch_size);
            vec_length *= batch_size;
        }
        if (channels > 0) {
            dimensions.emplace_back(0,
                                    channels,
                                    rows * cols,
                                    0);
            raw_shapes.emplace_back(channels);
            vec_length *= channels;
        }
        if (rows > 0) {
            dimensions.emplace_back(0,
                                    rows,
                                    cols,
                                    0);
            raw_shapes.emplace_back(rows);
            vec_length *= rows;
        }
        if (cols > 0) {
            dimensions.emplace_back(0,
                                    cols,
                                    1,
                                    0);
            raw_shapes.emplace_back(cols);
            vec_length *= cols;
        }
        // 设置 h_data_ 的维度信息
        h_data_.dimensions = static_cast<int32_t>(dimensions.size());
        h_data_.dim = new halide_dimension_t[h_data_.dimensions];
        h_data_.host = new uint8_t[vec_length * sizeof(float)];
        h_data_.type = halide_type_t(halide_type_float, 32, 1); // 32 位浮点数，标量

        std::reverse(dimensions.begin(), dimensions.end());
        // 将维度信息复制到h_data.dim
        std::copy(dimensions.begin(),
                  dimensions.end(),
                  h_data_.dim);
        raw_shapes_ = raw_shapes;
    }

    void Tensor<float>::set_data(const halide_buffer_t &data) {
        CHECK(h_data_.dimensions == data.dimensions);
        this->h_data_ = data;
    }

    void Tensor<float>::Relu() {
        // Step 1: 用输入Buffer进行包裹
        Halide::Buffer<float> input(h_data_);

        // Step 2: 定义Halide变量和方法
        std::vector<Halide::Var> vars(raw_shapes_.size());
        for (int i = 0; i < raw_shapes_.size(); i++)
            vars[i] = Halide::Var("dim" + std::to_string(i));


        Halide::Func relu("relu");

        // Step 3: 定义动态ReLU操作，
        // 动态创建indexing expression
        std::vector<Halide::Expr> index_expr_lst;
        index_expr_lst.reserve(raw_shapes_.size());
        for (int i = 0; i < raw_shapes_.size(); i++)
            index_expr_lst.emplace_back(vars[i]);

        // 获取输入的动态
        Halide::Expr value = input(index_expr_lst);
        relu(index_expr_lst) = Halide::max(0.0f, value);

        // Step 4: 并行策略
        if (input.dimensions() > 1)
            relu.parallel(vars[0]);

        // Step 6: 实现函数放入输出缓存
        relu.realize(input);

        // Step 7: 拷贝结果返回输入buffer
        input.copy_to_host();
    }

    Tensor<float>::Tensor(const Tensor& tensor) {
        // 创建一个新的 halide_buffer_t
        halide_buffer_t dst = tensor.h_data_; // 值拷贝元数据

        // 计算数据大小
        size_t data_size = tensor.size();;

        // 为新数据分配内存
        auto *new_data = static_cast<uint8_t*>(malloc(data_size * sizeof(float )));
//        CHECK(!new_data) << "Failed to allocate memory for halide_buffer_t deep copy.";

        // 复制原始数据到新内存
        memcpy(new_data, tensor.h_data_.host, data_size);

        // 更新目标 buffer 的 host 指针
        dst.host = static_cast<uint8_t *>(new_data);

        h_data_ = dst;
        raw_shapes_ = tensor.raw_shapes_;
    }

    void Tensor<float>::divide_by(const float &dom) {
        // Step 1: Wrap the input
        using namespace Halide;
        Buffer<float> in(h_data_);

        const int dimensions = h_data_.dimensions;

        // Step 2: Define Halide variables and a function
        std::vector<Var> vars;
        vars.reserve(dimensions);
        for (int i = 0; i < dimensions; ++i)
            vars.emplace_back("dim" + std::to_string(i));
        Func divide("divide");

        std::vector<Expr> args;
        args.reserve(dimensions);
        for (int i = 0; i < dimensions; ++i)
            args.emplace_back(vars[i]);

        // Step 3: Define the computation
        divide(args) = in(args) / dom;

        // Step 4: Schedule the computation
        // Parallelize across rows and vectorize across columns
        if (dimensions == 1)
            divide.vectorize(vars[0], 8, TailStrategy::GuardWithIf);
        else if (dimensions == 2)
            divide.parallel(vars[1]).vectorize(vars[0], 8, TailStrategy::GuardWithIf);
        else {
            // For higher dimensions, parallelize the outermost dimension
            divide.parallel(vars[dimensions - 1]);
            divide.vectorize(vars[0], 8, TailStrategy::GuardWithIf);
        }

        divide.realize(in);

        in.copy_to_host();
    }

    float *Tensor<float>::matrix_raw_ptr(uint32_t index) const {
        CHECK(h_data_.host != nullptr && h_data_.dimensions != 0 && h_data_.dim != nullptr); // NOLINT

        // 返回目标指针
        return reinterpret_cast<float*>(h_data_.host) + index * h_data_.dim[2].stride;
    }
}