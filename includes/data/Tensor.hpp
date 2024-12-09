//
// Created by Mason on 2024/10/11.
//

#ifndef BATMAN_INFER_TENSOR_H
#define BATMAN_INFER_TENSOR_H
#include <armadillo>
#include <memory>
#include <vector>
#include <HalideRuntime.h>

namespace BatmanInfer {
    // 模板类: 通用的模板类`Tensor`，可以接受任何类型的`T`（默认是`float`)
    template <typename T = float >
    class Tensor {};

    // 特定类型
    template <>
    class Tensor<uint8_t > {
        // 待实现
    };

    template <>
    class Tensor<bool> {
        // 通过直接初始化调用，而不能通过隐式转换进行调用
        explicit Tensor() = default;

        /**
         * 创建张量
         * @param channels 张量的通道数
         * @param rows  张量的行数
         * @param cols  张量的列数
         */
        explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

        /**
         * 创建一个一维向量
         * @param size  一维向量中元素的个数
         */
        explicit Tensor(uint32_t size);

        /**
         * 创建一个二维向量
         * @param rows  二维向量的高度
         * @param cols  二维向量的宽度
         */
        explicit Tensor(uint32_t rows, uint32_t cols);

        /**
         * 创建张量
         * @param shapes 张量的维度
         */
        explicit Tensor(const std::vector<uint32_t>& shapes);

        /**
         * 构造拷贝函数
         * @param tensor
         */
//        Tensor(const Tensor& tensor);
//
//        /**
//         * 移动构造函数: 用于从一个临时对象（右值）“移动”资源，而不是复制
//         * @param tensor
//         */
//        Tensor(Tensor&& tensor) noexcept;
//
//        /**
//         * 移动赋值运算符: 用于从一个临时对象（右值）移动资源到已经存在的对象中
//         * @param tensor
//         * @return
//         */
//        Tensor<float>& operator=(Tensor&& tensor) noexcept;
//
//        /**
//         * 拷贝赋值运算符
//         * @param tensor
//         * @return
//         */
//        Tensor<float>& operator=(const Tensor& tensor);

        /**
         * 返回张量的行数
         * @return 张量的行数
         */
        uint32_t rows() const;

        /**
         * 返回张量的列数
         * @return 张量的列数
         */
        uint32_t cols() const;

        /**
         * 返回张量的通道数
         * @return 张量的通道数
         */
        uint32_t channels() const;

        /**
         * 返回张量中元素的数量
         * @return 张量的元素数量
         */
        uint32_t size() const;

        /**
         * 设置张量中的具体数据
         * @param data 数据
         */
        void set_data(const arma::fcube& data);

        /**
         * 返回张量是否为空
         * @return 张量是否为空
         */
        bool empty() const;

        /**
         * 返回张量中offset位置的元素
         * @param offset 需要访问的位置
         * @return  offset位置的元素
         */
        float& index(uint32_t offset);

        /**
         * 张量的尺寸大小
         * @return 张量的尺寸大小
         */
        std::vector<uint32_t > shapes() const;

        /**
         * 张量的实际尺寸大小
         * @return 张量的实际尺寸大小
         */
        const std::vector<uint32_t >& raw_shapes() const;

        /**
         * 返回张量中的数据
         * @return 张量的数据
         */
        arma::fcube& data();

        /**
         * 返回张量中的数据
         * @return 张量中的数据
         */
        const arma::fcube& data() const;

        /**
         * 返回张量第channel通道中的数据
         * @param channel 需要返回的通道
         * @return 返回的通道
         */
        arma::Mat<arma::u8>& slice(uint32_t channel);

        /**
         * 返回张量第channel通道中的数据
         * @param channel 需要返回的通道
         * @return 返回的通道
         */
        const arma::Mat<arma::u8>& slice(uint32_t channel) const;

        /**
         * 返回特定位置的元素
         * @param channel 通道
         * @param row  行数
         * @param col 列数
         * @return 特定位置的元素
         */
        arma::u8 at(uint32_t channel, uint32_t row, uint32_t col) const;

        arma::u8 at(const std::vector<uint32_t>& indices) const;

        arma::u8& at(const std::vector<uint32_t>& indices);

        /**
         * 返回特定位置的元素
         * @param channel 通道
         * @param row  行数
         * @param col 列数
         * @return 特定位置的元素
         */
        arma::u8& at(uint32_t channel, uint32_t row, uint32_t col);

        /**
         * 填充张量
         * @param pads  填充张量的尺寸
         * @param padding_value  填充张量
         */
        void Padding(const std::vector<uint32_t>& pads, float padding_value);

        /**
         * 使用value值去初始化向量
         * @param value
         */
        void Fill(arma::u8 value);

        /**
         * 使用values中的数据初始化张量
         * @param values 用来初始化张量的数据
         * @param row_major 是否是行主序列的
         */
        void Fill(const std::vector<arma::u8>& values, bool row_major = true);

        /**
         * 返回Tensor内的所有数据
         * @param row_major 是否是行主序列的
         * @return Tensor内的所有数据
        */
        std::vector<float> values(bool row_major = true);

        /**
         * 以常量1初始化张量
         */
        void Ones();

        /**
         * 以随机值初始化张量
         */
        void Rand();

        /**
         * 打印张量
         */
        void Show();

        /**
         * 张量的实际尺寸大小的Reshape pytorch兼容
         * @param shape 张量的实际尺寸大小
         * @param row_major 根据行主序还是列主序进行reshape
         */
        void Reshape(const std::vector<uint32_t>& shape, bool row_major = false);

        /**
         * 展开张量
         * @param row_major
         */
        void Flatten(bool row_major = false);

        /**
         * 对张量中的元素进行过滤
         * @param filter 过滤函数
         */
        void Transform(const std::function<arma::u8(arma::u8)>& filter);

        /**
         * 返回数据的原始指针
         * @return  返回数据的原始指针
         */
        float* raw_ptr();

        /**
         * 将展开的索引转为多维索引
         * @param flat_index 扁平化的索引（展开后的序号）
         * @param shape  张量的形状
         * @return 多维索引的向量
         */
        std::vector<uint32_t> unravel_index(uint32_t flat_index,
                                            const std::vector<uint32_t>& shape) const;

        /**
         * 对Tensor进行转置
         */
        void Transpose();

        /**
         * 返回第index个矩阵的起始地址
         * @param index 第index个矩阵
         * @return 第index个矩阵的起始地址
         */
        float* matrix_raw_ptr(uint32_t index);

        /**
         * 按照指定形状进行广播(Broadcast)
         * 规则:
         *   1. 如果目标形状的维度数大于当前张量的维度数，在前面补充 1
         *   2. 如果原始维度为 1，可以扩展到目标维度
         *   3. 如果原始维度与目标维度不相等且不为 1，则报错
         * @param shapes 目标形状
         */
        void Expand(const std::vector<uint32_t>& shapes);

        /**
         * 进行Equal全量比较
         * @param compare_va
         */
        void Equal(const float& compare_va);

    private:
        std::vector<uint32_t > raw_shapes_; // 张量数据的实际尺寸大小
        arma::Cube<arma::u8> data_;  //张量数据
    };

    using btensor = Tensor<bool>;
    using bftensor = std::shared_ptr<Tensor<bool>>;

    template <>
    class Tensor<float> {
    public:
        // 通过直接初始化调用，而不能通过隐式转换进行调用
        explicit Tensor() = default;

        /**
         * 创建张量
         * @param channels 张量的通道数
         * @param rows  张量的行数
         * @param cols  张量的列数
         */
        explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

        /**
         * @brief 创建N, C, W, H张量
         * @param batch_size
         * @param channels
         * @param rows
         * @param cols
         */
        explicit Tensor(int32_t batch_size, int32_t channels, int32_t rows, int32_t cols);

        /**
         * 创建一个一维向量
         * @param size  一维向量中元素的个数
         */
        explicit Tensor(uint32_t size);

        /**
         * 创建一个二维向量
         * @param rows  二维向量的高度
         * @param cols  二维向量的宽度
         */
        explicit Tensor(uint32_t rows, uint32_t cols);

        /**
         * 创建张量
         * @param shapes 张量的维度
         */
        explicit Tensor(const std::vector<uint32_t>& shapes);

        /**
         * @brief 返回张量的batch size
         * @return
         */
        uint32_t batch_size() const;

        /**
         * 返回张量的行数
         * @return 张量的行数
         */
        uint32_t rows() const;

        /**
         * 返回张量的列数
         * @return 张量的列数
         */
        uint32_t cols() const;

        /**
         * 返回张量的通道数
         * @return 张量的通道数
         */
        uint32_t channels() const;

        /**
         * 返回张量中元素的数量
         * @return 张量的元素数量
         */
        uint32_t size() const;

        /**
         * 设置张量中的具体数据
         * @param data 数据
         */
        void set_data(const arma::fcube& data);

        /**
         * 返回张量是否为空
         * @return 张量是否为空
         */
        bool empty() const;

        /**
         * 返回张量中offset位置的元素
         * @param offset 需要访问的位置
         * @return  offset位置的元素
         */
        float& index(uint32_t offset);

        /**
         * 张量的尺寸大小
         * @return 张量的尺寸大小
         */
        std::vector<uint32_t > shapes() const;

        /**
         * 张量的实际尺寸大小
         * @return 张量的实际尺寸大小
         */
        const std::vector<uint32_t >& raw_shapes() const;

        /**
         * 返回张量中的数据
         * @return 张量的数据
         */
        arma::fcube& data();

        /**
         * 返回张量中的数据
         * @return 张量中的数据
         */
        const arma::fcube& data() const;

        /**
         * 返回张量第channel通道中的数据
         * @param channel 需要返回的通道
         * @return 返回的通道
         */
        halide_buffer_t* slice(uint32_t channel);

        /**
         * 返回张量第channel通道中的数据
         * @param channel 需要返回的通道
         * @return 返回的通道
         */
        const halide_buffer_t* slice(uint32_t channel) const;

        /**
         * 返回特定位置的元素
         * @param channel 通道
         * @param row  行数
         * @param col 列数
         * @return 特定位置的元素
         */
        float at(uint32_t batch_size,
                 uint32_t channel,
                 uint32_t row,
                 uint32_t col) const;

        float at(const std::vector<uint32_t>& indices) const;

        float& at(const std::vector<uint32_t>& indices);

        /**
         * 返回特定位置的元素
         * @param channel 通道
         * @param row  行数
         * @param col 列数
         * @return 特定位置的元素
         */
        float at(uint32_t channel, uint32_t row, uint32_t col);

        /**
         * 填充张量
         * @param pads  填充张量的尺寸
         * @param padding_value  填充张量
         */
        void Padding(const std::vector<uint32_t>& pads, float padding_value);

        /**
         * 使用value值去初始化向量
         * @param value
         */
        void Fill(float value);

        /**
         * 使用values中的数据初始化张量
         * @param values 用来初始化张量的数据
         * @param row_major 是否是行主序列的
         */
        void Fill(const std::vector<float>& values, bool row_major = true);

        /**
         * 返回Tensor内的所有数据
         * @param row_major 是否是行主序列的
         * @return Tensor内的所有数据
        */
        std::vector<float> values(bool row_major = true);

        /**
         * 以常量1初始化张量
         */
        void Ones();

        /**
         * 以随机值初始化张量
         */
        void Rand();

        /**
         * 打印张量
         */
        void Show();

        /**
         * 张量的实际尺寸大小的Reshape pytorch兼容
         * @param shape 张量的实际尺寸大小
         * @param row_major 根据行主序还是列主序进行reshape
         */
        void Reshape(const std::vector<uint32_t>& shape, bool row_major = false);

        /**
         * 展开张量
         * @param row_major
         */
        void Flatten(bool row_major = false);

        /**
         * 对张量中的元素进行过滤
         * @param filter 过滤函数
         */
        void Transform(const std::function<float(float)>& filter);

        /**
         * 返回数据的原始指针
         * @return  返回数据的原始指针
         */
        float* raw_ptr();

        /**
         * 将展开的索引转为多维索引
         * @param flat_index 扁平化的索引（展开后的序号）
         * @param shape  张量的形状
         * @return 多维索引的向量
         */
        std::vector<uint32_t> unravel_index(uint32_t flat_index,
                                            const std::vector<uint32_t>& shape) const;

        /**
         * 对Tensor进行转置
         */
        void Transpose();


        /**
         * @brief 对矩阵进行多维变换
         * @param new_order 新的维度
         * @param row_major 是不是行优先
         */
        void Transpose(const std::vector<uint32_t>& new_order,
                       bool row_major);

        /**
         * 返回第index个矩阵的起始地址
         * @param index 第index个矩阵
         * @return 第index个矩阵的起始地址
         */
        float* matrix_raw_ptr(uint32_t index);

        /**
         * 按照指定形状进行广播(Broadcast)
         * 规则:
         *   1. 如果目标形状的维度数大于当前张量的维度数，在前面补充 1
         *   2. 如果原始维度为 1，可以扩展到目标维度
         *   3. 如果原始维度与目标维度不相等且不为 1，则报错
         * @param shapes 目标形状
         */
        void Expand(const std::vector<uint32_t>& shapes);

        /**
         * 进行Equal全量比较
         * @param compare_va
         */
        void Equal(const float& compare_va);

        /**
         * 进行Where操作. 自己是条件张量，进行结果生成
         * @param x
         * @param y
         */
        void Where(const float& x,
                   const float& y);

        /**
        *  进行平方根处理
        */
        void Sqrt();

        /**
        * 张量同时除以一个数
        */
        void Div(const float& div_num);

        /**
         * @brief 对Tensor进行切分
         * @param split_axis 切分的轴
         * @param split_size 切分的长度
         */
        std::vector<std::shared_ptr<Tensor<float>>> Split(const uint32_t & split_axis,
                                                          const std::vector<uint32_t> & split_lst);

        /**
         * @brief 进行逐元素相乘
         * @param other
         */
        void Mul(const std::shared_ptr<Tensor<float>>& other);

    private:

        std::vector<uint32_t > raw_shapes_; // 张量数据的实际尺寸大小
//        arma::fcube data_;  //张量数据

        // 新增halide变量
        halide_buffer_t h_data_;
    };

    using ftensor = Tensor<float>;
    using sftensor = std::shared_ptr<Tensor<float>>;
}

#endif //BATMAN_INFER_TENSOR_H
