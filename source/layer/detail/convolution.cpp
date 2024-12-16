//
// Created by Mason on 2024/11/7.
//

#include <layer/detail/convolution.hpp>
#include <glog/logging.h>
#include <layer/abstract/layer_factory.hpp>
#include <Halide.h>

namespace BatmanInfer {
    sftensor ConvolutionLayer::Im2Col(const BatmanInfer::sftensor &input,
                                             uint32_t kernel_w,
                                             uint32_t kernel_h,
                                             uint32_t input_w,
                                             uint32_t input_h,
                                             uint32_t input_c_group,
                                             uint32_t group,
                                             uint32_t row_len,
                                             uint32_t col_len) const {
        /**
         * @brief 输出的矩阵:
         * @row 行数: 组内的条数 * 卷积核平摊长度
         * @col 列数: 平摊输出矩阵的长度
         */
        auto input_matrix = std::make_shared<ftensor>(input_c_group * row_len, col_len);
        // 输入的矩阵高度
        const uint32_t input_padded_h = input_h + top_padding_ + bottom_padding_;
        // 输入矩阵的宽度
        const uint32_t input_padded_w = input_w + left_padding_ + right_padding_;
        // padding的数字为0
        const float padding_value = 0.f;
        for (uint32_t ic = 0; ic < input_c_group; ++ic) {
            // 输入矩阵的第一个元素指针
            float* input_channel_ptr = input->matrix_raw_ptr(ic + group * input_c_group);
            // 当前列号
            uint32_t current_col = 0;
            // 当前通道index * 卷积核长度
            uint32_t channel_row = ic * row_len;
            for (uint32_t w = 0; w < input_padded_w - kernel_w + 1; w += stride_w_) {
                for (uint32_t r = 0; r < input_padded_h - kernel_h + 1; r += stride_h_) {
                    float* input_matrix_ptr =
                            input_matrix.colptr(current_col) + channel_row;
                    current_col += 1;
                    for (uint32_t kw = 0; kw < kernel_w; ++kw) {
                        const uint32_t region_w = input_h * (w + kw - padding_w_);
                        for (uint32_t kh = 0; kh < kernel_h; ++kh) {
                            if ((kh + r >= padding_h_ && kw + w >= padding_w_) &&
                                (kh + r < input_h + padding_h_ &&
                                 kw + w < input_w + padding_w_)) {
                                float* region_ptr =
                                        input_channel_ptr + region_w + (r + kh - padding_h_);
                                *input_matrix_ptr = *region_ptr;
                            } else {
                                *input_matrix_ptr = padding_value;  // only support zero mode
                            }
                            input_matrix_ptr += 1;
                        }
                    }
                }
            }

        }
        return input_matrix;
    }


    ConvolutionLayer::ConvolutionLayer(uint32_t output_channel,
                                       uint32_t in_channel,
                                       uint32_t kernel_h,
                                       uint32_t kernel_w,
                                       uint32_t padding_t,
                                       uint32_t padding_l,
                                       uint32_t padding_b,
                                       uint32_t padding_r,
                                       uint32_t stride_h,
                                       uint32_t stride_w,
                                       uint32_t groups,
                                       bool use_bias) : ParamLayer("Convolution"),
                                                        use_bias_(use_bias),
                                                        groups_(groups),
                                                        top_padding_(padding_t),
                                                        left_padding_(padding_l),
                                                        bottom_padding_(padding_b),
                                                        right_padding_(padding_r),
                                                        stride_h_(stride_h),
                                                        stride_w_(stride_w) {
        // 初始化权重
        if (groups != 1)
            in_channel /= groups;
        this->InitWeightParam(output_channel, in_channel, kernel_h, kernel_w);
        if (use_bias)
            this->InitBiasParam(output_channel, 0, 0, 0);
    }

    InferStatus ConvolutionLayer::Forward(const std::map<std::string, std::shared_ptr<Tensor<float>>> &inputs,
                                          std::map<std::string, std::shared_ptr<Tensor<float>>> &outputs) {
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the convolution layer is empty";
            return InferStatus::bInferFailedInputEmpty;
        }

        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the convolution "
                       << "layer do not match";
            return InferStatus::bInferFailedInputOutSizeMatchError;
        }

        if (weights_->empty()) {
            LOG(ERROR) << "The number of kernel matrix in the convolution layer should "
                          "be greater than zero";
            return InferStatus::bInferFailedWeightParameterError;
        }

        if (this->use_bias_ && this->bias_->cols() != this->weights_->batch_size()) {
            LOG(ERROR) << "The number of kernel matrix and bias matrix do not match";
            return InferStatus::bInferFailedBiasParameterError;
        }

        if (!stride_h_ || !stride_w_) {
            LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                          "greater than 0";
            return InferStatus::bInferFailedStrideParameterError;
        }

        // 卷积核的总数量(关注特征不同)
        const uint32_t kernel_count = weights_->batch_size();
        // 单个卷积核的高度
        const uint32_t kernel_h = weights_->rows();
        // 单个卷积核的宽度
        const uint32_t kernel_w = weights_->cols();
        // 单个卷积核的通道数（与输入特征必须一致）
        const uint32_t kernel_c = weights_->channels();
        // 单个卷积核的一个通道展开成一行后的长度
        const uint32_t row_len = kernel_h * kernel_w;
        CHECK(kernel_h > 0 && kernel_w > 0 && kernel_c > 0)
              << "The size of kernel matrix in the convolution layer should be greater "
              << "than zero";

        // 表示每个组内的卷积核数量 (每个组内的卷积核数量)
        // 分组卷积将输入通道和卷积核分为多个组，每个组只在内部进行卷积操作。这样可以减少计算量和参数数量
        const uint32_t kernel_count_group = kernel_count / groups_;
        const uint32_t batch_size = inputs.size();

        if (kernel_tensor_.empty()) {
            this->InitIm2ColWeight();
        }

        if (!kernel_tensor_.empty()) {
            if (groups_ == 1)
                CHECK(kernel_tensor_.size() == kernel_count_group)
                << "The number of kernel matrix and kernel_count_group do not match";
            else
                CHECK(kernel_tensor_.size() == kernel_count)
                << "The number of kernel matrix and kernel_count do not match";
        }

        for (const auto&[_, input_tensor]: inputs) {
            // 输入的channels
            const uint32_t input_c = input_tensor->channels();
            // 输入的padding的值
            const uint32_t input_padded_h = input_tensor->rows() + top_padding_ + bottom_padding_;
            const uint32_t input_padded_w = input_tensor->cols() + left_padding_ + right_padding_;

            const uint32_t output_h = std::floor((int(input_padded_h) - int(kernel_h)) / stride_h_ + 1);
            const uint32_t output_w =
                    std::floor((int(input_padded_w) - int(kernel_w)) / stride_w_ + 1);
            CHECK(output_h > 0 && output_w > 0)
                            << "The size of the output tensor should be greater than zero ";

            uint32_t col_len = output_h * output_w;
            CHECK(col_len > 0) << "Output_h x output_w for the convolution layer "
                                  "should be greater than zero";

            uint32_t input_c_group = input_c / groups_;
            CHECK(input_c_group == kernel_c) << "The number of channel for the kernel "
                                                "matrix and input tensor do not match";

            for (uint32_t g = 0; g < groups_; ++g) {
                const auto& input_matrix = Im2Col(input_tensor,
                                                  kernel_w,
                                                  kernel_h,
                                                  input_tensor->cols(),
                                                  input_tensor->rows(),
                                                  input_c_group,
                                                  g,
                                                  row_len,
                                                  col_len);
                std::cout << "Hello World" << std::endl;
            }
        }


//        for (uint32_t i = 0; i < batch_size; ++i) {
//            const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
//            CHECK(input != nullptr && !input->empty())
//                 << "The input tensor array in the convolution layer has an empty "
//                    "tensor " << i << " th";
//
//            const uint32_t input_c = input->channels();
//            const uint32_t input_padded_h = input->rows() + top_padding_ + bottom_padding_;
//            const uint32_t input_padded_w = input->cols() + left_padding_ + right_padding_;
//
//            const uint32_t output_h = std::floor((int(input_padded_h) - int(kernel_h)) / stride_h_ + 1);
//            const uint32_t output_w =
//                    std::floor((int(input_padded_w) - int(kernel_w)) / stride_w_ + 1);
//            CHECK(output_h > 0 && output_w > 0)
//                            << "The size of the output tensor should be greater than zero " << i
//                            << " th";
//
//            if (groups_ != 1) {
//                CHECK(kernel_count % groups_ == 0);
//                CHECK(input_c % groups_ == 0);
//            }
//
//            uint32_t col_len = output_h * output_w;
//            CHECK(col_len > 0) << "Output_h x output_w for the convolution layer "
//                                  "should be greater than zero"
//                                  << i << " th";
//
//            uint32_t input_c_group = input_c / groups_;
//            CHECK(input_c_group == kernel_c) << "The number of channel for the kernel "
//                                                "matrix and input tensor do not match";
//
//            for (uint32_t g = 0; g < groups_; ++g) {
//                const auto& input_matrix = Im2Col(input,
//                                                  kernel_w,
//                                                  kernel_h,
//                                                  input->cols(),
//                                                  input->rows(),
//                                                  input_c_group,
//                                                  g,
//                                                  row_len,
//                                                  col_len);
//                std::shared_ptr<Tensor<float>> output_tensor = outputs.at(i);
//                if (output_tensor == nullptr || output_tensor->empty()) {
//                    output_tensor = std::make_shared<Tensor<float>>(kernel_count,
//                            output_h,
//                            output_w);
//                    outputs.at(i) = output_tensor;
//                }
//
//                CHECK(output_tensor->rows() == output_h &&
//                      output_tensor->cols() == output_w &&
//                      output_tensor->channels() == kernel_count)
//                      << "The output tensor array in the convolution layer has an "
//                         "incorrectly sized tensor "
//                         << i << "th";
//
//                const uint32_t kernel_count_group_start = kernel_count_group * g;
//                for (uint32_t k = 0; k < kernel_count_group; ++k) {
//                    arma::frowvec kernel;
//                    if (groups_ == 1) {
//                        kernel = kernel_matrix_arr_.at(k);
//                    } else {
//                        kernel = kernel_matrix_arr_.at(kernel_count_group_start + k);
//                    }
//                    ConvGemmBias(input_matrix, output_tensor, g, k, kernel_count_group,
//                                 kernel, output_w, output_h);
//                }
//            }
//        }
        return InferStatus::bInferSuccess;
    }

    /**
     * 打印向量的数组
     * @param kernel_matrix_arr
     */
    void printKernelMatrixArr(const std::vector<arma::frowvec> &kernel_matrix_arr) {
        for (size_t i = 0; i < kernel_matrix_arr.size(); ++i) {
            std::cout << "Kernel Matrix " << i << ":" << std::endl;
            std::cout << kernel_matrix_arr[i] << std::endl;
        }
    }

    /**
     * 卷积核权重转为适合矩阵乘法操作的格式
     * 卷积层，包含一下参数
     * * 卷积核(kernel count): 4
     * * 每个卷积核的尺寸3 x 3(kernel_h和kernel_w)
     * *
     */
    void ConvolutionLayer::InitIm2ColWeight() {
        // 1. 检查卷积核数量
        const uint32_t kernel_count = weights_->batch_size();
        CHECK(kernel_count > 0) << "kernel count must greater than zero";
        // 2. 获取卷积核的尺寸和通道数
        // 卷积核高度
        const uint32_t kernel_h = weights_->rows();
        // 卷积核宽度
        const uint32_t kernel_w = weights_->cols();
        // 通道数
        const uint32_t kernel_c = weights_->channels();
        // 4. 初始化卷积核矩阵
        if (kernel_count % groups_ != 0)
            LOG(ERROR) << "Kernel count must be divisible by group";

        const uint32_t split_kernel_count = kernel_count / groups_;

        // Output dimensions: kernel_count x (C * H * W)
        const uint32_t flattened_size = split_kernel_count * kernel_c * kernel_h * kernel_w;

        Halide::Buffer<float> input(weights_->data());

        // Process each group separately
        for (int g = 0; g < groups_; ++g) {
            // Create the output halide_buffer_t
            auto temp_tensor = std::make_shared<ftensor>(1, flattened_size);

            // Halide function for Im2Row transformation
            Halide::Func im2col;
            // 定义展开逻辑
            Halide::Var i, n;

            // Define the computation for the current group
            // Map the input indices to the appropriate group
            im2col(n, i) = input(
                    (n % static_cast<int>(kernel_w)),                                // W
                    (n / static_cast<int>(kernel_w)) % static_cast<int>(kernel_h),                        // H
                    (n / static_cast<int>(kernel_w * kernel_h)) % static_cast<int>(kernel_c),            // C
                    g * static_cast<int>(split_kernel_count) + (n / static_cast<int>(kernel_w * kernel_h * kernel_c)) // K
            );
            // Schedule: optimize for parallelism and vectorization if needed
            im2col.parallel(i).vectorize(n, 8, Halide::TailStrategy::GuardWithIf);

            // Realize the output buffer for the current group
            Halide::Buffer<float> output(temp_tensor->data());
            im2col.realize(output);

            temp_tensor->Show();

            // Store the output buffer in the vector
            kernel_tensor_.push_back(temp_tensor);
        }
    }

    void ConvolutionLayer::ConvGemmBias(const arma::fmat &input_matrix, BatmanInfer::sftensor output_tensor,
                                        uint32_t group, uint32_t kernel_index, uint32_t kernel_count_group,
                                        const arma::frowvec &kernel, uint32_t output_w, uint32_t output_h) const {
//        arma::fmat output(output_tensor->matrix_raw_ptr(kernel_index + group * kernel_count_group),
//                          output_h, output_w, false, true);
//
//        CHECK(output.size() == output_h * output_w)
//             << "Output_h x output_w for the convolution layer " << "should be output tensor size";
//
//        bool use_bias = (!this->bias_.empty() && this->use_bias_);
//
//        float bias_value = 0.0f;
//        if (use_bias) {
//            std::shared_ptr<Tensor<float>> bias = this->bias_.at(kernel_index);
//            if (bias != nullptr && !bias->empty()) {
//                bias_value = bias->index(0);
//            } else {
//                LOG(FATAL) << "Bias tensor is empty or nullptr";
//            }
//        }
//
//        uint32_t total_elements = output_h * output_w;
//        uint32_t K = input_matrix.n_rows; // 输入矩阵的行数
//
//        // 使用 OpenMP 并行化外部循环
//#pragma omp parallel for
//        for (uint32_t n = 0; n < total_elements; ++n) {
//            float sum = 0.0f;
//            const float* input_col = input_matrix.colptr(n);
//
//            // 内部循环计算 kernel 与 input_matrix 的乘积
//            for (uint32_t k = 0; k < K; ++k) {
//                sum += kernel(k) * input_col[k];
//            }
//            output(n) = sum + bias_value; // 添加偏置值（如果存在）
//        }
    }

    ParseParameterAttrStatus ConvolutionLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                           std::shared_ptr<Layer> &conv_layer) {
        CHECK(op != nullptr) << "Convolution operator is nullptr";
        const std::map<std::string, std::shared_ptr<RuntimeParameter>> &params = op->params;
        const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &attributes = op->attribute;

        if (params.find("dilations") == params.end()) {
            LOG(ERROR) << "Can not find the dilation parameter";
            return ParseParameterAttrStatus::bParameterMissingDilation;
        }

        auto dilation_param = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("dilations"));
        if (dilation_param == nullptr || dilation_param->value.size() != 2) {
            LOG(ERROR) << "Can not find the dilation parameter";
            return ParseParameterAttrStatus::bParameterMissingDilation;
        }

        CHECK(dilation_param->value.at(0) != 1 || dilation_param->value.at(1))
                        << "Only support dilation value equals to one!";

        if (params.find("pads") == params.end()) {
            LOG(ERROR) << "Can not find the padding parameter";
            return ParseParameterAttrStatus::bParameterMissingPadding;
        }

        auto padding =
                std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("pads"));
        if (!padding) {
            LOG(ERROR) << "Can not find the padding parameter";
            return ParseParameterAttrStatus::bParameterMissingPadding;
        }

        if (params.find("strides") == params.end()) {
            LOG(ERROR) << "Can not find the stride parameter";
            return ParseParameterAttrStatus::bParameterMissingStride;
        }
        auto stride =
                std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("strides"));
        if (!stride) {
            LOG(ERROR) << "Can not find the stride parameter";
            return ParseParameterAttrStatus::bParameterMissingStride;
        }

        if (params.find("kernel_shape") == params.end()) {
            LOG(ERROR) << "Can not find the kernel parameter";
            return ParseParameterAttrStatus::bParameterMissingKernel;
        }
        auto kernel = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
                params.at("kernel_shape"));
        if (!kernel) {
            LOG(ERROR) << "Can not find the kernel parameter";
            return ParseParameterAttrStatus::bParameterMissingKernel;
        }

        if (attributes.size() != 2) {
            LOG(ERROR) << "Can not find the weights and bias params";
            return ParseParameterAttrStatus::bAttrMissingWeight;
        }

        // Judge which one is bias or weight
        auto shape_size = attributes.begin()->second->shape.size();
        std::shared_ptr<RuntimeAttribute> bias, weight;
        if (shape_size == 1) {
            bias = attributes.begin()->second;
            weight = (++attributes.begin())->second;
        } else {
            weight = attributes.begin()->second;
            bias = (++attributes.begin())->second;
        }

        auto groups =
                std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("group"));
        if (!groups) {
            LOG(ERROR) << "Can not find the groups parameter";
            return ParseParameterAttrStatus::bParameterMissingGroups;
        }

        const uint32_t dims = 2;
        const std::vector<int> &kernels = kernel->value;
        const std::vector<int> &paddings = padding->value;
        const std::vector<int> &strides = stride->value;
        if (paddings.size() != 4) {
            LOG(ERROR) << "Can not find the right padding parameter";
            return ParseParameterAttrStatus::bParameterMissingPadding;
        }

        if (strides.size() != dims) {
            LOG(ERROR) << "Can not find the right stride parameter";
            return ParseParameterAttrStatus::bParameterMissingStride;
        }

        if (kernels.size() != dims) {
            LOG(ERROR) << "Can not find the right kernel size parameter";
            return ParseParameterAttrStatus::bParameterMissingKernel;
        }

        // 从attribute里面获取onnx的weights和bias
        // 卷积核数量
        auto kernel_count = weight->shape.at(0);

        // 卷积核的channel
        auto kernel_channel = weight->shape.at(1);

        // kernel的方向是倒置的
        conv_layer = std::make_shared<ConvolutionLayer>(
                kernel_count,
                kernel_channel,
                kernels.at(0),
                kernels.at(1),
                paddings.at(0),
                paddings.at(1),
                paddings.at(2),
                paddings.at(3),
                strides.at(0),
                strides.at(1),
                groups->value, true);

        // load weights
        const std::vector<float> &bias_value = bias->weight_data;
        conv_layer->set_bias(bias_value);

        const std::vector<int> &weight_shape = weight->shape;
        if (weight_shape.empty()) {
            LOG(ERROR) << "The attribute of weight shape is wrong";
            return ParseParameterAttrStatus::bAttrMissingWeight;
        }
        const std::vector<float> &weight_values = weight->weight_data;
        conv_layer->set_weights(weight_values);

        auto conv_layer_derived = std::dynamic_pointer_cast<ConvolutionLayer>(conv_layer);
        CHECK(conv_layer_derived != nullptr);
        conv_layer_derived->InitIm2ColWeight();
        return ParseParameterAttrStatus::bParameterAttrParseSuccess;
    }

    LayerRegistererWrapper bConvGetInstance("Conv",
                                            ConvolutionLayer::GetInstance);
}