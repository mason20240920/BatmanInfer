//
// Created by Mason on 2024/10/13.
//

#ifndef BATMANINFER_ONNXUTILS_HPP
#define BATMANINFER_ONNXUTILS_HPP

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <onnx/onnx_pb.h>

namespace BatmanInfer {
    bool onnx_read_proto_from_binary(const char* filepath,
                                     google::protobuf::Message* message);

    bool onnx_write_proto_from_binary(const char* filepath,
                                      const google::protobuf::Message* message);

    void getOperatorAndOperandCount(const onnx::ModelProto& model,
                                    int& operator_count,
                                    int& operand_count);

    /**
     * 将ONNX的模型映射为自定义的model里面
     * @param onnx_type
     * @return
     */
    int map_onnx_type_to_custom_type(int onnx_type);

    /**
     * 获取数据输入的数量
     * @param node 当前算子
     * @return 算子当前的input数量
     */
    int get_data_input_count(const onnx::NodeProto &node);
}

#endif //BATMANINFER_ONNXUTILS_HPP
