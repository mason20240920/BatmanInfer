//
// Created by Mason on 2024/10/13.
//

#ifndef BATMAN_INFER_IR_H
#define BATMAN_INFER_IR_H

#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <onnx/onnx_pb.h>
//#include <onnx/proto_utils.h>

namespace BatmanInfer {
    class ONNXParameter {
    public:
        ONNXParameter() : type(0) {}

        explicit ONNXParameter(bool _b) : type(1), b(_b) {}

        explicit ONNXParameter(int _i) : type(2), i(_i) {}

        explicit ONNXParameter(long _l) : type(2), i(_l) {}

        explicit ONNXParameter(long long _l) : type(2), i(_l) {}

        explicit ONNXParameter(float _f) : type(3), f(_f) {}

        explicit ONNXParameter(double _f) : type(3), f(_f) {}

        explicit ONNXParameter(const char *_s) : type(4), s(_s) {}

        explicit ONNXParameter(const std::string &_s) : type(4), s(_s) {}

        explicit ONNXParameter(const std::initializer_list<int> &_ai) : type(5), ai(_ai) {}

        explicit ONNXParameter(const std::initializer_list<int64_t> &_ai) : type(5) {
            for (const auto &x: _ai)
                ai.push_back((int) x);
        }

        ONNXParameter(const std::initializer_list<float> &_af)
                : type(6), af(_af) {
        }

        ONNXParameter(const std::initializer_list<double> &_af)
                : type(6) {
            for (const auto &x: _af)
                af.push_back((float) x);
        }

        explicit ONNXParameter(const std::vector<float> &_af)
                : type(6), af(_af) {
        }

        ONNXParameter(const std::initializer_list<const char *> &_as)
                : type(7) {
            for (const auto &x: _as)
                as.push_back(std::string(x));
        }

        ONNXParameter(const std::initializer_list<std::string> &_as)
                : type(7), as(_as) {
        }

        explicit ONNXParameter(const std::vector<std::string> &_as)
                : type(7), as(_as) {
        }

        static ONNXParameter parse_from_string(const std::string& value);


        // 0=null 1=b 2=i 3=f 4=s 5=ai 6=af 7=as 8=others
        int type;

        // value
        bool b{};
        int i{};
        float f{};
        std::vector<int> ai;
        std::vector<float> af;

        // keep std::string typed member the last for cross cxxabi compatibility
        std::string s;
        std::vector<std::string> as;
    };

    bool operator==(const ONNXParameter& lhs, const ONNXParameter& rhs);

    class ONNXAttribute {
    public:
        ONNXAttribute() : type(0) {}

        explicit ONNXAttribute(const onnx::TensorProto& tensor);

        std::vector<int> shape;
        std::vector<float> data;
        // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool
        int type;
    };

    bool operator==(const ONNXAttribute& lhs, const ONNXAttribute& rhs);
    // concat two attributes along the first axis
    ONNXAttribute operator+(const ONNXAttribute& a, const ONNXAttribute& b);

    class ONNXOperator;
    class ONNXOperand {
    public:
        // 生产者
        ONNXOperator* producer;

        // 消费者列表
        std::vector<ONNXOperator *> consumers;

        // 数据类型
        // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool 10=cp64 11=cp128 12=cp32
        int type;

        // 操作数的形状
        std::vector<int> shape;

        // 操作数的名称
        // keep std::string typed member the last for cross cxxabi compatibility
        std::string name;

        // 存储参数
        std::map<std::string, ONNXParameter> params;
    };

    class ONNXOperator {
    public:
        // 操作符的名称
        std::string name;

        // 操作符的类型
        std::string type;

        // 输入操作数
        std::vector<ONNXOperand *> inputs;

        // 输出操作数
        std::vector<ONNXOperand *> outputs;

        // 输入操作数的名称
        std::vector<std::string > input_names;

        // 参数
        std::map<std::string, ONNXParameter> params;

        // 属性
        std::map<std::string, ONNXAttribute> attrs;
    };

    class ONNXGraph {
    public:
        ONNXGraph();
        ~ONNXGraph();

        int load(const std::string& model_path);

//        void save(const std::string& model_path);

        /**
         * 创建一个新的操作符
         * @param type: 操作符的类型
         * @param name: 操作符的名称
         * @return: 指向新创建的 ONNX Operator 的指针
         */
        ONNXOperator* new_operator(const std::string& type,
                                   const std::string& name);

        /**
         * 在指定操作符之前插入一个新的操作符
         * @param type: 操作符的类型
         * @param name: 操作符的名称
         * @param cur: 当前操作符的指针
         * @return: 指向新创建的 ONNX Operator 的指针
         */
        ONNXOperator* new_operator_before(const std::string& type,
                                          const std::string& name,
                                          const ONNXOperator* cur);

        /**
         * 在指定操作符之后插入一个新的操作符
         * @param type: 操作符的类型
         * @param name: 操作符的名称
         * @param cur: 当前操作符的指针
         * @return: 指向新创建的 ONNX Operator 的指针
         */
        ONNXOperator* new_operator_after(const std::string& type,
                                         const std::string& name,
                                         const ONNXOperator* cur);

        /**
         * 创建一个新的操作数
         * @param name: 操作数的名称
         * @return: 指向新创建的 ONNXOperand 的指针
         */
        ONNXOperand* new_operand(const std::string& name);

        /**
         * 获取指定名称的操作数
         * @param name: 操作数的名称
         * @return: 指向找到的 ONNXOperand 的指针，如果不存在则返回 nullptr
         */
        ONNXOperand* get_operand(const std::string& name);

        /**
         * 获取指定名称的操作数（常量版本）
         * @param name: 操作数的名称
         * @return: 指向找到的常量 ONNXOperand 的指针，如果不存在则返回 nullptr
         */
        [[nodiscard]] const ONNXOperand* get_operand(const std::string& name) const;

        // 存储图中的所有操作符
        std::vector<ONNXOperator*> operators;

        // 存储图中的所有操作数
        std::vector<ONNXOperand*> operands;

    private:
        /**
         * 私有复制构造函数：防止对象被复制
         * @param rhs
         */
        ONNXGraph(const ONNXGraph& rhs);

        /**
         * 私有赋值运算符：防止对象被复制
         * @param rhs
         * @return
         */
        ONNXGraph& operator=(const ONNXGraph& rhs);
    };
}

#endif //BATMAN_INFER_IR_H
