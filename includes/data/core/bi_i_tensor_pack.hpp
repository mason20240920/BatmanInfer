//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_BI_I_TENSOR_PACK_HPP
#define BATMANINFER_BI_I_TENSOR_PACK_HPP

#include <unordered_map>

namespace BatmanInfer {
    // 前向声明
    class BIITensor;

    /**
     * @brief 张量打包服务
     */
    class BIITensorPack {
    public:
        struct PackElement
        {
            PackElement() = default;

            PackElement(int id, BIITensor *tensor) : id(id), tensor(tensor), ctensor(nullptr) {

            }

            PackElement(int id, const BIITensor *ctensor) : id(id), tensor(nullptr), ctensor(ctensor) {

            }

            int id{-1};
            BIITensor *tensor{nullptr};
            const BIITensor *ctensor{nullptr};
        };

    public:
        /**
         * @brief 默认构造器
         */
        BIITensorPack() = default;

        /**
         * @brief 初始化函数
         * @param l
         */
        explicit BIITensorPack(std::initializer_list<PackElement> l);

        /**
         * @brief 增加张量到package
         * @param id  张量的id到增加中
         * @param tensor  张量加入
         */
        void add_tensor(int id, BIITensor *tensor);

        /**
         * @brief 增加张量到package
         * @param id
         * @param tensor
         */
        void add_tensor(int id, const BIITensor *tensor);

        /**
         * @brief 把常量张量加入包里面
         * @param id
         * @param tensor
         */
        void add_const_tensor(int id, const BIITensor *tensor);

        /**
         * @brief 从 package 中获取给定 ID 的张量
         * @param id 张量的id来导出
         * @return
         */
        BIITensor *get_tensor(int id);

        /**
         * @brief 获取给定 ID 的常量张量
         * @param id 张量的id来导出
         * @return
         */
        const BIITensor *get_const_tensor(int id) const;

        /**
         * @brief 用给定的id来删除张量
         * @param id
         */
        void remove_tensor(int id);

        /**
         * @brief 获取包里的张量数量
         * @return
         */
        size_t size() const;

        /**
         * @brief 查看包是不是空的
         * @return
         */
        bool empty() const;

    private:
        /**
         * @brief 包含打包张量的容器
         */
        std::unordered_map<int, PackElement> _pack{};
    };
}

#endif //BATMANINFER_BI_I_TENSOR_PACK_HPP
