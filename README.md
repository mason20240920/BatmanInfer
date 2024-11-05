# BatmanInfer

### 项目目标

* 设备: Mobile GPU、CPU

### 1.1 一期目标 

* 实现LSTM模型编译 (2024-10-31截止)
* 实现GPT-2模型编译
* 实现Int8量化

> **技术难点**
>
> KVCaches增加
>
> `ONNX`模型转译`Operator`
>
> `OpenCL`和`GPU` 中的算子加速

**已知问题**

1. 编译`onnx`到系统需要先进行```protoc -I=. --cpp_out=. onnx.proto```
2. 关注`CMakeLists`几个库的设置

### 1.2 OpenMP的`API`接口

#### `#pragma omp parallel for`

> 循环的每次迭代可以由一个独立的线程执行

`#pragma omp critical`

> 临界区是指一段必须由一个线程独占执行的代码，以防止数据竞争

`pragma omp simd`

> 用于将 SIMD（单指令多数据）向量化应用于后续的循环
