# BatmanInfer

## - Why do we fall, Bruce? So we can learn to pick ourselves up

 <p align="center">
   <img src="./images/image-20241128140900108.png" alt="image-20241128140900108" style="zoom:35%;" />
 </p>

 </p>

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

`#pragma omp parallel for collapse(3) schedule(static)`

> `#pragma omp parallel`
>
> * 创建一个并行区域
> * 生成一组并行执行的线程
> * 默认线程数通常等于系统的 CPU 核心数
>
> `for`
>
> * 指示接下来的 for 循环将被并行化
>
> * 循环的迭代将被分配给不同的线程执行
>
> * 每个线程负责执行一部分迭代
>
> `collapse(3)`
>
> * 将接下来的 3 层嵌套循环合并成一个大的迭代空间
>
> * collapse(3) 会将这 1000 次迭代(10×10×10)合并成一个扁平的迭代空间
>
> `schedule(static)`
>
> * 指定如何将迭代分配给线程
>
> * `static`调度方式：
>   * 将迭代空间平均分配给每个线程
>   * 每个线程获得连续的迭代块
>   * 分配在执行开始前就确定
>   * 适用于工作负载均匀的情况
>
> ```c++
> // 假设有3层循环，维度分别是 5x4x3
> #pragma omp parallel for collapse(3) schedule(static)
> for (int i = 0; i < 5; i++)
>     for (int j = 0; j < 4; j++)
>         for (int k = 0; k < 3; k++) {
>             // 总迭代次数 = 5 x 4 x 3 = 60 次
>             array[i][j][k] = compute();
>         }
> 
> ```
>
> 如果系统有 4 个线程，则：
> 1. 60 次迭代会被平均分配
> 2. 每个线程获得 15 次连续迭代
> 3. 分配方式大致如下：
>    - 线程 0：迭代 0-14
>    - 线程 1：迭代 15-29
>    - 线程 2：迭代 30-44
>    - 线程 3：迭代 45-59
>
> 除了 `static` 外，还有其他调度方式：
> - `schedule(dynamic)`: 动态分配，适合负载不均匀的情况
> - `schedule(guided)`: 动态分配，但块大小逐渐减小
> - `schedule(auto)`: 让系统自动选择最佳调度方式

### 1.3 Android编译指令

```shell
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_NATIVE_API_LEVEL=27 \
      -DANDROID=ON \
      -DOpenCV_DIR=../opencv_android_sdk/sdk/native/jni/abi-arm64-v8a \
      ..
```



### 1.4 算子模块

#### 1. `Conv`算子 - 非padding

<img src="./images/image-20241105161738102.png" alt="image-20241105161738102" style="zoom:50%;" />

> 1. **input_size**：
>    - 输入特征图的尺寸（可以是高度或宽度）。
> 2. **pooling_size**：
>    - 池化窗口的尺寸（可以是高度或宽度）。池化窗口是在输入特征图上滑动的一个小区域，用于计算最大值或平均值。
> 3. **stride**：
>    - 步长，指池化窗口在输入特征图上滑动时每次移动的距离。步长越大，输出特征图的尺寸越小，因为窗口覆盖的区域会更少。
> 4. **floor**：
>    - 表示向下取整操作。由于池化窗口可能无法整齐地覆盖输入特征图的所有位置，向下取整确保输出尺寸为整数。

#### 2. `Conv`算子 - 带padding

<img src="./images/image-20241105162502567.png" alt="image-20241105162502567" style="zoom:50%;" />

> 1. **input_size**：
>    - 输入特征图的尺寸（可以是高度或宽度）。
> 2. **padding**：
>    - 填充的大小，指在输入特征图的边缘添加的像素数。填充通常用于防止池化窗口在边缘处无法完全覆盖输入特征图。
> 3. **pooling_size**：
>    - 池化窗口的尺寸（可以是高度或宽度）。池化窗口在输入特征图上滑动，用于计算最大值或平均值。
> 4. **stride**：
>    - 步长，指池化窗口在输入特征图上滑动时每次移动的距离。步长越大，输出特征图的尺寸越小。
> 5. **floor**：
>    - 表示向下取整操作。因为池化窗口可能无法整齐地覆盖输入特征图的所有位置，向下取整确保输出尺寸为整数。

### 3. 代码形象解析

```c++
/**
         * 卷积计算转为: 矩阵计算 (现有较为成熟的矩阵)
         * 将一个nxn的矩阵转为 1 x (nxn)的行向量
         * @param input: 输入特征图像
         * @param kernel_w: 卷积核宽度
         * @param kernel_h: 卷积核高度
         * @param input_w: 输入特征的宽度
         * @param input_h: 输入特征的高度
         * @param input_c_group: 每个group处理的通道数量
         * @param group: 当前Im2Col的组数(Group)
         * @param row_len: 卷积核展开后的列数
         * @param col_len: 卷积计算的次数
         * @return
         */ 
arma::fmat ConvolutionLayer::Im2Col(BatmanInfer::sftensor input,
                                        uint32_t kernel_w,
                                        uint32_t kernel_h,
                                        uint32_t input_w,
                                        uint32_t input_h,
                                        uint32_t input_c_group,
                                        uint32_t group,
                                        uint32_t row_len,
                                        uint32_t col_len)
```

**输入特征**

> - **输入特征图尺寸**：$5 \times 5$
> - **卷积核尺寸**：$ 3 \times 3 $
> - **步长**：1
> - **填充**：1
> - **输入通道数**：4（每组）
> - **分组数**：2

```C++
arma::fmat input_matrix(input_c_group * row_len, col_len)
```

> * `input_c_group`:  每个组的输入通道数，这个例子是4
> * `row_len`:  对于$3 \times 3$ 的`卷积核`. `row_len`通常为$ 3 \times 3 = 9$
> * `col_len`:  展开矩阵的列数。这个值取决于`滑动窗口`在输入特征图上移动的次数，$7 \times 7$ 输入和$ 3 \times 3$ 的卷积核, 步长为1, 可以移动$ 5 \times 5 = 25$
>
> **具体操作:**
>
> * `input_matrix`: 一个大小为$(4 \times 9) \times 25 = 36 \times 25$的矩阵
>   * **列数**: $4 \times 9 = 36$, 因为每个通道展开为9行，每组有4个通道
>   * **行数**: 25  (`因为arma的特点`)

```C++
const uint32_t input_padded_h = input_h + 2 * padding_h_;
const uint32_t input_padded_w = input_w + 2 * padding_w_;
```

> * **`input_padded_h` 和 `input_padded_w`**：计算填充后的输入特征图尺寸。对于 $ 5 \times 5$的输入，加上 1 的填充，变为 $ 7 \times 7$

```c++
for (uint32_t ic = 0; ic < input_c_group; ++ic) 
  float* input_channel_ptr = input->matrix_raw_ptr(ic + group * input_c_group);
```

> * `ic`: 遍历当前组内的每个通道
> * `input_channel_ptr`: 指向当前通道的数据起始位置

```c++
uint32_t current_col = 0;
uint32_t channel_row = ic * row_len;
```

> * `current_col`: 当前列索引，用于在展开矩阵中定位
> * `channel_row`: 计算当前通道在展开矩阵中的起始行位置

```c++
for (uint32_t w = 0; w < input_padded_w - kernel_w + 1; w += stride_w_) {
    for (uint32_t r = 0; r < input_padded_h - kernel_h + 1; r += stride_h_) {
        float* input_matrix_ptr = input_matrix.colptr(current_col) + channel_row;
        current_col += 1;
    }
}
```

> - **外层循环**：遍历输入图像的宽度方向。
> - **内层循环**：遍历输入图像的高度方向。
> - **`input_matrix_ptr`**：指向 `input_matrix` 中当前列位置的指针，用于存放从 `input_channel_ptr` 复制过来的数据。
> - **`current_col`**：每次迭代递增，表示移动到下一个列位置。

#### 2. GlobalAveragePool (全局平均池化)

> **作用**:  对输入张量的每个通道的所有元素的平均值来减少"数据的维度"
>
> **数学解释**:
>
> 假设我们有一个**输入张量** $X$ 的形状为 $[N,C,H,W]$，其中：
>
> - $N$ 是批次大小（batch size）。
> - $C$ 是通道数（channels）。
> - $H$ 是高度（height）。
> - $W$ 是宽度（width）。
>
> 输出张量$Y$ 的形状$[N, C, 1, 1]$
>
> **数学公式**:
>
> $ Y_{n, c, 0, 0} = \frac{1}{H \times W} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} X_{n, c, h, w} $
>
> **数学例子**:
>
> 假设我们有一个简单的输入张量 $ X $ ，其形状为 $[1, 1, 2, 2]$，即一个批次、一个通道、高度为2、宽度为2。具体的张量值为：
>
> $  X = \begin{bmatrix}  \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \end{bmatrix}   $
>
> 对于这个输入张量，GlobalAveragePool的计算过程如下：
>
> 1. 计算通道内所有元素的和：
>
>    $ \text{Sum} = 1 + 2 + 3 + 4 = 10 $
>
> 2. 计算平均值：
>
>    $$ \text{Average} = \frac{\text{Sum}}{H \times W} = \frac{10}{2 \times 2} = \frac{10}{4} = 2.5 $$
>
> 因此，输出张量 $ Y $ 为：
>
> $ Y = \begin{bmatrix} \begin{bmatrix} 2.5 \end{bmatrix} \end{bmatrix} $
>
> **应用场景**
>
> \- **降维**：GlobalAveragePool通过减少空间维度从 $[H, W]$ 到 $[1, 1]$）来降低模型的参数量和计算复杂度。
>
> \- **特征提取**：保留整个特征图的全局信息，有助于在分类任务中利用全局特征。
>
> \- **替代全连接层**：在某些网络架构中，全局平均池化可以替代全连接层，减少参数数量并防止过拟合。
>
> 这种算子在卷积神经网络（CNN）中非常常见，尤其是在现代深度学习模型的最后几层中，用于生成固定大小的特征向量

#### 3. Flatten 算子

> **作用**:
>
> * **输入**: 一个具有任意维度的张量
> * **输出**: 一个二维张量 [`batch_size`, `seq_length`]
>
> **参数**:
>
> * `axis`: 这是一个整数参数, 指定从哪个轴开始展平。默认情况下, `axis = 1`, 这意味着输入张量的第一个维度（通常是批量大小）保持不变，其余的维度被展平成一个维度。
>
> **实例**
>
> 假设有一个输入张量 `X`，其形状为 $[N, C, H, W]$，其中 $N$ 是批量大小，$C$ 是通道数，$H$ 和 $W$ 是高度和宽度。
>
> - 如果 `axis=1`，Flatten 操作将把输入张量展平成形状为 $[N, C \times H \times W]$ 的二维张量。
> - 如果 `axis=2`，Flatten 操作将把输入张量展平成形状为 $[N \times C, H \times W]$的二维张量。
>
> **应用场景**
>
> Flatten 算子通常用于卷积神经网络（CNN）中，在卷积和池化层之后，将特征图展平，以便连接到全连接层。这是因为全连接层通常需要一维输入

#### 4. Gemm算子 - 通用矩阵乘法

> 公式:
>
> $Y = \alpha \times (A \times B) + \beta \times C$
>
> **1. Attributes(属性)**
>
> * **alpha**: (缩放因子), 默认为1, 用于缩放矩阵乘积 $ A \times B $
> * **beta**: (缩放因子), 默认为1。 用于缩放矩阵$C$
> * **transB**: 如果为1，表示矩阵$B$需要转置
>
> **2. Inputs(输入)**
>
> * **A**: 输入矩阵
> * **B**: 权重矩阵
> * **C**: 偏置矩阵
>
> **3.例子**
>
> - $ A $ 是一个 2x3 的矩阵：
>   $A = \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6\end{bmatrix}$
>   
> - $ B $ 是一个 3x2 的矩阵：
>   $B = \begin{bmatrix}7 & 8 \\9 & 10 \\11 & 12\end{bmatrix}$
>   
> - $ C $ 是一个 2x2 的矩阵：
>  $ C = \begin{bmatrix} 1 & 1 \\1 & 1\end{bmatrix}$
> 
> 根据Gemm的计算公式：
> 
> 1. 计算 $ A \times B $：
>    $A \times B = \begin{bmatrix}1 \times 7 + 2 \times 9 + 3 \times 11 & 1\times 8 + 2 \times 10 + 3 \times 12 \\4 \times 7 + 5 \times 9 + 6 \times 11 &4\times 8 + 5 \times 10 + 6 \times 12\end{bmatrix} = \begin{bmatrix}58 & 64 \\139 & 154\end{bmatrix}$
>    
> 2. 计算最终输出 $ Y = \alpha \times (A \times B) + \beta \times C $：
>    $Y = 1 \times \begin{bmatrix}58 & 64 \\139 & 154\end{bmatrix} + 1 \times \begin{bmatrix}1 & 1 \\1 & 1\end{bmatrix} = \begin{bmatrix}59 & 65 \\140 & 155\end{bmatrix}$

#### 5. Concat (Concatenate) 算子 - 合并张量

> **作用**:
>
> 在 ONNX 中，`Concat`（Concatenate）算子用于将多个张量沿指定的轴拼接（连接）起来。它的主要作用是将输入的张量合并为一个更大的张量。
>
> 1. **输入**：
>    - `Concat` 接收多个张量作为输入。这些张量必须在非拼接轴上的形状相同。
>    - `inputs` 包含两个输入张量 `/attention/Slice_1_output_0` 和 `/attention/Slice_2_output_0`。
>
> 2. **属性**：
>    - **`axis`**：指定沿哪个轴进行拼接。  
>      在你的截图中，`axis=0`，表示沿第 0 维（通常是 batch 维度）拼接。
>
> 3. **输出**：
>    - 输出是一个新的张量，其形状在拼接轴上是输入张量的形状之和，而其他轴的形状保持不变。
>    - 在你的截图中，输出张量是 `/attention/Concat_output_0`。
>
> **示例**
>
> 假设有两个输入张量：
>
> - $ \text{Tensor A} $ 形状为 $[2, 3]$:
>  $
>   \mathbf{A} = \begin{bmatrix}
>   1 & 2 & 3 \\
>   4 & 5 & 6
>   \end{bmatrix}
>   $
> 
> - $ \text{Tensor B} $ 形状为 $[3, 3]$:
>   $
>   \mathbf{B} = \begin{bmatrix}
>    7 & 8 & 9 \\
>   10 & 11 & 12 \\
>   13 & 14 & 15
>   \end{bmatrix}
>   $
> 
> 如果沿着 $ \text{axis} = 0 $ 进行拼接，结果为：
> 
> $
>\text{Concat}(\mathbf{A}, \mathbf{B}) = \begin{bmatrix}
> 1 & 2 & 3 \\
>4 & 5 & 6 \\
> 7 & 8 & 9 \\
> 10 & 11 & 12 \\
> 13 & 14 & 15
> \end{bmatrix}
> $
> 
> 输出的形状为 $[5, 3]$。

#### 6. Trilu 算子

##### 在本例中的具体表示

假设：
- $ \text{condition} $ 是布尔张量，表示是否满足某种条件。
- $ X = -\infty $（负无穷）
- $ Y = 0 $

则输出 $ \text{output} $ 的每个元素可以表示为：

$
\text{output}[i] =
\begin{cases}
-\infty, & \text{if } \text{condition}[i] = \text{True} \\
0, & \text{if } \text{condition}[i] = \text{False}
\end{cases}
$

##### 应用场景

`Where`算子常用于深度学习模型中实现条件逻辑操作，例如：

\- 掩码操作：将某些位置的值设为无穷大或零。

\- 动态选择：基于条件选择不同的输入。

----


#### 7. Where 算子 - 条件选择算子

##### 定义

给定：
- 条件张量 $ \text{condition} $
- 输入张量 $ X $
- 输入张量 $ Y $

`Where`算子的输出张量 $ \text{output} $ 可以表示为：

$
\text{output}[i] =
\begin{cases}
X[i], & \text{if } \text{condition}[i] \text{ is True} \\
Y[i], & \text{if } \text{condition}[i] \text{ is False}
\end{cases}
$

##### 在本例中的具体表示

假设：
- $ \text{condition} $ 是布尔张量，表示是否满足某种条件。
- $ X = -\infty $（负无穷）
- $ Y = 0 $

则输出 $ \text{output} $ 的每个元素可以表示为：

$
\text{output}[i] =
\begin{cases}
-\infty, & \text{if } \text{condition}[i] = \text{True} \\
0, & \text{if } \text{condition}[i] = \text{False}
\end{cases}
$

##### 应用场景

`Where`算子常用于深度学习模型中实现条件逻辑操作，例如：

\- 掩码操作：将某些位置的值设为无穷大或零。

\- 动态选择：基于条件选择不同的输入。

----

#### 8. Cast 算子 - 转换算子

> **作用**:
>
> 将输入数据从一种数据类型转换为另一种数据类型

#### 9. Sqrt算子 - 平方根算子

> **作用**:
>
> 计算输入张量中每个元素的平方根。它是一个简单的逐元素操作，支持广播机制
>
> 1. **类型**: `Sqrt`
>    - 该算子属于 ONNX 标准算子集合，版本是 `ai.onnx v13`。
>
> 2. **功能**: 
>    - 计算输入张量中每个元素的平方根，输出一个具有相同形状的张量。
>    - 数学公式为：  
>      $ Y = \sqrt{X} $
>      其中 $ X $ 是输入张量，$ Y $ 是输出张量。
>
> 3. **输入**:
>    - **`X`**:  
>      - 数据类型：支持 `float16`, `float`, `double` 等浮点类型。
>      - 形状：任意形状的张量。
>
> 4. **输出**:
>    - **`Y`**:  
>      - 数据类型：与输入 `X` 的数据类型相同。
>      - 形状：与输入 `X` 的形状相同。
>
> 5. **属性**:
>    - `Sqrt` 算子没有额外的属性，它是一个纯粹的数学操作。
>
> 6. **广播机制**:
>    - 如果输入张量需要广播（例如，将标量与张量相结合），ONNX 会自动处理广播规则。

#### 10. Div算子 - 逐元素除法操作

> **作用**:
>
> \- `Div` 算子的作用是将输入张量 `A` 和 `B` 中的每个元素逐一相除，并输出结果到张量 `C` 中。
>
> \- 数学公式为：
>
>   $
>
>   C[i] = A[i] \div B[i]
>
>   $
>
>   其中 $ A[i] $ 和 $ B[i] $ 是输入张量 `A` 和 `B` 中的对应元素。

#### 11. MatMul 算子 - 矩乘计算

> **作用**: 
>
> MatMul（Matrix Multiplication）是 ONNX（Open Neural Network Exchange）中用于执行矩阵乘法的算子。它接受两个输入张量 `A` 和 `B`，返回它们的乘积 `Y`。在深度学习和线性代数中，矩阵乘法是基本且重要的操作，广泛应用于神经网络的计算中。
>
> **操作**:
>
> - **输入**：
>   - `A`：任意形状的张量。
>   - `B`：任意形状的张量。
> - **输出**：
>   - `Y`：矩阵乘法的结果张量。
>
> **矩阵乘法规则**：
>
> - 对于二维矩阵，`A` 的列数必须等于 `B` 的行数。
> - 对于高维张量，MatMul 会将最后两个维度视为矩阵，其余维度需要匹配或者可以通过广播机制进行匹配。
>
> ---
>
> **示例**
>
> 假设有两个二维矩阵：
>
> **矩阵 A（2x3）****：
>
> $ A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ \end{bmatrix}$
>
> **矩阵 B（3x2）**：
>
> $ B = \begin{bmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \\ \end{bmatrix} $
>
> **计算结果矩阵 Y（2x2）****：
>
> $Y = A \times B$

#### 12. Mul算子 - 逐元素相乘

> **作用**:
>
> 用于执行 **逐元素相乘**（element-wise multiplication）操作。它接收两个张量作为输入，并返回它们逐元素相乘后的结果
> - 输入：
  - $ A =
  \begin{bmatrix}
  1 & 2 \\
  3 & 4
  \end{bmatrix} $
  - $ B =
  \begin{bmatrix}
  5 & 6 \\
  7 & 8
  \end{bmatrix} $
- 输出：
  - $ C =
  \begin{bmatrix}
  1 \times 5 & 2 \times 6 \\
  3 \times 7 & 4 \times 8
  \end{bmatrix} =
  \begin{bmatrix}
  5 & 12 \\
  21 & 32
  \end{bmatrix} $

---

#### 13. Transpose 算子  - 重排列
1. **`perm`（Permutation）**
   - `perm` 是 `Transpose` 的核心属性，用来指定维度的排列顺序。
   - 它是一个整数列表，表示输入张量维度的新顺序。
   - 例如：
     - 输入张量形状为 `(2, 3, 4)`。
     - 如果 `perm = [1, 2, 0]`，输出张量的形状会变成 `(3, 4, 2)`，即将第 0 维移动到最后。

2. **`data`（输入张量）**
   - 这是 `Transpose` 操作的输入张量，其形状和数据类型由前一层的输出决定。

3. **`transposed`（输出张量）**
   - 这是 `Transpose` 操作的输出张量，其形状由输入张量的形状和 `perm` 参数共同决定。

---
**原始张量**

张量 $ x $ 的形状是 $[1, 2, 3, 4]$，其内容为：

$
x = \begin{bmatrix}
  \begin{bmatrix}
    \begin{bmatrix}
      1 & 2 & 3 & 4 \\
      5 & 6 & 7 & 8 \\
      9 & 10 & 11 & 12
    \end{bmatrix} \\
    \begin{bmatrix}
      13 & 14 & 15 & 16 \\
      17 & 18 & 19 & 20 \\
      21 & 22 & 23 & 24
    \end{bmatrix}
  \end{bmatrix}
\end{bmatrix}
$

**维度变换**

执行 `x.permute(0, 2, 1, 3)`，意味着我们要将维度从 $[1, 2, 3, 4]$ 变为 $[1, 3, 2, 4]$。具体步骤如下：

1. **保持第一个维度不变**：仍然是大小为 1。
2. **将第三个维度移到第二个位置**：原来大小为 3 的维度，现在移到第二个位置。
3. **将第二个维度移到第三个位置**：原来大小为 2 的维度，现在移到第三个位置。
4. **保持第四个维度不变**：大小为 4。

**重排后的张量**

根据上述变换，张量内容变为：

$
x_{\text{permuted}} = \begin{bmatrix}
  \begin{bmatrix}
    \begin{bmatrix}
      1 & 2 & 3 & 4 \\
      13 & 14 & 15 & 16
    \end{bmatrix} \\
    \begin{bmatrix}
      5 & 6 & 7 & 8 \\
      17 & 18 & 19 & 20
    \end{bmatrix} \\
    \begin{bmatrix}
      9 & 10 & 11 & 12 \\
      21 & 22 & 23 & 24
    \end{bmatrix}
  \end{bmatrix}
\end{bmatrix}
$

**解释**

- **第一块**：原来是 $[1, 2, 3, 4]$ 中的第一行，现在变为 $[1, 3, 2, 4]$ 中的第一块。
- **第二块**：原来是 $[1, 2, 3, 4]$ 中的第二行，现在变为 $[1, 3, 2, 4]$ 中的第二块。
- **第三块**：原来是 $[1, 2, 3, 4]$ 中的第三行，现在变为 $[1, 3, 2, 4]$ 中的第三块。
----

#### 14. Reshape算子 - 改变张量形状

> **作用**
>
> 改变张量形状
>
> **举例**
> **原始形状 (1×2×768)** 
> 输入是一个三维张量，可以看作是1个batch，包含2个序列，每个序列是768维的向量：
> 
> $
> X_{1×2×768} = \begin{bmatrix}
> \begin{pmatrix}
> x_{1,1} & x_{1,2} & \cdots & x_{1,768} \\
> x_{2,1} & x_{2,2} & \cdots & x_{2,768}
> \end{pmatrix}
> \end{bmatrix}
> $
> 
> **Reshape后的形状 (2×768)** 
> 经过Reshape后，变成一个二维矩阵，本质上是"展平"了batch维度：
> 
> $
> X'_{2×768} = \begin{bmatrix}
> x_{1,1} & x_{1,2} & \cdots & x_{1,768} \\
> x_{2,1} & x_{2,2} & \cdots & x_{2,768}
> \end{bmatrix}
> $
> 
> **数值示例** 
> 为了更具体，让我们看一个简化的例子（这里用更小的维度以便展示）：
> 
> 假设有一个 1×2×4 的张量：
> 
> $
> X_{1×2×4} = \begin{bmatrix}
> \begin{pmatrix}
> 1 & 2 & 3 & 4 \\
> 5 & 6 & 7 & 8
> \end{pmatrix}
> \end{bmatrix}
> $
> 
> Rshape 到 2×4 后：
>
> $
> X'_{2×4} = \begin{bmatrix}
> 1 & 2 & 3 & 4 \\
> 5 & 6 & 7 & 8
> \end{bmatrix}
> $

#### 15. Split算子 - 输入张量沿指定轴分割

> **作用**:
>
> 分割张量
>
> **示例**:
>
> - **输入张量**：形状为 `[1, 2, 2304]`。这是需要被分割的张量。
> - **分割轴 (`axis`)**：值为 `2`，意味着分割操作将在第三个维度（从0开始计数）进行。
> - **分割点 (`split`)**：`[768, 768, 768]`，表示将第三个维度分割成三个部分，每个部分的大小为768。

