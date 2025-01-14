//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BCL_VERSION_HPP
#define BATMANINFER_BCL_VERSION_HPP

// 常用于 C 和 C++ 混合编程中。它的作用是确保在 C++ 环境下，C 代码可以被正确地链接和调用
// 检查当前代码是否在 C++ 编译器下编译
#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

/**
 * 语义化版本信息
 */
typedef struct BclVersion {
    int major;      /**< 主版本号，在 API 不兼容的更改时增加 */
    int minor;      /**< 次版本号，在添加向后兼容的功能时增加 */
    int patch;      /**< 修订版本号，在进行向后兼容的修复时增加 */
    const char *build_info; /**< 构建相关的信息 */
} BclVersion;

/**< 主版本号，在 API 不兼容的更改时增加 */
#define BI_COMPUTE_LIBRARY_VERSION_MAJOR 0
/**< 次版本号，在添加向后兼容的功能时增加 */
#define BI_COMPUTE_LIBRARY_VERSION_MINOR 1
/**< 修订版本号，在进行向后兼容的修复时增加 */
#define BI_COMPUTE_LIBRARY_VERSION_PATCH 0


/**
 * 获取库的版本元数据
 * @return
 */
const BclVersion *BclVersionInfo();

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif //BATMANINFER_BCL_VERSION_HPP
