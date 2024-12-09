//
// Created by Mason on 2024/12/9.
//

#ifndef BATMANINFER_MARCO_HPP
#define BATMANINFER_MARCO_HPP

// 这个宏的作用是将一个数 x 向上对齐到 y 的倍数
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
// 这个宏是 ROUND_UP 的一个特化版本，用于将 x 向上对齐到 4 的倍数
#define ALIGN_UP4(x) ROUND_UP((x), 4)

#endif //BATMANINFER_MARCO_HPP
