//
// Created by Mason on 2025/1/3.
//

#include <data/core/bi_i_kernel.hpp>

using namespace BatmanInfer;

const BIWindow &BIIKernel::window() const {
    return _window;
}

BIIKernel::BIIKernel() : _window()
{
    // 创建一个空窗口，以确保子类自己设置窗口值。
    _window.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 1));
    _window.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 1));
}

bool BIIKernel::is_parallelisable() const {
    return true;
}

BIBorderSize BIIKernel::border_size() const {
    return BIBorderSize(0);
}

bool BIIKernel::is_window_configured() const {
    return !((_window.x().start() == _window.x().end()) && (_window.x().end() == 0));
}

void BIIKernel::configure(const BatmanInfer::BIWindow &window) {
    _window = window;
}