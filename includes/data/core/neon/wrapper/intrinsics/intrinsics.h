/*
 * Copyright (c) 2018-2022 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef ARM_COMPUTE_WRAPPER_INTRINSICS_H
#define ARM_COMPUTE_WRAPPER_INTRINSICS_H

#include "data/core/neon/wrapper/intrinsics/abs.hpp"
#include "data/core/neon/wrapper/intrinsics/add.hpp"
#include "data/core/neon/wrapper/intrinsics/and.hpp"
#include "data/core/neon/wrapper/intrinsics/bsl.hpp"
#include "data/core/neon/wrapper/intrinsics/ceq.hpp"
#include "data/core/neon/wrapper/intrinsics/cge.hpp"
#include "data/core/neon/wrapper/intrinsics/cgt.hpp"
#include "data/core/neon/wrapper/intrinsics/cgtz.hpp"
#include "data/core/neon/wrapper/intrinsics/cle.hpp"
#include "data/core/neon/wrapper/intrinsics/clt.hpp"
#include "data/core/neon/wrapper/intrinsics/combine.hpp"
#include "data/core/neon/wrapper/intrinsics/cvt.hpp"
#include "data/core/neon/wrapper/intrinsics/div.hpp"
#include "data/core/neon/wrapper/intrinsics/dup_n.hpp"
#include "data/core/neon/wrapper/intrinsics/eor.hpp"
#include "data/core/neon/wrapper/intrinsics/erf.hpp"
#include "data/core/neon/wrapper/intrinsics/exp.hpp"
#include "data/core/neon/wrapper/intrinsics/ext.hpp"
#include "data/core/neon/wrapper/intrinsics/gethigh.hpp"
#include "data/core/neon/wrapper/intrinsics/getlane.hpp"
#include "data/core/neon/wrapper/intrinsics/getlow.hpp"
#include "data/core/neon/wrapper/intrinsics/inv.hpp"
#include "data/core/neon/wrapper/intrinsics/invsqrt.hpp"
#include "data/core/neon/wrapper/intrinsics/load.hpp"
#include "data/core/neon/wrapper/intrinsics/log.hpp"
#include "data/core/neon/wrapper/intrinsics/max.hpp"
#include "data/core/neon/wrapper/intrinsics/min.hpp"
#include "data/core/neon/wrapper/intrinsics/mla.hpp"
#include "data/core/neon/wrapper/intrinsics/movl.hpp"
#include "data/core/neon/wrapper/intrinsics/movn.hpp"
#include "data/core/neon/wrapper/intrinsics/mul.hpp"
#include "data/core/neon/wrapper/intrinsics/neg.hpp"
#include "data/core/neon/wrapper/intrinsics/not.hpp"
#include "data/core/neon/wrapper/intrinsics/orr.hpp"
#include "data/core/neon/wrapper/intrinsics/pmax.hpp"
#include "data/core/neon/wrapper/intrinsics/pmin.hpp"
#include "data/core/neon/wrapper/intrinsics/pow.hpp"
#include "data/core/neon/wrapper/intrinsics/qmov.hpp"
#include "data/core/neon/wrapper/intrinsics/qmovun.hpp"
#include "data/core/neon/wrapper/intrinsics/reinterpret.hpp"
#include "data/core/neon/wrapper/intrinsics/rev64.hpp"
#include "data/core/neon/wrapper/intrinsics/round.hpp"
#include "data/core/neon/wrapper/intrinsics/setlane.hpp"
#include "data/core/neon/wrapper/intrinsics/shr.hpp"
#include "data/core/neon/wrapper/intrinsics/sin.hpp"
#include "data/core/neon/wrapper/intrinsics/sqrt.hpp"
#include "data/core/neon/wrapper/intrinsics/store.hpp"
#include "data/core/neon/wrapper/intrinsics/sub.hpp"
#include "data/core/neon/wrapper/intrinsics/tanh.hpp"
#include "data/core/neon/wrapper/intrinsics/tbl.hpp"

#if defined(__ARM_FEATURE_SVE)
//#include "data/core/neon/wrapper/intrinsics/svcnt.hpp"
//#include "data/core/neon/wrapper/intrinsics/svcvt.hpp"
//#include "data/core/neon/wrapper/intrinsics/svdup_n.hpp"
//#include "data/core/neon/wrapper/intrinsics/svexp.hpp"
//#include "data/core/neon/wrapper/intrinsics/svlog.hpp"
//#include "data/core/neon/wrapper/intrinsics/svpow.hpp"
//#include "data/core/neon/wrapper/intrinsics/svptrue.hpp"
//#include "data/core/neon/wrapper/intrinsics/svqadd.hpp"
//#include "data/core/neon/wrapper/intrinsics/svsin.hpp"
//#include "data/core/neon/wrapper/intrinsics/svwhilelt.hpp"
#endif /* defined(__ARM_FEATURE_SVE) */

#endif /* ARM_COMPUTE_WRAPPER_INTRINSICS_H */
