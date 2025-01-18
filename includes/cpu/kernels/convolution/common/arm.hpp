//
// Created by Mason on 2025/1/18.
//

#pragma once

/** Sets the macro __arm_any__ if compiling for Aarch32 or Aarch64.
 *  Includes `arm_neon.h` if compiling for either architecture.
 */

#ifdef __arm__
#define __arm_any__
#endif  // __arm__

#ifdef __aarch64__
#define __arm_any__
#endif  // __aarch64__

#ifdef __arm_any__

#include <arm_neon.h>

#endif  // __arm_any__