#pragma once
#if (defined __ARM_NEON__ || defined __aarch64__)
#pragma message("-----------neon----------")
#include <arm_neon.h>
#else
#pragma message("-----------sse----------")
#include "neon/NEON_2_SSE.h"
#endif
