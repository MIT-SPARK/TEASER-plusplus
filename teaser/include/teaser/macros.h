/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <iostream>

#define TEASER_DEBUG_ERROR_VAR(x)                                                                    \
  do {                                                                                             \
    std::cerr << #x << " = " << x << std::endl;                                                    \
  } while (0)

#if defined(NDEBUG) && !defined(TEASER_DIAG_PRINT)
#define TEASER_DEBUG_ERROR_MSG(x)                                                                  \
  do {                                                                                             \
  } while (0)
#define TEASER_DEBUG_INFO_MSG(x)                                                                   \
  do {                                                                                             \
  } while (0)
#else
#define TEASER_DEBUG_ERROR_MSG(x)                                                                  \
  do {                                                                                             \
    std::cerr << x << std::endl;                                                                   \
  } while (0)
#define TEASER_DEBUG_INFO_MSG(x)                                                                   \
  do {                                                                                             \
    std::cout << x << std::endl;                                                                   \
  } while (0)
#endif