// Copyright (c) 2019-2023. Created on 7/7/23 1:20 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#pragma once

#include <iostream>

namespace ns_ctraj {

#define UNUSED(x) (void)(x)

    inline void AssertionFailed(char const *expr, char const *function, char const *file, long line) {
        std::cerr << "[*] Assertion (" << expr << ") failed in " << function << ": 'file': " << file << ", 'line': "
                  << line << std::endl;
        std::abort();
    }

    inline void
    AssertionFailedMsg(char const *expr, char const *msg, char const *function, char const *file, long line) {
        std::cerr << "[*] Assertion (" << expr << ") failed in " << function << ": ['file': " << file << ", 'line': "
                  << line << "，'msg': " << msg << std::endl;
        std::abort();
    }
}  // namespace ns_ctraj

#define BASALT_LIKELY(x) __builtin_expect(x, 1)

#if defined(BASALT_DISABLE_ASSERTS)

#define CTRAJ_ASSERT(expr) ((void)0)

#define CTRAJ_ASSERT_MSG(expr, msg) ((void)0)

#define CTRAJ_ASSERT_STREAM(expr, msg) ((void)0)

#else


#define CTRAJ_ASSERT(expr)                                                 \
  (BASALT_LIKELY(!!(expr))                                                 \
       ? ((void)0)                                                         \
       : ::ns_ctraj::AssertionFailed(#expr, __PRETTY_FUNCTION__, __FILE__, \
                                     __LINE__))

#define CTRAJ_ASSERT_MSG(expr, msg)                                      \
  (BASALT_LIKELY(!!(expr))                                               \
       ? ((void)0)                                                       \
       : ::ns_ctraj::AssertionFailedMsg(#expr, msg, __PRETTY_FUNCTION__, \
                                        __FILE__, __LINE__))

#define CTRAJ_ASSERT_STREAM(expr, msg)                                      \
  (BASALT_LIKELY(!!(expr))                                                  \
       ? ((void)0)                                                          \
       : (std::cerr << msg << std::endl,                                    \
          ::ns_ctraj::AssertionFailed(#expr, __PRETTY_FUNCTION__, __FILE__, \
                                      __LINE__)))

#endif
