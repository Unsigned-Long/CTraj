// CTraj: Continuous-Time Trajectory (Time-Varying State) Representation and Estimation Library
// Copyright 2024, the School of Geodesy and Geomatics (SGG), Wuhan University, China
// https://github.com/Unsigned-Long/CTraj.git
//
// Author: Shuolong Chen (shlchen@whu.edu.cn)
// GitHub: https://github.com/Unsigned-Long
//  ORCID: 0000-0002-5283-9057
//
// Purpose: See .h/.hpp file.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * The names of its contributors can not be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef CTRAJ_ASSERT_H
#define CTRAJ_ASSERT_H

#include <iostream>

namespace ns_ctraj {

#define UNUSED(x) (void)(x)

inline void AssertionFailed(char const *expr, char const *function, char const *file, long line) {
    std::cerr << "[*] Assertion (" << expr << ") failed in " << function << ": 'file': " << file
              << ", 'line': " << line << std::endl;
    std::abort();
}

inline void AssertionFailedMsg(
    char const *expr, char const *msg, char const *function, char const *file, long line) {
    std::cerr << "[*] Assertion (" << expr << ") failed in " << function << ": ['file': " << file
              << ", 'line': " << line << "，'msg': " << msg << std::endl;
    std::abort();
}
}  // namespace ns_ctraj

#define BASALT_LIKELY(x) __builtin_expect(x, 1)

#if defined(BASALT_DISABLE_ASSERTS)

    #define CTRAJ_ASSERT(expr) ((void)0)

    #define CTRAJ_ASSERT_MSG(expr, msg) ((void)0)

    #define CTRAJ_ASSERT_STREAM(expr, msg) ((void)0)

#else

    #define CTRAJ_ASSERT(expr)   \
        (BASALT_LIKELY(!!(expr)) \
             ? ((void)0)         \
             : ::ns_ctraj::AssertionFailed(#expr, __PRETTY_FUNCTION__, __FILE__, __LINE__))

    #define CTRAJ_ASSERT_MSG(expr, msg)                                                            \
        (BASALT_LIKELY(!!(expr)) ? ((void)0)                                                       \
                                 : ::ns_ctraj::AssertionFailedMsg(#expr, msg, __PRETTY_FUNCTION__, \
                                                                  __FILE__, __LINE__))

    #define CTRAJ_ASSERT_STREAM(expr, msg)     \
        (BASALT_LIKELY(!!(expr))               \
             ? ((void)0)                       \
             : (std::cerr << msg << std::endl, \
                ::ns_ctraj::AssertionFailed(#expr, __PRETTY_FUNCTION__, __FILE__, __LINE__)))

#endif

#endif