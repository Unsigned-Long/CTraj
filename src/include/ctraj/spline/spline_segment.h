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

#ifndef CTRAJ_SPLINE_SEGMENT_H
#define CTRAJ_SPLINE_SEGMENT_H

#include <Eigen/Dense>
#include <cstdint>
#include <ceres/jet.h>
#include <iomanip>

namespace ns_ctraj {

// Define time types
using time_span_t = std::pair<double, double>;
using time_init_t = std::initializer_list<time_span_t>;

struct MetaData {
    [[nodiscard]] virtual size_t NumParameters() const = 0;
};

template <int Order>
struct SplineSegmentMeta : public MetaData {
    static constexpr int N = Order;        // Order of the spline.
    static constexpr int DEG = Order - 1;  // Degree of the spline.

    double t0;  // First valid time
    double dt;  // Knot spacing
    size_t n;   // Number of knots

    SplineSegmentMeta(double _t0, double _dt, size_t _n = 0)
        : t0(_t0),
          dt(_dt),
          n(_n) {}

    [[nodiscard]] size_t NumParameters() const override { return n; }

    [[nodiscard]] double MinTime() const { return t0; }

    [[nodiscard]] double MaxTime() const { return t0 + (n - DEG) * dt; }

    template <typename T>
    size_t PotentiallyUnsafeFloor(T x) const {
        return static_cast<size_t>(std::floor(x));
    }

    // This way of treating Jets are potentially unsafe, hence the function name
    template <typename Scalar, int N>
    size_t PotentiallyUnsafeFloor(const ceres::Jet<Scalar, N> &x) const {
        return static_cast<size_t>(ceres::floor(x.a));
    };

    template <typename T>
    bool ComputeTIndex(const T &timestamp, T &u, size_t &s) const {
        T t = timestamp;
        if (timestamp >= T(MaxTime()))
            t = timestamp - T(1E-6);  // 1us
        else if (timestamp < T(MinTime()))
            t = timestamp + T(1E-6);

        if (t >= T(MinTime()) && t < T(MaxTime())) {
            T st = (t - T(t0)) / T(dt);
            s = PotentiallyUnsafeFloor(st);  // Take integer part
            u = st - T(s);                   // Take Decimal part
            return true;
        } else {
            return false;
        }
    }
};

template <int N>
struct SplineMeta {
    std::vector<SplineSegmentMeta<N>> segments;

    [[nodiscard]] size_t NumParameters() const {
        size_t n = 0;
        for (auto &segment_meta : segments) {
            n += segment_meta.NumParameters();
        }
        return n;
    }

    template <typename T>
    bool ComputeSplineIndex(const T &timestamp, size_t &idx, T &u) const {
        idx = 0;
        for (auto const &seg : segments) {
            size_t s = 0;
            if (seg.ComputeTIndex(timestamp, u, s)) {
                idx += s;
                return true;
            } else {
                idx += seg.NumParameters();
            }
        }
        std::cout << std::fixed << std::setprecision(15) << "[ComputeSplineIndex] t: " << timestamp
                  << std::endl;
        std::cout << " not in [" << segments[0].t0 << ", " << segments[0].MaxTime() << "]"
                  << std::endl;

        assert(timestamp >= segments[0].t0 && timestamp < segments[0].MaxTime() &&
               "[ComputeSplineIndex] not in range");
        return false;
    }
};

}  // namespace ns_ctraj
#endif