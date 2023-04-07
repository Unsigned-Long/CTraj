#pragma once

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

    template<int Order>
    struct SplineSegmentMeta : public MetaData {
        static constexpr int N = Order;        // Order of the spline.
        static constexpr int DEG = Order - 1;  // Degree of the spline.

        double t0; // First valid time
        double dt; // Knot spacing
        size_t n;  // Number of knots

        SplineSegmentMeta(double _t0, double _dt, size_t _n = 0)
                : t0(_t0), dt(_dt), n(_n) {}

        [[nodiscard]] size_t NumParameters() const override {
            return n;
        }

        [[nodiscard]] double MinTime() const {
            return t0;
        }

        [[nodiscard]] double MaxTime() const {
            return t0 + (n - DEG) * dt;
        }

        template<typename T>
        size_t PotentiallyUnsafeFloor(T x) const {
            return static_cast<size_t>(std::floor(x));
        }

        // This way of treating Jets are potentially unsafe, hence the function name
        template<typename Scalar, int N>
        size_t PotentiallyUnsafeFloor(const ceres::Jet <Scalar, N> &x) const {
            return static_cast<size_t>(ceres::floor(x.a));
        };

        template<typename T>
        bool ComputeTIndex(const T &timestamp, T &u, size_t &s) const {
            T t = timestamp;
            if (timestamp >= T(MaxTime()))
                t = timestamp - T(1e-9);  // 1ns
            else if (timestamp < T(MinTime()))
                t = timestamp + T(1e-9);

            if (t >= T(MinTime()) && t < T(MaxTime())) {
                T st = (t - T(t0)) / T(dt);
                s = PotentiallyUnsafeFloor(st); // Take integer part
                u = st - T(s); // Take Decimal part
                return true;
            } else {
                return false;
            }
        }
    };

    template<int N>
    struct SplineMeta {
        std::vector<SplineSegmentMeta<N>> segments;

        [[nodiscard]] size_t NumParameters() const {
            size_t n = 0;
            for (auto &segment_meta: segments) {
                n += segment_meta.NumParameters();
            }
            return n;
        }

        template<typename T>
        bool ComputeSplineIndex(const T &timestamp, size_t &idx, T &u) const {
            idx = 0;
            for (auto const &seg: segments) {
                size_t s = 0;
                if (seg.ComputeTIndex(timestamp, u, s)) {
                    idx += s;
                    return true;
                } else {
                    idx += seg.NumParameters();
                }
            }
            std::cout << std::fixed << std::setprecision(15)
                      << "[ComputeSplineIndex] t: " << timestamp << std::endl;
            std::cout << " not in [" << segments[0].t0 << ", " << segments[0].MaxTime()
                      << "]" << std::endl;

            assert(timestamp >= segments[0].t0 && timestamp < segments[0].MaxTime() &&
                   "[ComputeSplineIndex] not in range");
            return false;
        }
    };

}  // namespace ns_ctraj
