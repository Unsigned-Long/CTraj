// Copyright (c) 2023. Created on 10/18/23 7:45 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#ifndef CTRAJ_SPLINE_BUNDLE_H
#define CTRAJ_SPLINE_BUNDLE_H

#include <utility>
#include <ostream>
#include "ctraj/spline/rd_spline.h"
#include "ctraj/spline/so3_spline.h"
#include "cereal/types/map.hpp"
#include "ctraj/utils/sophus_utils.hpp"
#include "ctraj/spline/spline_segment.h"

namespace ns_ctraj {
    // b-splines that live on the vector space (Rd) or manifold of Lie Group (So3)
    enum class SplineType {
        RdSpline, So3Spline
    };

    struct SplineInfo {
        /**
         * @brief name: the name of spline, as a search key for access in spline bundle
         * @brief type: the type of spline: Rd or So3
         * @brief st: the start timestamp
         * @brief et: the end timestamp
         * @brief dt: the time distance of the uniform spline
         */
        std::string name;
        SplineType type;
        double st, et, dt;

        SplineInfo(std::string name, SplineType type, double st, double et, double dt)
                : name(std::move(name)), type(type), st(st), et(et), dt(dt) {}
    };

    template<int Order>
    class SplineBundle {
    public:
        static constexpr int N = Order;
        using Ptr = std::shared_ptr<SplineBundle>;
        using RdSplineType = RdSpline<3, Order, double>;
        using So3SplineType = So3Spline<Order, double>;
        // Eigen::Vector3d
        using RdSplineKnotType = typename RdSplineType::VecD;
        // Sophus::SO3d
        using So3SplineKnotType = typename So3SplineType::SO3;
        using SplineMetaType = ns_ctraj::SplineMeta<Order>;

    private:
        /**
         * @brief _so3Splines: container for so3 splines
         * @brief _rdSplines: container for rd splines
         */
        std::map<std::string, So3SplineType> _so3Splines;
        std::map<std::string, RdSplineType> _rdSplines;

    public:
        explicit SplineBundle(const std::vector<SplineInfo> &splines) {
            for (const auto &spline: splines) { AddSpline(spline); }
        }

        /**
         * @brief add a spline to the bundle based on its type
         */
        SplineBundle &AddSpline(const SplineInfo &spline) {
            switch (spline.type) {
                case SplineType::RdSpline: {
                    _rdSplines.insert({spline.name, RdSplineType(spline.dt, spline.st)});
                    ExtendKnotsTo(_rdSplines.at(spline.name), spline.et, RdSplineKnotType::Zero());
                }
                    break;
                case SplineType::So3Spline: {
                    _so3Splines.insert({spline.name, So3SplineType(spline.dt, spline.st)});
                    ExtendKnotsTo(_so3Splines.at(spline.name), spline.et, So3SplineKnotType());
                }
                    break;
            }
            return *this;
        }

        static Ptr Create(const std::vector<SplineInfo> &splines) {
            return std::make_shared<SplineBundle>(splines);
        }

        void Save(const std::string &filename) const {
            std::ofstream file(filename);
            cereal::JSONOutputArchive ar(file);
            ar(cereal::make_nvp("SplineBundle", *this));
        }

        static SplineBundle::Ptr Load(const std::string &filename) {
            auto bundle = SplineBundle::Create({});
            std::ifstream file(filename);
            cereal::JSONInputArchive ar(file);
            ar(cereal::make_nvp("SplineBundle", *bundle));
            return bundle;
        }

        So3SplineType &GetSo3Spline(const std::string &name) {
            return _so3Splines.at(name);
        }

        RdSplineType &GetRdSpline(const std::string &name) {
            return _rdSplines.at(name);
        }

        [[nodiscard]] bool TimeInRangeForSo3(double time, const std::string &name) const {
            // left closed right open interval
            auto &spline = _so3Splines.at(name);
            return time >= spline.MinTime() + 1E-6 && time < spline.MaxTime() - 1E-6;
        }

        [[nodiscard]] bool TimeInRangeForRd(double time, const std::string &name) const {
            // left closed right open interval
            auto &spline = _rdSplines.at(name);
            return time >= spline.MinTime() + 1E-6 && time < spline.MaxTime() - 1E-6;
        }

        template<class SplineType>
        bool TimeInRange(double time, const SplineType &spline) const {
            // left closed right open interval
            return time >= spline.MinTime() + 1E-6 && time < spline.MaxTime() - 1E-6;
        }

        friend ostream &operator<<(ostream &os, const SplineBundle &bundle) {
            os << "SplineBundle:\n";
            for (const auto &[name, spline]: bundle._so3Splines) {
                os << "'name': " << name << ", 'SplineType::So3Spline', ['st': " << spline.MinTime()
                   << ", 'et': " << spline.MaxTime() << ", 'dt': " << spline.GetTimeInterval() << "]\n";
            }
            for (const auto &[name, spline]: bundle._rdSplines) {
                os << "'name': " << name << ", 'SplineType::RdSpline', ['st': " << spline.MinTime()
                   << ", 'et': " << spline.MaxTime() << ", 'dt': " << spline.GetTimeInterval() << "]\n";
            }
            return os;
        }

        void CalculateSo3SplineMeta(const std::string &name, time_init_t times, SplineMetaType &splineMeta) const {
            CalculateSplineMeta(_so3Splines.at(name), times, splineMeta);
        }

        void CalculateRdSplineMeta(const std::string &name, time_init_t times, SplineMetaType &splineMeta) const {
            CalculateSplineMeta(_rdSplines.at(name), times, splineMeta);
        }

        /**
         * @brief extent the spline to support state access at time t
         * @param init: the initial state of newly-added knots
         */
        template<class SplineType, class KnotType>
        static void ExtendKnotsTo(SplineType &spline, double t, const KnotType &init) {
            while ((spline.GetKnots().size() < N) || (spline.MaxTime() < t)) {
                spline.KnotsPushBack(init);
            }
        }

        /**
         * @brief compute the spline meta for the time piece sequence (@param times)
         * @param times: a sequence of time piece
         */
        template<class SplineType>
        static void CalculateSplineMeta(const SplineType &spline, time_init_t times, SplineMetaType &splineMeta) {
            double master_dt = spline.GetTimeInterval();
            double master_t0 = spline.MinTime();
            size_t current_segment_start = 0;
            size_t current_segment_end = 0; // Negative signals no segment created yet

            // Times are guaranteed to be sorted correctly and t2 >= t1
            for (auto tt: times) {
                std::pair<double, size_t> ui_1, ui_2;
                ui_1 = spline.ComputeTIndex(tt.first);
                ui_2 = spline.ComputeTIndex(tt.second);

                size_t i1 = ui_1.second;
                size_t i2 = ui_2.second;

                // Create new segment, or extend the current one
                if (splineMeta.segments.empty() || i1 > current_segment_end) {
                    double segment_t0 = master_t0 + master_dt * double(i1);
                    splineMeta.segments.emplace_back(segment_t0, master_dt);
                    current_segment_start = i1;
                } else {
                    i1 = current_segment_end + 1;
                }

                auto &current_segment_meta = splineMeta.segments.back();

                for (size_t i = i1; i < (i2 + N); ++i) {
                    current_segment_meta.n += 1;
                }

                current_segment_end = current_segment_start + current_segment_meta.n - 1;
            } // for times
        }

    public:
        template<class Archive>
        void serialize(Archive &ar) {
            ar(cereal::make_nvp("So3Splines", _so3Splines),
               cereal::make_nvp("RdSplines", _rdSplines));
        }
    };
}

#endif //CTRAJ_SPLINE_BUNDLE_H
