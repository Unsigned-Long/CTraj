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

#ifndef CTRAJ_SO3_FACTOR_HPP
#define CTRAJ_SO3_FACTOR_HPP

#include "ctraj/utils/eigen_utils.hpp"

namespace ns_ctraj {
template <int Order>
struct SO3Functor {
private:
    SplineMeta<Order> _splineMeta;
    Posed ItoG{};

    double _dtInv;
    double _so3Weight;

public:
    explicit SO3Functor(SplineMeta<Order> splineMeta, Posed ItoG, double so3Weight)
        : _splineMeta(std::move(splineMeta)),
          ItoG(std::move(ItoG)),
          _dtInv(1.0 / _splineMeta.segments.front().dt),
          _so3Weight(so3Weight) {}

    static auto Create(const SplineMeta<Order> &splineMeta, const Posed &ItoG, double so3Weight) {
        return new ceres::DynamicAutoDiffCostFunction<SO3Functor>(
            new SO3Functor(splineMeta, ItoG, so3Weight));
    }

    static std::size_t TypeHashCode() { return typeid(SO3Functor).hash_code(); }

public:
    /**
     * param blocks:
     * [ SO3 | ... | SO3]
     */
    template <class T>
    bool operator()(T const *const *sKnots, T *sResiduals) const {
        // array offset
        std::size_t SO3_OFFSET;
        double u;
        // generate the param: 'SO3_OFFSET', 'u'
        _splineMeta.template ComputeSplineIndex(ItoG.timeStamp, SO3_OFFSET, u);

        Sophus::SO3<T> predSO3ItoG;
        CeresSplineHelper<Order>::template EvaluateLie<T, Sophus::SO3>(sKnots + SO3_OFFSET, u,
                                                                       _dtInv, &predSO3ItoG);

        Eigen::Map<Eigen::Vector3<T>> so3Residuals(sResiduals);

        so3Residuals = (predSO3ItoG * ItoG.so3.inverse().template cast<T>()).log();

        so3Residuals = T(_so3Weight) * so3Residuals;

        return true;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace ns_ctraj

#endif  // CTRAJ_SO3_FACTOR_HPP
