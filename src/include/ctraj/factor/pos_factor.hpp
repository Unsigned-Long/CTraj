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

#ifndef CTRAJ_POS_FACTOR_HPP
#define CTRAJ_POS_FACTOR_HPP

#include "ctraj/utils/eigen_utils.hpp"

namespace ns_ctraj {
template <int Order>
struct PO3Functor {
private:
    SplineMeta<Order> _splineMeta;
    Posed ItoG{};

    double _dtInv;
    double _posWeight;

public:
    explicit PO3Functor(SplineMeta<Order> splineMeta, Posed ItoG, double posWeight)
        : _splineMeta(std::move(splineMeta)),
          ItoG(std::move(ItoG)),
          _dtInv(1.0 / _splineMeta.segments.front().dt),
          _posWeight(posWeight) {}

    static auto Create(const SplineMeta<Order> &splineMeta, const Posed &ItoG, double posWeight) {
        return new ceres::DynamicAutoDiffCostFunction<PO3Functor>(
            new PO3Functor(splineMeta, ItoG, posWeight));
    }

    static std::size_t TypeHashCode() { return typeid(PO3Functor).hash_code(); }

public:
    /**
     * param blocks:
     * [ POS | ... | POS ]
     */
    template <class T>
    bool operator()(T const *const *sKnots, T *sResiduals) const {
        // array offset
        std::size_t POS_OFFSET;
        double u;
        _splineMeta.template ComputeSplineIndex(ItoG.timeStamp, POS_OFFSET, u);

        Eigen::Vector3<T> predPOSTtoG;
        // if 'DERIV = 0' returns value of the spline, otherwise corresponding derivative.
        CeresSplineHelper<Order>::template Evaluate<T, 3, 0>(sKnots + POS_OFFSET, u, _dtInv,
                                                             &predPOSTtoG);

        Eigen::Map<Eigen::Vector3<T>> poseResiduals(sResiduals + 0);

        poseResiduals = predPOSTtoG - ItoG.t.template cast<T>();

        poseResiduals = T(_posWeight) * poseResiduals;

        return true;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ns_ctraj

#endif  // CTRAJ_POS_FACTOR_HPP
