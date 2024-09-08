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

#include "ctraj/core/imu.h"
#include "utility"

namespace ns_ctraj {

IMUFrame::IMUFrame(double timestamp, Eigen::Vector3d gyro, Eigen::Vector3d acce)
    : _timestamp(timestamp),
      _gyro(std::move(gyro)),
      _acce(std::move(acce)) {}

IMUFrame::Ptr IMUFrame::Create(double timestamp,
                               const Eigen::Vector3d &gyro,
                               const Eigen::Vector3d &acce) {
    return std::make_shared<IMUFrame>(timestamp, gyro, acce);
}

double IMUFrame::GetTimestamp() const { return _timestamp; }

const Eigen::Vector3d &IMUFrame::GetGyro() const { return _gyro; }

const Eigen::Vector3d &IMUFrame::GetAcce() const { return _acce; }

std::ostream &operator<<(std::ostream &os, const IMUFrame &frame) {
    os << "timestamp: " << frame._timestamp << ", gyro: " << frame._gyro.transpose()
       << ", acce: " << frame._acce.transpose();
    return os;
}

void IMUFrame::SetTimestamp(double timestamp) { _timestamp = timestamp; }

bool IMUFrame::SaveFramesToDisk(const std::string &filename,
                                const std::vector<IMUFrame::Ptr> &frames,
                                int precision) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(precision);
    cereal::JSONOutputArchive ar(file);
    ar(cereal::make_nvp("imu_frames", frames));
    return true;
}

}  // namespace ns_ctraj
