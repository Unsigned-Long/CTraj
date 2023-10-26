#pragma once

#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include "chrono"

namespace ns_vins {

    class Utility {
    public:
        template<typename Derived>
        static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q) {
            //printf("a: %f %f %f %f", q.w(), q.x(), q.y(), q.z());
            //Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(), -q.z());
            //printf("b: %f %f %f %f", p.w(), p.x(), p.y(), p.z());
            //return q.template w() >= (typename Derived::Scalar)(0.0) ? q : Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
            return q;
        }
    };

    class TicToc {
    public:
        TicToc() {
            tic();
        }

        void tic() {
            start = std::chrono::system_clock::now();
        }

        double toc() {
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            return elapsed_seconds.count() * 1000;
        }

    private:
        std::chrono::time_point<std::chrono::system_clock> start, end;
    };


    const int NUM_THREADS = 1;

    struct ResidualBlockInfo {
        ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function,
                          std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
                : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks),
                  drop_set(_drop_set) {}

        void Evaluate();

        ceres::CostFunction *cost_function;
        ceres::LossFunction *loss_function;
        std::vector<double *> parameter_blocks;
        std::vector<int> drop_set;

        double **raw_jacobians;
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
        Eigen::VectorXd residuals;

        int localSize(int size) {
            return size == 7 ? 6 : size;
        }
    };

    struct ThreadsStruct {
        std::vector<ResidualBlockInfo *> sub_factors;
        Eigen::MatrixXd A;
        Eigen::VectorXd b;
        std::unordered_map<long, int> parameter_block_size; //global size
        std::unordered_map<long, int> parameter_block_idx; //local size
    };

    class MarginalizationInfo {
    public:
        ~MarginalizationInfo();

        int localSize(int size) const;

        int globalSize(int size) const;

        void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);

        void preMarginalize();

        void marginalize();

        std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

        std::vector<ResidualBlockInfo *> factors;
        int m, n;
        std::unordered_map<long, int> parameter_block_size; //global size
        int sum_block_size;
        std::unordered_map<long, int> parameter_block_idx; //local size
        std::unordered_map<long, double *> parameter_block_data;

        std::vector<int> keep_block_size; //global size
        std::vector<int> keep_block_idx;  //local size
        std::vector<double *> keep_block_data;

        Eigen::MatrixXd linearized_jacobians;
        Eigen::VectorXd linearized_residuals;
        const double eps = 1e-8;

    };

    class MarginalizationFactor : public ceres::CostFunction {
    public:
        MarginalizationFactor(MarginalizationInfo *_marginalization_info);

        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

        MarginalizationInfo *marginalization_info;
    };
}
