#pragma once
#include <numeric>
#include "Tensor.h"
#include "layer.h"

namespace utec {
    namespace nn {

        template<typename T>
        class MSELoss {
        public:
            T forward(const Tensor2<T>& pred, const Tensor2<T>& target) {
                last_pred = pred;
                last_target = target;
                auto diff = pred - target;
                auto sq = diff * diff;
                T sum = std::accumulate(sq.begin(), sq.end(), T(0));
                return sum / T(pred.shape()[0]);
            }

            Tensor2<T> backward() {
                auto diff = last_pred - last_target;
                return diff * (T(2) / T(last_pred.shape()[0]));
            }

        private:
            Tensor2<T> last_pred = Tensor2<T>(1, 1);
            Tensor2<T> last_target = Tensor2<T>(1, 1);
        };

    } // namespace nn
} // namespace utec