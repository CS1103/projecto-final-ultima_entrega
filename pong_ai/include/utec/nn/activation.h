#pragma once
#include "Tensor.h"
#include "layer.h"

namespace utec {
    namespace nn {

        template<typename T>
        class ReLU : public ILayer<T> {
        public:
            Tensor2<T> forward(const Tensor2<T>& x) override {
                mask = x;
                return x.apply([](T v) { return v > T(0) ? v : T(0); });
            }

            Tensor2<T> backward(const Tensor2<T>& grad) override {
                auto result = grad;
                for (size_t i = 0; i < result.shape()[0]; ++i) {
                    for (size_t j = 0; j < result.shape()[1]; ++j) {
                        if (mask(i, j) <= T(0)) {
                            result(i, j) = T(0);
                        }
                    }
                }
                return result;
            }

        private:
            Tensor2<T> mask;
        };

    } // namespace nn
} // namespace utec