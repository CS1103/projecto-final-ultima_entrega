#pragma once
#include <memory>
#include "Tensor.h"

namespace utec {
    namespace nn {

        template<typename T>
        using Tensor2 = algebra::Tensor<T, 2>;

        template<typename T>
        class ILayer {
        public:
            virtual ~ILayer() = default;
            virtual Tensor2<T> forward(const Tensor2<T>& input) = 0;
            virtual Tensor2<T> backward(const Tensor2<T>& grad_output) = 0;
        };

    } // namespace nn
} // namespace utec