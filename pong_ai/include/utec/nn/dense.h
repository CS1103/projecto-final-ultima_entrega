#pragma once
#include <random>
#include "Tensor.h"
#include "layer.h"

namespace utec {
    namespace nn {

        template<typename T>
        class Dense : public ILayer<T> {
        public:
            Dense(size_t in_features, size_t out_features)
              : W(in_features, out_features),
                b(1, out_features),
                dW(in_features, out_features),
                db(1, out_features)
            {
                std::mt19937 gen(42);
                std::normal_distribution<T> dist(0.0, 0.1);
                for (auto& v : W) v = dist(gen);
                b.fill(0);
            }

            Tensor2<T> forward(const Tensor2<T>& input) override {
                X = input;
                auto Y = algebra::matrix_product(X, W);
                // Broadcasting para sumar bias
                for (size_t i = 0; i < Y.shape()[0]; ++i) {
                    for (size_t j = 0; j < Y.shape()[1]; ++j) {
                        Y(i, j) += b(0, j);
                    }
                }
                return Y;
            }

            Tensor2<T> backward(const Tensor2<T>& grad_output) override {
                auto XT = X.transpose_2d();
                dW = algebra::matrix_product(XT, grad_output);

                for (size_t j = 0; j < grad_output.shape()[1]; ++j) {
                    T sum = 0;
                    for (size_t i = 0; i < grad_output.shape()[0]; ++i)
                        sum += grad_output(i, j);
                    db(0, j) = sum;
                }

                auto WT = W.transpose_2d();
                return algebra::matrix_product(grad_output, WT);
            }

            void optimize(T lr) {
                for (size_t i = 0; i < W.shape()[0]; ++i) {
                    for (size_t j = 0; j < W.shape()[1]; ++j) {
                        W(i, j) -= lr * dW(i, j);
                    }
                }
                for (size_t j = 0; j < b.shape()[1]; ++j) {
                    b(0, j) -= lr * db(0, j);
                }
            }

        private:
            Tensor2<T> W, b;
            Tensor2<T> X;
            Tensor2<T> dW, db;
        };

    } // namespace nn
} // namespace utec
