#pragma once
#include <vector>
#include <memory>
#include "Tensor.h"
#include "layer.h"
#include "dense.h"
#include "loss.h"

namespace utec {
    namespace nn {

        template<typename T>
        class NeuralNetwork {
        public:
            using LayerPtr = std::unique_ptr<ILayer<T>>;

            void add_layer(LayerPtr l) {
                layers.push_back(std::move(l));
            }

            Tensor2<T> forward(const Tensor2<T>& X) const {
                auto out = X;
                for (auto& l : layers)
                    out = l->forward(out);
                return out;
            }

            void backward(const Tensor2<T>& Y_pred, const Tensor2<T>& Y_true) {
                T loss = criterion.forward(Y_pred, Y_true);
                auto grad = criterion.backward();
                for (auto it = layers.rbegin(); it != layers.rend(); ++it)
                    grad = (*it)->backward(grad);
            }

            void optimize(T lr) {
                for (auto& l : layers) {
                    if (auto* d = dynamic_cast<Dense<T>*>(l.get()))
                        d->optimize(lr);
                }
            }

            void train(const Tensor2<T>& X, const Tensor2<T>& Y, size_t epochs, T lr) {
                for (size_t e = 0; e < epochs; ++e) {
                    auto Y_pred = forward(X);
                    backward(Y_pred, Y);
                    optimize(lr);
                    if (e % 500 == 0) {
                        T loss = criterion.forward(Y_pred, Y);
                        std::cout << "Epoch " << e << ", Loss: " << loss << std::endl;
                    }
                }
            }

        private:
            std::vector<LayerPtr> layers;
            MSELoss<T> criterion;
        };

    } // namespace nn
} // namespace utec