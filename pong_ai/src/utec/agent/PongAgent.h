#pragma once
#include "Tensor.h"
#include "neural_network.h"
#include "activation.h"
#include "layer.h"
#include "dense.h"
#include "loss.h"
#include "EnvGym.h"

namespace utec {
    namespace agent {

        template<typename T>
        class PongAgent {
        public:
            explicit PongAgent(const nn::NeuralNetwork<T>& model)
              : model_(model) {}

            int act(const State& s) const {
                algebra::Tensor<T, 2> input(1, 3);
                input(0, 0) = static_cast<T>(s.ball_x);
                input(0, 1) = static_cast<T>(s.ball_y);
                input(0, 2) = static_cast<T>(s.paddle_y);
                auto out = model_.forward(input);
                T v = out(0, 0);
                if (v > T(0.5)) return +1;
                if (v < T(-0.5)) return -1;
                return 0;
            }

        private:
            const nn::NeuralNetwork<T>& model_;
        };

    } // namespace agent
} // namespace utec