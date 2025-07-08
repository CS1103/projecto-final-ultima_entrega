#include "EnvGym.h"
#include "utec/algebra/Tensor.h"
#include "utec/nn/neural_network.h"

namespace utec::agent {

template<typename T>
class PongAgent {
public:
    /**
     * El agente recibe una red ya entrenada que
     * mapea [1×3] → [1×1], donde las 3 entradas son
     * {ball_x, ball_y, paddle_y} y la salida es un valor
     * que interpretamos como movimiento:
     *   salida >  0.5 → +1
     *   salida < -0.5 → -1
     *   en otro caso   →  0
     */
    explicit PongAgent(const utec::nn::NeuralNetwork<T> &model)
      : model_(model) {}

    int act(const State &s) const {
        // Construir tensor [1×3]
        utec::algebra::Tensor<T,2> input(1, 3);
        input(0,0) = static_cast<T>(s.ball_x);
        input(0,1) = static_cast<T>(s.ball_y);
        input(0,2) = static_cast<T>(s.paddle_y);
        // Forward
        auto out = model_.forward(input);  // [1×1]
        T v = out(0,0);
        if (v > T(0.5))   return +1;
        if (v < T(-0.5))  return -1;
        return 0;
    }

private:
    const utec::nn::NeuralNetwork<T> &model_;
};

} // namespace utec::agent
