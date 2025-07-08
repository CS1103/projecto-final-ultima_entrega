#include <iostream>

// Epic 1: Tensor
#include "utec/algebra/Tensor.h"

// Epic 2: Neural Network
#include "utec/nn/neural_network.h"
#include "utec/nn/dense.h"
#include "utec/nn/activation.h"
#include "utec/nn/loss.h"

// Epic 3: Agent + Mock Env
#include "utec/agent/EnvGym.h"
#include "utec/agent/PongAgent.h"

using namespace utec;

int main() {
    // -----------------------
    // Epic 1: Prueba Tensor
    // -----------------------
    std::cout << "=== Epic 1: Tensor Demo ===\n";
    algebra::Tensor<int,2> t(2,3);
    t.fill(5);
    std::cout << "Tensor 2×3 lleno de 5:\n" << t << "\n";
    t.reshape(3,2);
    std::cout << "Tras reshape a 3×2:\n" << t << "\n\n";

    // ----------------------------
    // Epic 2: Prueba Neural Network
    // ----------------------------
    std::cout << "=== Epic 2: Neural Network XOR ===\n";
    using Tensor2f = algebra::Tensor<float,2>;
    nn::NeuralNetwork<float> net;
    net.add_layer(std::make_unique<nn::Dense<float>>(2,4));
    net.add_layer(std::make_unique<nn::ReLU<float>>());
    net.add_layer(std::make_unique<nn::Dense<float>>(4,1));

    // Datos XOR
    Tensor2f X(4,2);
    X = {0,0, 0,1, 1,0, 1,1};
    Tensor2f Y(4,1);
    Y = {0,1,1,0};

    net.train(X, Y, 2000, 0.1f);
    auto Yp = net.forward(X);
    std::cout << "Predicciones XOR:\n" << Yp << "\n\n";

    // -----------------------------------
    // Epic 3: Prueba PongAgent con Mock
    // -----------------------------------
    std::cout << "=== Epic 3: PongAgent Demo ===\n";
    // Definimos un mock muy simple en línea
    struct MockEnv : agent::EnvGym {
        State reset() override {
            return State{0.0f, 0.0f, 0.0f};
        }
        State step(int action, float &reward, bool &done) override {
            // reward = acción, terminamos tras un paso arbitrario
            reward = static_cast<float>(action);
            done = true;
            return State{0.0f, 0.0f, 0.0f};
        }
    };

    // Creamos agente con la red ya entrenada de arriba
    agent::PongAgent<float> agent(nn::NeuralNetwork<float>(net));
    MockEnv env;

    auto state = env.reset();
    float reward;
    bool done;
    int action = agent.act(state);
    state = env.step(action, reward, done);
    std::cout << "Agente eligió acción: " << action
              << ", reward simulado: " << reward << "\n";

    return 0;
}

