#include <iostream>
#include <chrono>
#include "Tensor.h"
#include "layer.h"
#include "neural_network.h"
#include "dense.h"
#include "loss.h"
#include "EnvGym.h"
#include "PongAgent.h"
#include "activation.h"

using namespace utec;

int main() {
    try {
        // Epic 1: Tensor Demo
        std::cout << "=== Epic 1: Tensor Demo ===\n";
        algebra::Tensor<int, 2> t(2, 3);
        t.fill(5);
        std::cout << "Tensor 2x3 lleno de 5:\n" << t << "\n";
        t.reshape(3, 2);
        std::cout << "Tras reshape a 3x2:\n" << t << "\n\n";

        // Epic 2: Neural Network XOR
        std::cout << "=== Epic 2: Neural Network XOR ===\n";

        algebra::Tensor<float, 2> X(4, 2);
        X = {0, 0, 0, 1, 1, 0, 1, 1};  // 4 filas, 2 columnas
        X.print_shape("X");

        algebra::Tensor<float, 2> Y(4, 1);
        Y = {0, 1, 1, 0};  // 4 filas, 1 columna
        Y.print_shape("Y");

        std::cout << std::endl;

        std::cout << "Inicializando red neuronal XOR...\n";
        utec::nn::NeuralNetwork<float> net;
        net.add_layer(std::make_unique<utec::nn::Dense<float>>(2, 4));
        net.add_layer(std::make_unique<utec::nn::ReLU<float>>());
        net.add_layer(std::make_unique<utec::nn::Dense<float>>(4, 1));

        std::cout << "Entrenando...\n";
        auto start = std::chrono::high_resolution_clock::now();
        net.train(X, Y, 2000, 0.1f);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Entrenamiento finalizado.\n";
        std::cout << "Tiempo de entrenamiento: " << diff.count() << " segundos\n";

        auto preds = net.forward(X);
        int correct = 0;
        for (int i = 0; i < 4; ++i) {
            float y_pred = preds(i, 0);
            float y_true = Y(i, 0);
            if ((y_pred > 0.5 && y_true == 1.0f) || (y_pred <= 0.5 && y_true == 0.0f))
                correct++;
        }
        float acc = 100.0f * correct / 4;
        std::cout << "Precision: " << acc << "%\n";

        // Epic 3: PongAgent Demo
        std::cout << std::endl;
        std::cout << "=== Epic 3: PongAgent Demo ===\n";

        utec::nn::NeuralNetwork<float> pong_model;
        pong_model.add_layer(std::make_unique<utec::nn::Dense<float>>(3, 4));
        pong_model.add_layer(std::make_unique<utec::nn::ReLU<float>>());
        pong_model.add_layer(std::make_unique<utec::nn::Dense<float>>(4, 1));

        struct MockEnv : agent::EnvGym {
            agent::State current_state{0.5f, 0.3f, 0.2f};

            agent::State reset() override {
                current_state = {0.5f, 0.3f, 0.2f};
                return current_state;
            }

            agent::State step(int action, float &reward, bool &done) override {
                current_state.paddle_y += action * 0.1f;
                reward = (action != 0) ? 1.0f : 0.0f;
                done = false;
                return current_state;
            }
        };

        agent::PongAgent<float> pong_agent(pong_model);
        MockEnv env;

        auto state = env.reset();
        std::cout << "Estado inicial - Ball: (" << state.ball_x << ", " << state.ball_y
                  << "), Paddle: " << state.paddle_y << "\n";

        float reward;
        bool done;
        int action = pong_agent.act(state);
        state = env.step(action, reward, done);

        std::cout << "Agente eligio accion: " << action
                  << ", reward: " << reward << "\n";
        std::cout << "Nuevo estado - Paddle: " << state.paddle_y << "\n";

        std::cout << "\n=== Todo funcionando correctamente ===\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
