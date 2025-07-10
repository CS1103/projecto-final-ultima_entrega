#include <cassert>
#include "neural_network.h"
#include "dense.h"
#include "activation.h"
#include "loss.h"

using namespace utec::nn;

void test_neural_network_xor() {
    Tensor2<float> X(4, 2);
    X = {0, 0, 0, 1, 1, 0, 1, 1};
    Tensor2<float> Y(4, 1);
    Y = {0, 1, 1, 0};

    NeuralNetwork<float> net;
    net.add_layer(std::make_unique<Dense<float>>(2, 4));
    net.add_layer(std::make_unique<ReLU<float>>());
    net.add_layer(std::make_unique<Dense<float>>(4, 1));

    net.train(X, Y, 2000, 0.1f);

    auto preds = net.forward(X);
    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        float y_pred = preds(i, 0);
        float y_true = Y(i, 0);
        if ((y_pred > 0.5 && y_true == 1.0f) || (y_pred <= 0.5 && y_true == 0.0f))
            correct++;
    }
    assert(correct == 4);
    std::cout << "test_neural_network_xor passed\n";
}