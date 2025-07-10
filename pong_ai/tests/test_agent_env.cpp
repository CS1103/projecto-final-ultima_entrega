#include <cassert>
#include "PongAgent.h"
#include "EnvGym.h"
#include "dense.h"
#include "activation.h"

using namespace utec::agent;
using namespace utec::nn;

class MockEnv : public EnvGym {
public:
    State current_state{0.5f, 0.3f, 0.2f};

    State reset() override {
        current_state = {0.5f, 0.3f, 0.2f};
        return current_state;
    }

    State step(int action, float& reward, bool& done) override {
        current_state.paddle_y += action * 0.1f;
        reward = (action != 0) ? 1.0f : 0.0f;
        done = false;
        return current_state;
    }
};

void test_agent_decision() {
    NeuralNetwork<float> model;
    model.add_layer(std::make_unique<Dense<float>>(3, 4));
    model.add_layer(std::make_unique<ReLU<float>>());
    model.add_layer(std::make_unique<Dense<float>>(4, 1));

    PongAgent<float> agent(model);
    State s{0.5f, 0.3f, 0.2f};
    int action = agent.act(s);
    assert(action >= -1 && action <= 1);
    std::cout << "test_agent_decision passed\n";
}

int main() {
    test_agent_decision();
    return 0;
}