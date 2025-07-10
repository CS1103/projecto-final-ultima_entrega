#include <cassert>
#include "Tensor.h"

using namespace utec::algebra;

void test_tensor_basic() {
    Tensor<int, 2> t(2, 3);
    t.fill(5);
    assert(t(0, 0) == 5);
    t.reshape(3, 2);
    assert(t.shape()[0] == 3 && t.shape()[1] == 2);
    std::cout << "test_tensor_basic passed\n";
}