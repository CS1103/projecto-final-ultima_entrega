#include "layer.h"
#include "utec/algebra/Tensor.h"

namespace utec::nn {

template<typename T>
class ReLU : public ILayer<T> {
public:
    Tensor2<T> forward(const Tensor2<T>& x) override {
        mask = x;  // guardamos dónde x>0
        for (size_t i = 0; i < mask.total_size(); ++i)
            mask[i] = (mask[i] > T(0) ? T(1) : T(0));
        // aplicamos elemento a elemento
        return apply(x, [](T v){ return v > T(0) ? v : T(0); });
    }

    Tensor2<T> backward(const Tensor2<T>& grad) override {
        // simplemente multiplicamos por la máscara
        return grad * mask;
    }

private:
    Tensor2<T> mask;  // 1 donde x>0, 0 en otro caso
};

} // namespace utec::nn
