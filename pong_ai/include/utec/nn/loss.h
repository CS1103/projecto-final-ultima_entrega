#include "utec/algebra/Tensor.h"

namespace utec::nn {

template<typename T>
class MSELoss {
public:
    /// Calcula la pérdida media sobre el batch
    T forward(const Tensor2<T>& pred, const Tensor2<T>& target) {
        last_pred = pred;
        last_target = target;
        auto diff = pred - target;              // [batch,features]
        auto sq   = diff * diff;                // elemento a elemento
        T sum = std::accumulate(sq.begin(), sq.end(), T(0));
        return sum / T(pred.shape()[0]);
    }

    /// Devuelve dL/dPred (mismo tamaño que pred)
    Tensor2<T> backward() {
        // gradiente de MSE: 2*(pred - target)/batch_size
        auto diff = last_pred - last_target;
        return diff * (T(2) / T(last_pred.shape()[0]));
    }

private:
    Tensor2<T> last_pred, last_target;  // caches para backward
};

} // namespace utec::nn
