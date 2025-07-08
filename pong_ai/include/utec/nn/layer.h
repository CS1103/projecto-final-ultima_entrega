#include "utec/algebra/Tensor.h"
#include <memory>

namespace utec::neural_network {

template<typename T>
using Tensor2 = utec::algebra::Tensor<T, 2>;

template<typename T>
class ILayer {
public:
    virtual ~ILayer() = default;
    /// Toma un batch de entradas [batch_size × in_features] y devuelve [batch_size × out_features]
    virtual Tensor2<T> forward(const Tensor2<T>& input) = 0;
    /// Recibe dL/dY ([batch_size × out_features]) y devuelve dL/dX ([batch_size × in_features])
    virtual Tensor2<T> backward(const Tensor2<T>& grad_output) = 0;
};
  
} // namespace utec::neural_network
