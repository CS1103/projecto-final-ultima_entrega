#include "layer.h"
#include <random>

namespace utec::neural_network {

template<typename T>
class Dense : public ILayer<T> {
public:
    Dense(size_t in_features, size_t out_features)
      : W(in_features, out_features),
        b(1, out_features),
        dW(in_features, out_features),
        db(1, out_features)
    {
        // Inicialización simples: valores pequeños aleatorios
        std::mt19937 gen(42);
        std::normal_distribution<T> dist(0.0, 0.1);
        W = Tensor2<T>(in_features, out_features);
        for (auto& v : W) v = dist(gen);
        b.fill(0);
    }

    Tensor2<T> forward(const Tensor2<T>& input) override {
        X = input;  // cache para backward
        // output = X · W + b (broadcast en filas)
        Tensor2<T> Y = matrix_product(X, W) + b;
        return Y;
    }

    Tensor2<T> backward(const Tensor2<T>& grad_output) override {
        // grad_output: [batch, out_features]
        // dW = Xᵀ · grad_output
        Tensor2<T> XT = X.transpose_2d();
        dW = matrix_product(XT, grad_output);
        // db = suma a lo largo del batch
        db = Tensor2<T>(1, grad_output.shape()[1]);
        for (size_t j = 0; j < grad_output.shape()[1]; ++j) {
            T sum = 0;
            for (size_t i = 0; i < grad_output.shape()[0]; ++i)
                sum += grad_output(i,j);
            db(0,j) = sum;
        }
        // dX = grad_output · Wᵀ
        Tensor2<T> WT = W.transpose_2d();
        Tensor2<T> dX = matrix_product(grad_output, WT);
        return dX;
    }

    void optimize(T lr) {
        // W ← W − lr·dW, b ← b − lr·db
        W = W - dW * lr;
        b = b - db * lr;
    }

private:
    Tensor2<T> W, b;       // parámetros
    Tensor2<T> X;          // cache de entrada
public:
    Tensor2<T> dW, db;     // gradientes públicos para inspección
};

} // namespace utec::neural_network
