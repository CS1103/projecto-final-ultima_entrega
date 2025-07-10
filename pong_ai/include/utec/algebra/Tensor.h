#pragma once
#include <iostream>
#include <exception>
#include <initializer_list>
#include <numeric>
#include <utility>
#include <functional>
#include <algorithm>

namespace utec {
namespace algebra {

class TensorError : public std::exception {
    const char* mensaje;
public:
    TensorError(const char* m) : mensaje(m) {}
    const char* what() const noexcept override { return mensaje; }
};

template <typename T, unsigned long N>
class Tensor {
public:

    Tensor() {
        for (unsigned long i = 0; i < N; ++i)
            dimensiones[i] = 0;
        datos = nullptr;
        capacidad = 0;
    }

    template <typename... Args>
    Tensor(Args... dims) {
        constexpr unsigned long num = sizeof...(Args);
        if (num != N) {
            std::cerr << "[ERROR] Constructor: esperado N=" << N << ", recibido=" << num << "\n";
            throw TensorError("Number of dimensions do not match");
        }
        unsigned long vals[num] = { static_cast<unsigned long>(dims)... };
        unsigned long total = 1;
        for (unsigned long i = 0; i < N; ++i) {
            dimensiones[i] = vals[i];
            total *= vals[i];
        }
        datos = new T[total];
        capacidad = total;
        for (unsigned long i = 0; i < total; ++i) {
            datos[i] = T();
        }
    }

    void print_shape(const std::string& nombre = "Tensor") const {
        std::cout << nombre << " shape: (";
        for (unsigned long i = 0; i < N; ++i) {
            std::cout << dimensiones[i];
            if (i + 1 < N) std::cout << ", ";
        }
        std::cout << ")\n";
    }

    Tensor(const Tensor& other) {
        unsigned long total = other.tamano_total();
        for (unsigned long i = 0; i < N; ++i)
            dimensiones[i] = other.dimensiones[i];
        datos = new T[total];
        capacidad = total;
        for (unsigned long i = 0; i < total; ++i)
            datos[i] = other.datos[i];
    }

    Tensor(Tensor&& other) noexcept {
        for (unsigned long i = 0; i < N; ++i)
            dimensiones[i] = other.dimensiones[i];
        datos = other.datos;
        capacidad = other.capacidad;
        other.datos = nullptr;
        other.capacidad = 0;
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            unsigned long total = other.tamano_total();
            delete[] datos;
            for (unsigned long i = 0; i < N; ++i)
                dimensiones[i] = other.dimensiones[i];
            datos = new T[total];
            capacidad = total;
            for (unsigned long i = 0; i < total; ++i)
                datos[i] = other.datos[i];
        }
        return *this;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            delete[] datos;
            for (unsigned long i = 0; i < N; ++i)
                dimensiones[i] = other.dimensiones[i];
            datos = other.datos;
            capacidad = other.capacidad;
            other.datos = nullptr;
            other.capacidad = 0;
        }
        return *this;
    }

    ~Tensor() {
        delete[] datos;
    }

    unsigned long* shape() const {
        return const_cast<unsigned long*>(dimensiones);
    }

    void fill(const T& valor) {
        unsigned long total = tamano_total();
        for (unsigned long i = 0; i < total; ++i)
            datos[i] = valor;
    }

    Tensor<T, N>& operator=(std::initializer_list<T> lista) {
        if (lista.size() != tamano_total())
            throw TensorError("Data size does not match tensor size");
        unsigned long i = 0;
        for (auto& v : lista)
            datos[i++] = v;
        return *this;
    }

    template <typename... Args>
    void reshape(Args... dims) {
        constexpr unsigned long num = sizeof...(Args);
        if (num != N) {
            std::cerr << "[ERROR] reshape(): esperado N=" << N << ", recibido=" << num << "\n";
            throw TensorError("Number of dimensions do not match");
        }
        unsigned long nuevos[num] = { static_cast<unsigned long>(dims)... };
        unsigned long nuevo_total = 1;
        for (unsigned long i = 0; i < N; ++i)
            nuevo_total *= nuevos[i];
        unsigned long viejo_total = tamano_total();
        if (nuevo_total > capacidad) {
            T* nuevo_buffer = new T[nuevo_total];
            for (unsigned long i = 0; i < viejo_total; ++i)
                nuevo_buffer[i] = datos[i];
            for (unsigned long i = viejo_total; i < nuevo_total; ++i)
                nuevo_buffer[i] = T();
            delete[] datos;
            datos = nuevo_buffer;
            capacidad = nuevo_total;
        }
        for (unsigned long i = 0; i < N; ++i)
            dimensiones[i] = nuevos[i];
    }

    template <typename... Args>
    T& operator()(Args... idxs) {
        constexpr unsigned long num = sizeof...(Args);
        if (datos == nullptr)
            throw TensorError("Tensor not initialized (nullptr)");
        if (num != N) {
            throw TensorError("Number of dimensions do not match");
        }
        unsigned long indices[num] = { static_cast<unsigned long>(idxs)... };
        unsigned long lin = 0;
        for (unsigned long d = 0; d < N; ++d) {
            unsigned long stride = 1;
            for (unsigned long k = d + 1; k < N; ++k)
                stride *= dimensiones[k];
            lin += indices[d] * stride;
        }
        return datos[lin];
    }

    template <typename... Args>
    const T& operator()(Args... idxs) const {
        constexpr unsigned long num = sizeof...(Args);
        if (num != N) {
            throw TensorError("Number of dimensions do not match");
        }
        unsigned long indices[num] = { static_cast<unsigned long>(idxs)... };
        unsigned long lin = 0;
        for (unsigned long d = 0; d < N; ++d) {
            unsigned long stride = 1;
            for (unsigned long k = d + 1; k < N; ++k)
                stride *= dimensiones[k];
            lin += indices[d] * stride;
        }
        return datos[lin];
    }

    T* begin() { return datos; }
    T* end()   { return datos + tamano_total(); }
    const T* begin() const { return datos; }
    const T* end()   const { return datos + tamano_total(); }
    const T* cbegin() const { return datos; }
    const T* cend()   const { return datos + tamano_total(); }

    friend Tensor<T, N> operator+(const Tensor<T, N>& a, const T& s) {
        Tensor<T, N> r = a;
        unsigned long total = a.tamano_total();
        for (unsigned long i = 0; i < total; ++i)
            r.datos[i] = a.datos[i] + s;
        return r;
    }
    friend Tensor<T, N> operator+(const T& s, const Tensor<T, N>& a) { return a + s; }

    friend Tensor<T, N> operator-(const Tensor<T, N>& a, const T& s) {
        Tensor<T, N> r = a;
        unsigned long total = a.tamano_total();
        for (unsigned long i = 0; i < total; ++i)
            r.datos[i] = a.datos[i] - s;
        return r;
    }
    friend Tensor<T, N> operator-(const T& s, const Tensor<T, N>& a) {
        Tensor<T, N> r = a;
        unsigned long total = a.tamano_total();
        for (unsigned long i = 0; i < total; ++i)
            r.datos[i] = s - a.datos[i];
        return r;
    }

    friend Tensor<T, N> operator*(const Tensor<T, N>& a, const T& s) {
        Tensor<T, N> r = a;
        unsigned long total = a.tamano_total();
        for (unsigned long i = 0; i < total; ++i)
            r.datos[i] = a.datos[i] * s;
        return r;
    }
    friend Tensor<T, N> operator*(const T& s, const Tensor<T, N>& a) { return a * s; }

    friend Tensor<T, N> operator/(const Tensor<T, N>& a, const T& s) {
        Tensor<T, N> r = a;
        unsigned long total = a.tamano_total();
        for (unsigned long i = 0; i < total; ++i)
            r.datos[i] = a.datos[i] / s;
        return r;
    }

    friend Tensor<T, N> operator+(const Tensor<T, N>& a, const Tensor<T, N>& b) {
        if (a.tamano_total() != b.tamano_total())
            throw TensorError("Tensor sizes do not match for element-wise operation");
        Tensor<T, N> r = a;
        unsigned long total = a.tamano_total();
        for (unsigned long i = 0; i < total; ++i)
            r.datos[i] = a.datos[i] + b.datos[i];
        return r;
    }

    friend Tensor<T, N> operator-(const Tensor<T, N>& a, const Tensor<T, N>& b) {
        if (a.tamano_total() != b.tamano_total())
            throw TensorError("Tensor sizes do not match for element-wise operation");
        Tensor<T, N> r = a;
        unsigned long total = a.tamano_total();
        for (unsigned long i = 0; i < total; ++i)
            r.datos[i] = a.datos[i] - b.datos[i];
        return r;
    }

    friend Tensor<T, N> operator*(const Tensor<T, N>& a, const Tensor<T, N>& b) {
        if (a.tamano_total() != b.tamano_total())
            throw TensorError("Tensor sizes do not match for element-wise operation");
        Tensor<T, N> r = a;
        unsigned long total = a.tamano_total();
        for (unsigned long i = 0; i < total; ++i)
            r.datos[i] = a.datos[i] * b.datos[i];
        return r;
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor<T, N>& t) {
        if constexpr (N == 1) {
            unsigned long total = t.tamano_total();
            for (unsigned long i = 0; i < total; ++i) {
                os << t.datos[i];
                if (i + 1 < total) os << " ";
            }
        }
        else if constexpr (N == 2) {
            unsigned long D0 = t.dimensiones[0];
            unsigned long D1 = t.dimensiones[1];
            os << "{\n";
            for (unsigned long i = 0; i < D0; ++i) {
                for (unsigned long j = 0; j < D1; ++j) {
                    unsigned long idx = i * D1 + j;
                    os << t.datos[idx];
                    if (j + 1 < D1) os << " ";
                }
                os << "\n";
            }
            os << "}";
        }
        return os;
    }

    unsigned long tamano_total() const {
        unsigned long p = 1;
        for (unsigned long i = 0; i < N; ++i)
            p *= dimensiones[i];
        return p;
    }

    void calcular_strides(const unsigned long dims[N], unsigned long strides[N]) const {
        strides[N - 1] = 1;
        for (int i = int(N) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
    }

    template<typename Func>
    Tensor<T, N> apply(Func f) const {
        Tensor<T, N> result = *this;
        unsigned long total = tamano_total();
        for (unsigned long i = 0; i < total; ++i) {
            result.datos[i] = f(datos[i]);
        }
        return result;
    }

    Tensor<T, N> transpose_2d() const {
        if constexpr (N != 2) {
            throw TensorError("transpose_2d only works for 2D tensors");
        }
        Tensor<T, N> result(dimensiones[1], dimensiones[0]);
        for (unsigned long i = 0; i < dimensiones[0]; ++i) {
            for (unsigned long j = 0; j < dimensiones[1]; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

private:
    T* datos;
    unsigned long dimensiones[N];
    unsigned long capacidad;
};

template <typename T>
Tensor<T, 2> matrix_product(const Tensor<T, 2>& A, const Tensor<T, 2>& B) {
    unsigned long* a_dims = A.shape();
    unsigned long* b_dims = B.shape();
    unsigned long M = a_dims[0], K = a_dims[1];
    unsigned long K2 = b_dims[0], N = b_dims[1];
    if (K != K2) {
        throw TensorError("Matrix dimensions are incompatible for multiplication");
    }
    Tensor<T, 2> R(M, N);
    for (unsigned long i = 0; i < M; ++i) {
        for (unsigned long j = 0; j < N; ++j) {
            T sum = T();
            for (unsigned long k = 0; k < K; ++k) {
                sum += A(i, k) * B(k, j);
            }
            R(i, j) = sum;
        }
    }
    return R;
}

template<typename T>
using Tensor2 = Tensor<T, 2>;

} // namespace algebra
} // namespace utec