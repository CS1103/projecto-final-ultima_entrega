#include <iostream>
#include <exception>
#include <initializer_list>
#include <numeric>
#include <utility>

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
    template <typename... Args>
    Tensor(Args... dims) {
        constexpr unsigned long num = sizeof...(Args);
        if (num != N) {
            if (N == 1) throw TensorError("Number of dimensions do not match with 1");
            if (N == 2) throw TensorError("Number of dimensions do not match with 2");
            if (N == 3) throw TensorError("Number of dimensions do not match with 3");
            if (N == 4) throw TensorError("Number of dimensions do not match with 4");
            throw TensorError("Number of dimensions do not match with");
        }
        unsigned long vals[num] = { static_cast<unsigned long>(dims)... };
        unsigned long total = 1;
        for (unsigned long i = 0; i < N; ++i) {
            dimensiones[i] = vals[i];
            total *= vals[i];
        }
        datos = new T[total];
        capacidad = total;
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
            if (N == 1) throw TensorError("Number of dimensions do not match with 1");
            if (N == 2) throw TensorError("Number of dimensions do not match with 2");
            if (N == 3) throw TensorError("Number of dimensions do not match with 3");
            if (N == 4) throw TensorError("Number of dimensions do not match with 4");
            throw TensorError("Number of dimensions do not match with");
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
        if (num != N) {
            if (N == 1) throw TensorError("Number of dimensions do not match with 1");
            if (N == 2) throw TensorError("Number of dimensions do not match with 2");
            if (N == 3) throw TensorError("Number of dimensions do not match with 3");
            if (N == 4) throw TensorError("Number of dimensions do not match with 4");
            throw TensorError("Number of dimensions do not match with");
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
            if (N == 1) throw TensorError("Number of dimensions do not match with 1");
            if (N == 2) throw TensorError("Number of dimensions do not match with 2");
            if (N == 3) throw TensorError("Number of dimensions do not match with 3");
            if (N == 4) throw TensorError("Number of dimensions do not match with 4");
            throw TensorError("Number of dimensions do not match with");
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

    static Tensor<T, N> broadcast_op(
        const Tensor<T, N>& a,
        const Tensor<T, N>& b,
        const char* err_msg,
        T (*op)(const T&, const T&)
    ) {
        unsigned long result_dims[N];
        for (unsigned long i = 0; i < N; ++i) {
            if (a.dimensiones[i] == b.dimensiones[i]) {
                result_dims[i] = a.dimensiones[i];
            } else if (a.dimensiones[i] == 1) {
                result_dims[i] = b.dimensiones[i];
            } else if (b.dimensiones[i] == 1) {
                result_dims[i] = a.dimensiones[i];
            } else {
                throw TensorError(err_msg);
            }
        }
        Tensor<T, N> result = [&]() {
            if constexpr (N == 2) {
                return Tensor<T,2>(result_dims[0], result_dims[1]);
            } else if constexpr (N == 3) {
                return Tensor<T,3>(result_dims[0], result_dims[1], result_dims[2]);
            } else { // N == 4
                return Tensor<T,4>(
                    result_dims[0],
                    result_dims[1],
                    result_dims[2],
                    result_dims[3]
                );
            }
        }();
        unsigned long total = result.tamano_total();

        unsigned long a_strides[N], b_strides[N], r_strides[N];
        a.calcular_strides(a.dimensiones, a_strides);
        b.calcular_strides(b.dimensiones, b_strides);
        result.calcular_strides(result.dimensiones, r_strides);

        for (unsigned long lin = 0; lin < total; ++lin) {
            unsigned long rem = lin;
            unsigned long idx_r[N];
            for (unsigned long d = 0; d < N; ++d) {
                idx_r[d] = rem / r_strides[d];
                rem %= r_strides[d];
            }
            unsigned long idx_a[N], idx_b[N];
            for (unsigned long d = 0; d < N; ++d) {
                idx_a[d] = (a.dimensiones[d] == 1 ? 0 : idx_r[d]);
                idx_b[d] = (b.dimensiones[d] == 1 ? 0 : idx_r[d]);
            }
            unsigned long lin_a = 0, lin_b = 0;
            for (unsigned long d = 0; d < N; ++d) {
                lin_a += idx_a[d] * a_strides[d];
                lin_b += idx_b[d] * b_strides[d];
            }
            result.datos[lin] = op(a.datos[lin_a], b.datos[lin_b]);
        }
        return result;
    }

    friend Tensor<T, N> operator+(const Tensor<T, N>& a, const Tensor<T, N>& b) {
        return broadcast_op(
            a, b,
            "Shapes do not match and they are not compatible for broadcasting",
            [](const T& x, const T& y){ return x + y; }
        );
    }
    friend Tensor<T, N> operator-(const Tensor<T, N>& a, const Tensor<T, N>& b) {
        return broadcast_op(
            a, b,
            "Shapes do not match and they are not compatible for broadcasting",
            [](const T& x, const T& y){ return x - y; }
        );
    }
    friend Tensor<T, N> operator*(const Tensor<T, N>& a, const Tensor<T, N>& b) {
        return broadcast_op(
            a, b,
            "Shapes do not match and they are not compatible for broadcasting",
            [](const T& x, const T& y){ return x * y; }
        );
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
        else if constexpr (N == 3) {
            unsigned long D0 = t.dimensiones[0];
            unsigned long D1 = t.dimensiones[1];
            unsigned long D2 = t.dimensiones[2];
            os << "{\n";
            for (unsigned long i = 0; i < D0; ++i) {
                os << "{\n";
                for (unsigned long j = 0; j < D1; ++j) {
                    for (unsigned long k = 0; k < D2; ++k) {
                        unsigned long idx = (i * D1 + j) * D2 + k;
                        os << t.datos[idx];
                        if (k + 1 < D2) os << " ";
                    }
                    os << "\n";
                }
                os << "}\n";
            }
            os << "}";
        }
        else /* N == 4 */ {
            unsigned long D0 = t.dimensiones[0];
            unsigned long D1 = t.dimensiones[1];
            unsigned long D2 = t.dimensiones[2];
            unsigned long D3 = t.dimensiones[3];
            os << "{\n";
            for (unsigned long i = 0; i < D0; ++i) {
                os << "{\n";
                for (unsigned long j = 0; j < D1; ++j) {
                    os << "{\n";
                    for (unsigned long k = 0; k < D2; ++k) {
                        for (unsigned long l = 0; l < D3; ++l) {
                            unsigned long idx = ((i * D1 + j) * D2 + k) * D3 + l;
                            os << t.datos[idx];
                            if (l + 1 < D3) os << " ";
                        }
                        os << "\n";
                    }
                    os << "}\n";
                }
                os << "}\n";
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

private:
    T* datos;
    unsigned long dimensiones[N];
    unsigned long capacidad;
};

template <typename T, unsigned long N, size_t... Is>
Tensor<T, N> construct_from_dims(const unsigned long dims[N],
                                  std::index_sequence<Is...>) {
    return Tensor<T, N>(dims[Is]...);
}

template <typename T, unsigned long N>
Tensor<T, N> transpose_2d(const Tensor<T, N>& t) {
    if constexpr (N < 2) {
        throw TensorError("Cannot transpose 1D tensor: need at least 2 dimensions");
    }
    const unsigned long* orig_dims = t.shape();
    unsigned long new_dims[N];
    for (unsigned long i = 0; i < N; ++i)
        new_dims[i] = orig_dims[i];

    if constexpr (N == 2) {
        std::swap(new_dims[0], new_dims[1]);
    } else {
        std::swap(new_dims[N - 2], new_dims[N - 1]);
    }

    Tensor<T, N> result = construct_from_dims<T, N>(
        new_dims, std::make_index_sequence<N>{}
    );

    unsigned long total = t.tamano_total();
    unsigned long old_strides[N], new_strides[N];
    t.calcular_strides(orig_dims, old_strides);
    result.calcular_strides(new_dims, new_strides);

    const T* old_data = t.cbegin();
    T* new_data = result.begin();

    for (unsigned long lin = 0; lin < total; ++lin) {
        unsigned long rem = lin;
        unsigned long idx_old[N];
        for (unsigned long d = 0; d < N; ++d) {
            idx_old[d] = rem / old_strides[d];
            rem %= old_strides[d];
        }
        unsigned long idx_new[N];
        for (unsigned long d = 0; d < N; ++d)
            idx_new[d] = idx_old[d];

        if constexpr (N == 2) {
            std::swap(idx_new[0], idx_new[1]);
        } else {
            std::swap(idx_new[N - 2], idx_new[N - 1]);
        }

        unsigned long lin_new = 0;
        for (unsigned long d = 0; d < N; ++d) {
            lin_new += idx_new[d] * new_strides[d];
        }
        new_data[lin_new] = old_data[lin];
    }
    return result;
}

// ------------------ matrix_product OVERLOADS ------------------

// 2D case: simple matrix multiplication
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

// 3D case: batched matrix multiplication
template <typename T>
Tensor<T, 3> matrix_product(const Tensor<T, 3>& A, const Tensor<T, 3>& B) {
    unsigned long a0 = A.shape()[0], a1 = A.shape()[1], a2 = A.shape()[2];
    unsigned long b0 = B.shape()[0], b1 = B.shape()[1], b2 = B.shape()[2];
    // First check inner dims
    if (a2 != b1) {
        throw TensorError("Matrix dimensions are incompatible for multiplication");
    }
    // Then check batch dims
    if (a0 != b0) {
        throw TensorError("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
    }
    // Result shape: (batch, M, N) = (a0, a1, b2)
    Tensor<T, 3> R(a0, a1, b2);
    for (unsigned long batch = 0; batch < a0; ++batch) {
        for (unsigned long i = 0; i < a1; ++i) {
            for (unsigned long j = 0; j < b2; ++j) {
                T sum = T();
                for (unsigned long k = 0; k < a2; ++k) {
                    sum += A(batch, i, k) * B(batch, k, j);
                }
                R(batch, i, j) = sum;
            }
        }
    }
    return R;
}

}
}
