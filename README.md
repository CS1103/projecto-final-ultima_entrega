[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

Implementación en C++ de un agente para el juego Pong que utiliza una red neuronal multicapa entrenada desde cero. El proyecto incluye:

- Una librería de álgebra de tensores genéricos (Tensor<T, Rank>) con reshape, broadcasting y operaciones element‑wise.  
- Capas densas (Dense), función de activación ReLU y función de pérdida MSE, todo implementado sin dependencias de frameworks de deep learning.  
- Un entorno simulado (EnvGym) que replica la dinámica básica de Pong y una clase PongAgent que, usando la red entrenada, decide acciones (mover la paleta hacia arriba, abajo o quedarse).  

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `group_3_custom_name`
* **Integrantes**:
  * Requelmy Limaco Porras – 202410015 (Responsable de investigación teórica, Pruebas y benchmarking)
  * Leonardo Martinez Aquino – 202410148 (Implementación del modelo, Documentación y demo)
  * Jorge Armando Martínez Palomino - 202310094 (Desarrollo de la arquitectura y desarrollo del código)
---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior (con soporte para C++20)
2. **Dependencias**:
   * Ninguna (solo STL)
3. **Instalación**:

```bash
git clone https://github.com/CS1103/projecto-final-ultima_entrega
cd 
mkdir build && cd build
cmake ..
make
```
---

### 1. Investigación teórica

**Objetivo**: Explorar los fundamentos de las redes neuronales artificiales y su aplicación en agentes inteligentes.

#### 1. Historia y evolución de las NNs

- **1943 – 1960** : McCulloch y Pitts proponen la primera neurona binaria. Rosenblatt crea el Perceptrón (1958), pero sus limitaciones (no resuelve XOR) y se produce la falta de potencia computacional.
- **1980** : Renace el interés con el algoritmo de retropropagación (Rumelhart, 1986), que permite entrenar redes multicapa.
- **1990 – 2000** : Aparecen redes más complejas como las CNN (LeCun) para visión, y las LSTM (1997) para secuencias. Hinton introduce el preentrenamiento profundo, una tecnica dentro del deep learning (2006).
- **2010 – hoy** : Auge del deep learning gracias al uso de GPUs. Surgen redes neuronales mejoradas como AlexNet (2012), GANs (2014), y Transformers (2017), que dieron origen a modelos como *GPT*, *BERT* y *ChatGPT*.

#### 2. Principales arquitecturas

- **MLP (Perceptrón Multicapa)**:  
  Redes feedforward totalmente conectadas. Utilizan funciones de activación como sigmoide o ReLU. Se emplean en tareas de clasificación y regresión general.

- **CNN (Redes Convolucionales)**:  
  Usadas en visión por computador. Aplican filtros para detectar patrones visuales y reducen la dimensionalidad. Ideales para imágenes y video.

- **RNN (Redes Recurrentes)**:  
  Procesan datos secuenciales (texto, audio, etc). Mantienen un estado interno para tener memoria. Algunas variantes populares son: LSTM, GRU.

#### 3. Algoritmos de entrenamiento

- **Retropropagación (Backpropagation)**:  
  Ajusta los pesos propagando el error desde la salida hacia las capas anteriores usando la regla de la cadena. Calcula el gradiente y actualiza pesos.

- **Funciones de pérdida**:
  - Regresión: Error Cuadrático Medio (MSE).
  - Clasificación: Entropía Cruzada (Cross-Entropy).

- **Optimizadores**:
  - SGD (Stochastic Gradient Descent): Actualiza pesos en minilotes, acelerando el entrenamiento.
  - Adam: Combina tasas de aprendizaje adaptativas con momentos. Muy eficiente y popular.
  - Otros: RMSprop, Adagrad, Adadelta, etc.

---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**: ejemplo: Factory para capas, Strategy para optimizadores.
* **Estructura de carpetas (ejemplo)**:

  ```
  pong_ai/
  ├── include/
  │   └── utec/
  │       ├── algebra/               
  │       │   └── Tensor.h
  │       ├── nn/                    
  │       │   ├── layer.h
  │       │   ├── dense.h
  │       │   ├── activation.h
  │       │   ├── loss.h
  │       │   └── neural_network.h
  │       └── agent/                 
  │           ├── PongAgent.h
  │           └── EnvGym.h
  ├── src/                           
  │   └── utec/
  │       └── agent/
  │           ├── PongAgent.cpp
  │           └── EnvGym.cpp
  ├── tests/                         
  │   ├── test_tensor.cpp
  │   ├── test_neural_network.cpp
  │   └── test_agent_env.cpp
  ├── docs/                          
  ├── README.md
  └── BIBLIOGRAFIA.md

  ```

## 2.2 Manual de uso y casos de prueba

### Cómo ejecutar

El programa se compila y ejecuta desde CLion o por línea de comandos. No requiere archivos de entrada.

```bash
./pong_ai
```
### Casos de prueba
**Test unitario de capa densa (Dense)**
* Se comprueba su funcionamiento correcto durante el entrenamiento del XOR.

* El forward realiza una multiplicación matricial y suma el bias.

* El backward propaga correctamente el gradiente hacia atrás y calcula dW y db.

**Test de función de activación ReLU**
* Aplicada tras la primera capa Dense.

* Corrige los valores negativos a cero y permite el paso de valores positivos.

Se valida con propagación hacia adelante y hacia atrás (forward y backward).

**Test de convergencia en dataset XOR**
* El modelo se entrena con 4 entradas representando el XOR lógico.

* Debido a errores en la ejecución o configuración, no se obtuvo convergencia completa en esta versión.

---

### 3. Ejecución

#### 🔧 Pasos para ejecutar

1. Clona el repositorio y compila el proyecto:

```bash
git clone https://github.com/CS1103/projecto-final-ultima_entrega/
cd proyecto-pong-nn
mkdir build && cd build
cmake ..
make
```
---

### 4. Análisis del rendimiento

## 4. Análisis del rendimiento

### Métricas de entrenamiento

- **Iteraciones**:
- **Tiempo total de entrenamiento**:
- **Precisión final**: 

### Ventajas y desventajas

**Ventajas:**

- Implementación ligera en C++.
- Dependencias mínimas (solo STL).
- Estructura clara y educativa del flujo forward → backward → optimize.

**Desventajas:**

- Sin paralelización (entrenamiento en un solo hilo).
- Sin soporte de batches.
- Matriz multiplicada con bucles simples (poco eficiente).

### Mejoras futuras propuestas

- **Uso de BLAS**: Para acelerar las multiplicaciones, se podría integrar bibliotecas como Eigen o OpenBLAS, aprovechando sus rutinas optimizadas en C++.
- **Paralelización del entrenamiento**: Dividir el dataset en minibatches y aplicar hilos con std::thread permitiría escalar en CPUs multinúcleo, acelerando el entrenamiento.

---

### 5. Trabajo en equipo

| Tarea                     | Miembro  | Rol                       |
| ------------------------- | -------- | ------------------------- |
| Investigación teórica     | Requelmy Limaco Porras (202410015) | Documentar bases teóricas |
| Diseño de la arquitectura | Leonardo Martinez Aquino (202410148) | UML y esquemas de clases  |
| Implementación del modelo | Leonardo Martinez Aquino (202410148) | Código C++ de la NN       |
| Implementación del modelo | Jorge Armnando Martínez Palomino (202310094) | Código C++ de la NN       |
| Pruebas y benchmarking    | Requelmy Limaco Porras (202410015 | Generación de métricas    |

---

### 6. Conclusiones

- **Logros**  
  • Se implementó desde cero un sistema de álgebra de tensores genéricos (Tensor<T, Rank>) capaz de reshape, broadcasting y operaciones element‑wise.  
  • Se desarrolló una red neuronal modular en C++ (NeuralNetwork) con capas densas (Dense), activación ReLU, y función de pérdida MSE.  
  • Se integró la red en un agente de Pong (PongAgent) y un entorno simulado (EnvGym), validando el pipeline completo de forward, backpropagation y actualización de pesos.  

- **Evaluación**  
  • El código muestra una calidad adecuada: patrones de diseño claros, gestión de memoria (copy/move) correcta, y un CMakeLists.txt fácil de mantener.  
  • El rendimiento en CPU es aceptable para redes pequeñas y batches reducidos, aunque presenta cuellos de botella en operaciones de multiplicación de matrices y broadcasting en tensores de mayor tamaño.

- **Aprendizajes** 
  • Profundización en el algoritmo de retropropagación: caché de activaciones, cálculo de gradientes en Dense::backward y en MSELoss::backward.  
  • Manejo manual de memoria dinámica en C++ (new/delete) y diseño de constructores copy/move para evitar fugas y dobles liberaciones.  
  • Comprensión de la importancia de inicialización de pesos, escalado de gradientes y elección de tasa de aprendizaje.

- **Recomendaciones**  
  • Extender la librería de tensores para soportar paralelismo (OpenMP/CUDA) y tipos de datos mixtos.  
  • Agregar funcionalidad de persistencia (guardar/cargar pesos) y herramientas de visualización de métricas (p. ej. gráficas de pérdida y precisión).  
  • Probar el agente en datasets reales (MNIST, CIFAR-10) o conectar con un entorno gráfico para evaluar rendimiento en escenarios más complejos.

---

### 7. Bibliografía

[1] I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*, Cambridge, MA: MIT Press, 2016. [Online]. Disponible: https://www.deeplearningbook.org

[2] D. E. Rumelhart, G. E. Hinton y R. J. Williams, “Learning representations by back-propagating errors,” *Nature*, vol. 323, pp. 533–536, 1986. [Online]. Disponible: https://www.nature.com/articles/323533a0

[3] D. P. Kingma y J. Ba, “Adam: A method for stochastic optimization,” en *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*, 2015. [Online]. Disponible: https://arxiv.org/abs/1412.6980

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
