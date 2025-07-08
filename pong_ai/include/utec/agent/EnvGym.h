namespace utec::agent {
/// Estado del juego Pong
struct State {
    float ball_x;
    float ball_y;
    float paddle_y;
};

/// Interface para un entorno tipo Gym de Pong
class EnvGym {
public:
    virtual ~EnvGym() = default;
    /// Reinicia el entorno y devuelve el estado inicial
    virtual State reset() = 0;
    /**
     * Aplica la acción:
     *  - action = -1 mueve la paleta hacia abajo  
     *  - action =  0 no mueve  
     *  - action = +1 mueve la paleta hacia arriba  
     * Devuelve el nuevo estado, y asigna reward y done.
     */
    virtual State step(int action, float &reward, bool &done) = 0;
};

} // namespace utec::agent
