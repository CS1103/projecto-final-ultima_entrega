#pragma once

namespace utec {
    namespace agent {

        struct State {
            float ball_x;
            float ball_y;
            float paddle_y;
        };

        class EnvGym {
        public:
            virtual ~EnvGym() = default;
            virtual State reset() = 0;
            virtual State step(int action, float &reward, bool &done) = 0;
        };

    } // namespace agent
} // namespace utec