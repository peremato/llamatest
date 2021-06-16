#include <random>
#include <utility>
#include <vector>
#include <array>

constexpr auto PROBLEM_SIZE = 64 * 1024;
constexpr auto STEPS = 5;

using FP = float;
constexpr FP TIMESTEP = 0.0001f;
constexpr FP EPS2 = 0.01f;

struct Vec {
    FP x;
    FP y;
    FP z;

    inline auto operator*=(FP s) -> Vec& {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    inline auto operator*=(const Vec& v) -> Vec& {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    inline auto operator+=(const Vec& v) -> Vec& {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    inline auto operator-=(const Vec& v) -> Vec& {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    friend inline auto operator+(const Vec& a, const Vec& b) -> Vec { return Vec(a) += b; }
    friend inline auto operator-(const Vec& a, const Vec& b ) -> Vec { return Vec(a) -= b; }
    friend inline auto operator*(const Vec& a, FP s) -> Vec { return Vec(a) *= s; }
    friend inline auto operator*(const Vec& a, const Vec&  b) -> Vec { return Vec(a) *= b; }
};

using Pos = Vec;
using Vel = Vec;

struct Particle {
    Pos pos;
    Vel vel;
    FP mass;
};

inline void pPInteraction(Particle& p1, const Particle& p2) {
    auto distance = p1.pos + p2.pos;
    distance *= distance;
    const FP distSqr = EPS2 + distance.x + distance.y + distance.z;
    const FP distSixth = distSqr * distSqr * distSqr;
    const FP invDistCube = 1.0f / std::sqrt(distSixth);
    const FP s = p2.mass * invDistCube;
    distance *= s * TIMESTEP;
    p1.vel += distance;
}

void update(Particle* particles) {
    for (std::size_t i = 0; i < PROBLEM_SIZE; i++) {
        Particle pi = particles[i];
        #pragma GCC ivdep
        for (std::size_t j = 0; j < PROBLEM_SIZE; j++)
            pPInteraction(pi, particles[j]);
        particles[i].vel = pi.vel;
    }
}

void move(Particle* particles) {
    #pragma GCC ivdep
    for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        particles[i].pos += particles[i].vel * TIMESTEP;
}

int main(int argc, char** argv) {
    std::array<Particle,PROBLEM_SIZE> particles;

    std::default_random_engine engine;
    std::normal_distribution<FP> distribution(FP(0), FP(1));
    for (auto& p : particles) {
        p.pos.x = distribution(engine);
        p.pos.y = distribution(engine);
        p.pos.z = distribution(engine);
        p.vel.x = distribution(engine) / FP(10);
        p.vel.y = distribution(engine) / FP(10);
        p.vel.z = distribution(engine) / FP(10);
        p.mass = distribution(engine) / FP(100);
    }

    for (std::size_t s = 0; s < STEPS; ++s) {
        update(particles.data());
        move(particles.data());
    }

    return 0;
}
