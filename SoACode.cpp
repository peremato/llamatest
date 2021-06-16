#include <random>
#include <utility>
#include <vector>
#include <array>
#include <iostream>

constexpr auto PROBLEM_SIZE = 1 * 1024;
constexpr auto STEPS = 5;

using FP = float;
constexpr FP TIMESTEP = 0.0001f;
constexpr FP EPS2 = 0.01f;

//---Possibly generated Generated code ()
struct Vec {
    FP&& x;
    FP&& y;
    FP&& z;

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

    friend inline auto operator+(const Vec& a, const Vec& b) -> Vec { Vec r{0,0,0}; r += a; r += b; return r; }
    friend inline auto operator-(const Vec& a, const Vec& b ) -> Vec { Vec r{0,0,0}; r += a; r -= b; return r; }
    friend inline auto operator*(const Vec&  a, FP s) -> Vec { Vec r{0,0,0}; r += a; r *= s; return r; }
    friend inline auto operator*(const Vec&  a, const Vec&  b) -> Vec { Vec r{0,0,0}; r += a; r *= b; return r; }
};

using Pos = Vec;
using Vel = Vec;

struct Particle {
    Pos pos;
    Vel vel;
    FP&& mass;
};


//---Using a facade
template <size_t SIZE>
struct Vecs {
  FP x[SIZE];
  FP y[SIZE];
  FP z[SIZE];
  inline Vec operator[](size_t i) { return Vec{std::move(x[i]),std::move(y[i]),std::move(z[i])}; }  
};

template <size_t SIZE>
struct Particles {
  Vecs<SIZE> pos;
  Vecs<SIZE> vel;
  FP mass[SIZE];
  inline Particle operator[](size_t i) { return Particle{ pos[i], vel[i], std::move(mass[i])}; } 
};


inline void pPInteraction(Particle& p1, const Particle& p2) {
    auto distance = p1.pos - p2.pos;
    distance *= distance;
    const FP distSqr = EPS2 + distance.x + distance.y + distance.z;
    const FP distSixth = distSqr * distSqr * distSqr;
    const FP invDistCube = 1.0f / std::sqrt(distSixth);
    const FP s = p2.mass * invDistCube;
    distance *= s * TIMESTEP;
    p1.vel += distance;
}

void update(Particles<PROBLEM_SIZE>& particles) {
    for (std::size_t i = 0; i < PROBLEM_SIZE; i++) {
        Particle pi = particles[i];
        std::cout << pi.pos.x << std::endl;

        #pragma GCC ivdep
        for (std::size_t j = 0; j < PROBLEM_SIZE; j++)
            pPInteraction(pi, particles[j]);
        //particles[i].vel = pi.vel;
    }
}

void move(Particles<PROBLEM_SIZE>& particles) {
    #pragma GCC ivdep
    for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        particles[i].pos += particles[i].vel * TIMESTEP;
}

struct A {
    int&& a;
};



int main(int argc, char** argv) {


    int aa[10];

    A c{999};

    A b{ std::move(aa[0]) };
    b.a = 99;

    std::cout << aa[0] << std::endl;

    Particles<PROBLEM_SIZE> particles;

    std::default_random_engine engine;
    std::normal_distribution<FP> distribution(FP(0), FP(1));
    for (std::size_t i = 0; i < PROBLEM_SIZE; i++) {
        auto p = particles[i];
        p.pos.x = distribution(engine);
        p.pos.y = distribution(engine);
        p.pos.z = distribution(engine);
        p.vel.x = distribution(engine) / FP(10);
        p.vel.y = distribution(engine) / FP(10);
        p.vel.z = distribution(engine) / FP(10);
        p.mass = distribution(engine) / FP(100);
    }
    std::cout << "here " << particles[10].pos.x << std::endl;

    for (std::size_t s = 0; s < STEPS; ++s) {
        update(particles);
        move(particles);
    }

    return 0;
}
