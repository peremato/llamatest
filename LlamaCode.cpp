#include "llama.hpp"
#include <random>
#include <utility>
#include <vector>

constexpr auto Mapping = 1; /// 0 AoS, 1 SoA, 2 AoSoA
constexpr auto PROBLEM_SIZE = 64 * 1024;
constexpr auto STEPS = 5;
constexpr auto AoSoALanes = 16;

using FP = float;
constexpr FP TIMESTEP = 0.0001f;
constexpr FP EPS2 = 0.01f;

namespace tag {
    struct Pos{}; struct Vel{};
    struct X{}; struct Y{}; struct Z{}; struct Mass{};
}
using Vec = llama::Record<
    llama::Field<tag::X, FP>,
    llama::Field<tag::Y, FP>,
    llama::Field<tag::Z, FP>>;
using Particle = llama::Record<
    llama::Field<tag::Pos, Vec>,
    llama::Field<tag::Vel, Vec>,
    llama::Field<tag::Mass, FP>>;

//LLAMA_FN_HOST_ACC_INLINE
template <typename PI, typename PJ>
void pPInteraction(PI&& pi, PJ&& pj) {
    auto dist = pi(tag::Pos{}) - pj(tag::Pos{});
    dist *= dist;
    const FP distSqr = EPS2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
    const FP distSixth = distSqr * distSqr * distSqr;
    const FP invDistCube = 1.0f / std::sqrt(distSixth);
    const FP sts = pj(tag::Mass{}) * invDistCube * TIMESTEP;
    pi(tag::Vel{}) += dist * sts;
}

template <typename View>
void update(View& particles) {
    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < PROBLEM_SIZE; i++) {
        llama::One<Particle> pi;
        pi = particles(i);
        for (std::size_t j = 0; j < PROBLEM_SIZE; ++j)
            pPInteraction(pi, particles(j));
        particles(i)(tag::Vel{}) = pi(tag::Vel{});
    }
}

template <typename View>
void move(View& particles) {
    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        particles(i)(tag::Pos()) += particles(i)(tag::Vel()) * TIMESTEP;
}

int main() {
    auto mapping = [] {
        const auto arrayDims = llama::ArrayDims{PROBLEM_SIZE};
        if constexpr (Mapping == 0)
            return llama::mapping::AoS<decltype(arrayDims), Particle>{arrayDims};
        if constexpr (Mapping == 1)
            return llama::mapping::SoA<decltype(arrayDims), Particle, true>{arrayDims};
        if constexpr (Mapping == 2)
            return llama::mapping::AoSoA<decltype(arrayDims), Particle, AoSoALanes>{arrayDims};
    }();

    auto particles = llama::allocView(mapping);

    std::default_random_engine engine;
    std::normal_distribution<FP> distribution(FP(0), FP(1));
    for (std::size_t i = 0; i < PROBLEM_SIZE; ++i) {
        auto p = particles(i);
        p(tag::Pos(), tag::X()) = distribution(engine);
        p(tag::Pos(), tag::Y()) = distribution(engine);
        p(tag::Pos(), tag::Z()) = distribution(engine);
        p(tag::Vel(), tag::X()) = distribution(engine) / FP(10);
        p(tag::Vel(), tag::Y()) = distribution(engine) / FP(10);
        p(tag::Vel(), tag::Z()) = distribution(engine) / FP(10);
        p(tag::Mass()) = distribution(engine) / FP(100);
    }

    for (std::size_t s = 0; s < STEPS; ++s) {
        update(particles);
        move(particles);
    }

    return 0;
}