//---Possibly generated Generated code ()
#include <random>
#include <utility>
#include <vector>
#include <array>
#include <iostream>
using std::cout;
using std::endl;

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


constexpr auto PROBLEM_SIZE = 64 * 1024;
using FP = float;

struct Vec {
    FP* local;
    FP&& x;
    FP&& y;
    FP&& z;

    inline Vec() : local{new FP[3]}, x{std::move(local[0])}, y{std::move(local[1])}, z{std::move(local[2])} {x=y=z=0;}
    inline Vec(FP&& a, FP&& b, FP&& c) : local{0}, x{std::move(a)}, y{std::move(b)}, z{std::move(c)} {}
    inline Vec(const Vec& v): local{0}, x{std::move(v.x)}, y{std::move(v.y)}, z{std::move(v.z)} {}
    inline ~Vec() { if(local) std::cout << "delete" << std::endl; delete [] local; }
    inline Vec& operator= ( const Vec& v) { x = v.x; y = v.y; z = v.z; return *this; }

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
    friend inline auto operator*(const Vec&  a, FP s) -> Vec { return Vec(a) *= s; }
    friend inline auto operator*(const Vec&  a, const Vec&  b) -> Vec { return Vec(a) *= b; }
};

//---Using a facade
template <size_t SIZE>
struct Vecs {
  FP x[SIZE];
  FP y[SIZE];
  FP z[SIZE];
  inline Vec operator[](size_t i) { return Vec{std::move(x[i]),std::move(y[i]),std::move(z[i])}; }  
};

Vec one(1,1,1);

void mytest(Vecs<PROBLEM_SIZE>& vecs) {
    #pragma GCC ivdep
    for (std::size_t i = 0; i < PROBLEM_SIZE; i++) {
        vecs[i] += one;
    }
}

void mytest2(FP* x, FP* y, FP* z) {
  for (std::size_t i = 0; i < PROBLEM_SIZE; i++) {
    x[i] += 1.;
    y[i] += 1.;
    z[i] += 1.;
  }
}

int main(int argc, char** argv) {
    Vecs<PROBLEM_SIZE> vecs;
    auto t1 = high_resolution_clock::now();
    mytest(vecs);
    auto t2 = high_resolution_clock::now();
    auto t3 = high_resolution_clock::now();
    mytest2(vecs.x, vecs.y, vecs.z);
    auto t4 = high_resolution_clock::now();

    cout << duration_cast<std::chrono::nanoseconds>(t2-t1).count() << endl;
    cout << duration_cast<std::chrono::nanoseconds>(t4-t3).count() << endl;
}
