cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(llamatest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-march=native -ffast-math")

find_package(Boost)
find_package(fmt)

add_executable(llama LlamaCode.cpp)
target_include_directories(llama PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(llama fmt::fmt)

add_executable(naive AoSCode.cpp)
add_executable(killer SoACode.cpp)


