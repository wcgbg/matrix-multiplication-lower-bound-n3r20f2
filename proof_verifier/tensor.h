#pragma once

#include <array>
#include <cstdint>

template <int n0, int n1, int n2>
using Tensor =
    std::array<std::array<std::array<uint8_t, n2 * n0>, n1 * n2>, n0 * n1>;

template <int n> using SquareTensor = Tensor<n, n, n>;
