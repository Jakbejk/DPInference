#pragma once
#include <vector>
#include <iostream>
#include <algorithm>

template<typename T>
int argmax(const std::vector<T> &v) {
    if (v.empty()) {
        throw std::invalid_argument("Cannot find argmax of an empty vector");
    }
    auto it = std::max_element(v.begin(), v.end());
    return std::distance(v.begin(), it);
}

template<typename T, size_t N>
int argmax(const std::array<T, N> &v) {
    if (v.empty()) {
        throw std::invalid_argument("Cannot find argmax of an empty vector");
    }
    auto it = std::max_element(v.begin(), v.end());
    return std::distance(v.begin(), it);
}
