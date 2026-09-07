//
// Created by Saluvaji Vishal on 07/08/25.
//

#include "Neuron.hpp"

#include <random>

namespace {
std::mt19937& rng() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
}  // namespace

Neuron::Neuron(int n) {
    this->n = n;
    weights.resize(n);

    for (int i = 0; i < n; ++i) {
        weights[i] = std::make_shared<Value>(dist(rng()));
    }

    bias = std::make_shared<Value>(dist(rng()));
}

std::shared_ptr<Value> Neuron::operator()(const std::vector<std::shared_ptr<Value>>& x) {
    auto out = std::make_shared<Value>(0.0);

    for (int i = 0; i < n; ++i) {
        out = out + this->weights[i] * x[i];
    }

    out = out + this->bias;
    out = out->tanh();

    return out;
}

std::vector<std::shared_ptr<Value>> Neuron::parameters() {
    std::vector<std::shared_ptr<Value>> params;

    for (int i = 0; i < n; ++i) {
        params.push_back(weights[i]);
    }
    params.push_back(bias);

    return params;
}
