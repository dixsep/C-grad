//
// Created by Saluvaji Vishal on 07/08/25.
//

#pragma once

#include "Value.hpp"

#include <memory>
#include <vector>

/**
 * Single neuron: y = tanh(w · x + b).
 *
 * `n` is the number of inputs (and therefore the number of weights).
 * Weights and bias are Value leaves so they participate in autograd.
 */
class Neuron {
private:
    int n;

public:
    std::vector<std::shared_ptr<Value>> weights;
    std::shared_ptr<Value> bias;

    explicit Neuron(int n);

    /// Forward pass for one example `x` (length must equal `n`).
    std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& x);

    /// Trainable parameters: weights followed by bias.
    std::vector<std::shared_ptr<Value>> parameters();
};
