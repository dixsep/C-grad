//
// Created by Saluvaji Vishal on 07/08/25.
//

#pragma once

#include "Layer.hpp"

#include <memory>
#include <vector>

/**
 * Multi-layer perceptron: a sequence of dense tanh layers.
 *
 * `input_size` is the width of the first layer's input.
 * `layers_size` lists the neuron count of each layer, including the output.
 * Example: MLP(3, {4, 4, 1}) is 3 -> 4 -> 4 -> 1.
 */
class MLP {
private:
    std::vector<Layer> layers;
    int n;  // number of layers

public:
    MLP(int input_size, std::vector<int> layers_size);

    std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> x);

    std::vector<std::shared_ptr<Value>> parameters();
};
