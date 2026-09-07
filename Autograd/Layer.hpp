//
// Created by Saluvaji Vishal on 07/08/25.
//

#pragma once

#include "Neuron.hpp"

#include <memory>
#include <vector>

/**
 * Dense layer: a list of neurons that all receive the same input vector.
 *
 * `inputs` is the incoming feature count; `neuron_cnt` is the layer width.
 */
class Layer {
private:
    int n;  // number of neurons
    std::vector<Neuron> neurons;

public:
    Layer(int inputs, int neuron_cnt);

    std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>>& x);

    std::vector<std::shared_ptr<Value>> parameters();
};
