//
// Created by Saluvaji Vishal on 07/08/25.
//

#include "Layer.hpp"

Layer::Layer(int inputs, int neuron_cnt) {
    this->n = neuron_cnt;

    for (int i = 0; i < neuron_cnt; ++i) {
        neurons.emplace_back(inputs);
    }
}

std::vector<std::shared_ptr<Value>> Layer::operator()(std::vector<std::shared_ptr<Value>>& x) {
    std::vector<std::shared_ptr<Value>> outs(n);

    for (int i = 0; i < n; ++i) {
        outs[i] = neurons[i](x);
    }

    return outs;
}

std::vector<std::shared_ptr<Value>> Layer::parameters() {
    std::vector<std::shared_ptr<Value>> params;

    for (int i = 0; i < n; ++i) {
        auto neuron_params = neurons[i].parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }

    return params;
}
