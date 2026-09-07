//
// Created by Saluvaji Vishal on 07/08/25.
//

#include "MLP.hpp"

MLP::MLP(int input_size, std::vector<int> layers_size) {
    this->n = static_cast<int>(layers_size.size());

    std::vector<int> sz;
    sz.push_back(input_size);

    for (auto& layer_size : layers_size) {
        sz.push_back(layer_size);
    }

    for (int i = 0; i < static_cast<int>(layers_size.size()); ++i) {
        layers.emplace_back(sz[i], sz[i + 1]);
    }
}

std::vector<std::shared_ptr<Value>> MLP::operator()(std::vector<std::shared_ptr<Value>> x) {
    for (int i = 0; i < n; ++i) {
        x = layers[i](x);
    }

    return x;
}

std::vector<std::shared_ptr<Value>> MLP::parameters() {
    std::vector<std::shared_ptr<Value>> params;

    for (int i = 0; i < n; ++i) {
        auto layer_params = layers[i].parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }

    return params;
}
