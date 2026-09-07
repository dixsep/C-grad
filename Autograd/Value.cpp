//
// Created by Saluvaji Vishal on 02/08/25.
//

#include "Value.hpp"

#include <algorithm>
#include <cmath>

Value::Value(double data, std::string op)
    : _data(data), _grad(0.0), _backward([]() {}), _op(std::move(op)) {}

Value::Value(double data, std::initializer_list<std::shared_ptr<Value>> children,
             std::string op)
    : _data(data),
      _grad(0.0),
      _backward([]() {}),
      _prev(children.begin(), children.end()),
      _op(std::move(op)) {}

std::shared_ptr<Value> Value::relu() {
    auto self = shared_from_this();
    auto out = std::make_shared<Value>(self->_data > 0 ? self->_data : 0,
                                       std::initializer_list<std::shared_ptr<Value>>{self},
                                       "Relu");

    // d(relu)/dx = 1 if x > 0 else 0; out->_data already holds relu(x).
    out->_backward = [self, out]() {
        self->_grad += (out->_data > 0) * out->_grad;
    };

    return out;
}

std::shared_ptr<Value> Value::power(double p) {
    auto self = shared_from_this();
    auto out = std::make_shared<Value>(std::pow(self->_data, p),
                                       std::initializer_list<std::shared_ptr<Value>>{self},
                                       "pow");

    // d(x^p)/dx = p * x^(p-1)
    out->_backward = [self, out, p]() {
        self->_grad += out->_grad * (p * std::pow(self->_data, p - 1));
    };

    return out;
}

std::shared_ptr<Value> Value::exp() {
    auto self = shared_from_this();
    auto out = std::make_shared<Value>(std::exp(self->_data),
                                       std::initializer_list<std::shared_ptr<Value>>{self},
                                       "exp");

    // d(e^x)/dx = e^x = out->_data
    out->_backward = [self, out]() {
        self->_grad += (out->_data) * out->_grad;
    };

    return out;
}

std::shared_ptr<Value> Value::tanh() {
    auto self = shared_from_this();
    auto out = std::make_shared<Value>(std::tanh(self->_data),
                                       std::initializer_list<std::shared_ptr<Value>>{self},
                                       "Tanh");

    // d(tanh x)/dx = 1 - tanh(x)^2
    out->_backward = [self, out]() {
        self->_grad += (1 - out->_data * out->_data) * out->_grad;
    };

    return out;
}

void Value::build_topo(std::shared_ptr<Value> node,
                       std::set<std::shared_ptr<Value>>& visited,
                       std::vector<std::shared_ptr<Value>>& topo) {
    visited.insert(node);

    for (auto& child : node->_prev) {
        if (visited.find(child) == visited.end()) {
            build_topo(child, visited, topo);
        }
    }

    topo.push_back(node);
}

void Value::backward() {
    std::set<std::shared_ptr<Value>> visited;
    std::vector<std::shared_ptr<Value>> topo;

    build_topo(shared_from_this(), visited, topo);
    std::reverse(topo.begin(), topo.end());

    // Seed: derivative of the output w.r.t. itself is 1.
    this->_grad = 1.0;
    for (auto& node : topo) {
        node->_backward();
    }
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs,
                                 const std::shared_ptr<Value>& rhs) {
    auto out = std::make_shared<Value>(lhs->_data + rhs->_data,
                                       std::initializer_list<std::shared_ptr<Value>>{lhs, rhs},
                                       "+");

    out->_backward = [lhs, rhs, out]() {
        lhs->_grad += out->_grad;
        rhs->_grad += out->_grad;
    };

    return out;
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const double& value) {
    return operator+(lhs, std::make_shared<Value>(value));
}

std::shared_ptr<Value> operator+(const double& value, const std::shared_ptr<Value>& rhs) {
    return operator+(rhs, value);
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs,
                                 const std::shared_ptr<Value>& rhs) {
    auto out = std::make_shared<Value>(lhs->_data * rhs->_data,
                                       std::initializer_list<std::shared_ptr<Value>>{lhs, rhs},
                                       "*");

    out->_backward = [lhs, rhs, out]() {
        lhs->_grad += out->_grad * rhs->_data;
        rhs->_grad += out->_grad * lhs->_data;
    };

    return out;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const double& value) {
    return operator*(lhs, std::make_shared<Value>(value));
}

std::shared_ptr<Value> operator*(const double& value, const std::shared_ptr<Value>& rhs) {
    return operator*(rhs, value);
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs,
                                 const std::shared_ptr<Value>& rhs) {
    auto out = std::make_shared<Value>(lhs->_data - rhs->_data,
                                       std::initializer_list<std::shared_ptr<Value>>{lhs, rhs},
                                       "-");

    out->_backward = [lhs, rhs, out]() {
        lhs->_grad += out->_grad;
        rhs->_grad += out->_grad * -1;
    };

    return out;
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs,
                                 const std::shared_ptr<Value>& rhs) {
    auto RHS = rhs->power(-1);
    return operator*(lhs, RHS);
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const double& value) {
    return operator/(lhs, std::make_shared<Value>(value));
}

std::shared_ptr<Value> operator/(const double& value, const std::shared_ptr<Value>& rhs) {
    return operator/(std::make_shared<Value>(value), rhs);
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs) {
    return operator*(lhs, -1);
}
