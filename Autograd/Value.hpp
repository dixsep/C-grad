//
// Created by Saluvaji Vishal on 02/08/25.
//

#pragma once

#include <functional>
#include <initializer_list>
#include <memory>
#include <set>
#include <string>
#include <vector>

/**
 * Scalar node in a reverse-mode autodiff graph.
 *
 * Each Value stores a number (_data), the gradient of the loss w.r.t. that
 * number (_grad), and enough graph metadata to run backpropagation:
 * the local backward closure, parent nodes, and the op that produced this node.
 *
 * Nodes must be owned by std::shared_ptr so shared_from_this() is valid
 * when building child nodes (relu, +, *, ...).
 */
class Value : public std::enable_shared_from_this<Value> {
public:
    double _data;
    double _grad;

private:
    std::function<void()> _backward;
    std::set<std::shared_ptr<Value>> _prev;
    std::string _op;

public:
    /// Leaf node (no parents); typically a constant, input, or parameter.
    Value(double data, std::string op = "");

    /// Intermediate node produced by an operation on `children`.
    Value(double data, std::initializer_list<std::shared_ptr<Value>> children,
          std::string op = "");

    std::shared_ptr<Value> relu();
    std::shared_ptr<Value> power(double p);
    std::shared_ptr<Value> exp();
    std::shared_ptr<Value> tanh();

    /// Depth-first walk that appends nodes after their children (post-order).
    void build_topo(std::shared_ptr<Value> node,
                    std::set<std::shared_ptr<Value>>& visited,
                    std::vector<std::shared_ptr<Value>>& topo);

    /// Reverse-mode autodiff: d(this)/d(ancestor) for every ancestor.
    void backward();

    friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs,
                                            const std::shared_ptr<Value>& rhs);
    friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs,
                                            const std::shared_ptr<Value>& rhs);
    friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs,
                                            const std::shared_ptr<Value>& rhs);
    friend std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs,
                                            const std::shared_ptr<Value>& rhs);
    friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs);
};

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs,
                                 const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const double& value);
std::shared_ptr<Value> operator+(const double& value, const std::shared_ptr<Value>& rhs);

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs,
                                 const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const double& value);
std::shared_ptr<Value> operator*(const double& value, const std::shared_ptr<Value>& rhs);

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs,
                                 const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs);

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs,
                                 const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const double& value);
std::shared_ptr<Value> operator/(const double& value, const std::shared_ptr<Value>& rhs);
