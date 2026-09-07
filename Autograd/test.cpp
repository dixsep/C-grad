#include "MLP.hpp"
#include "Value.hpp"

#include <iostream>
#include <memory>
#include <vector>

void Value_check() {
    std::cout << "--- Simple Test ---" << std::endl;

    auto a = std::make_shared<Value>(-4.0);
    auto b = std::make_shared<Value>(2.0);

    auto c = a + b;
    auto d = a * b + b->power(3);

    c = c + c + 1;
    c = c + 1 + c + (-a);

    d = d + d * 2 + (b + a)->relu();
    d = d + 3 * d + (b - a)->relu();

    auto e = c - d;
    auto f = e->power(2);

    auto g = f / 2;
    g = g + 10 / f;

    auto ee = a->exp();
    auto tt = a->tanh();
    auto rr = a->relu();

    (void)ee;
    (void)tt;
    (void)rr;

    std::cout << "Forward Pass \n";
    std::cout << "G : " << g->_data << '\n';

    std::cout << "---------Backward Pass-----------\n";
    g->backward();

    std::cout << "dg/da : " << a->_grad << '\n';
    std::cout << "dg/db : " << b->_grad << '\n';
}

void mlp_check1() {
    MLP n = MLP(3, {4, 4, 1});

    std::vector<std::shared_ptr<Value>> x = {
        std::make_shared<Value>(2.0),
        std::make_shared<Value>(3.0),
        std::make_shared<Value>(5.0)};

    std::vector<std::shared_ptr<Value>> y = n(x);

    for (auto& ele : y) {
        std::cout << ele->_data << ' ';
    }
}

void mlp_check2() {
    MLP n = MLP(3, {4, 4, 1});

    std::vector<std::vector<std::shared_ptr<Value>>> xs = {
        {std::make_shared<Value>(2.0), std::make_shared<Value>(3.0), std::make_shared<Value>(-1.0)},
        {std::make_shared<Value>(3.0), std::make_shared<Value>(-1.0), std::make_shared<Value>(0.5)},
        {std::make_shared<Value>(0.5), std::make_shared<Value>(1.0), std::make_shared<Value>(1.0)},
        {std::make_shared<Value>(1.0), std::make_shared<Value>(1.0), std::make_shared<Value>(-1.0)}};

    std::vector<std::shared_ptr<Value>> ys = {
        std::make_shared<Value>(1.0),
        std::make_shared<Value>(-1.0),
        std::make_shared<Value>(-1.0),
        std::make_shared<Value>(1.0)};

    double lr = 0.05;

    for (int step = 0; step < 5; ++step) {
        std::vector<std::shared_ptr<Value>> y_pred;

        for (int i = 0; i < static_cast<int>(xs.size()); ++i) {
            auto x = xs[i];
            auto y = n(x)[0];
            y_pred.push_back(y);
        }

        auto loss = std::make_shared<Value>(0.0);

        for (int i = 0; i < static_cast<int>(xs.size()); ++i) {
            loss = loss + (ys[i] - y_pred[i]) * (ys[i] - y_pred[i]);
        }

        auto params = n.parameters();
        for (int i = 0; i < static_cast<int>(params.size()); ++i) {
            params[i]->_grad = 0.0;
        }

        loss->backward();

        for (int i = 0; i < static_cast<int>(params.size()); ++i) {
            params[i]->_data += lr * -params[i]->_grad;
        }

        std::cout << "Loss : " << loss->_data << "\n";
        std::cout << "Predictions : \n";

        for (int i = 0; i < static_cast<int>(xs.size()); ++i) {
            std::cout << y_pred[i]->_data << " " << ys[i]->_data << '\n';
        }
    }
}

int main() {
    // Value_check();
    // mlp_check1();
    // mlp_check2();
}
