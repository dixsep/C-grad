//
// Created by Saluvaji Vishal on 02/08/25.
//

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <memory>         // For shared_ptr, enable_shared_from_this
#include <functional>     // For std::function
#include <initializer_list> // For std::initializer_list
#include <cmath>
#include <algorithm>
#include<math.h>

using namespace std;

/*
   enable_shared_from_this<Value> : this curr object can be used as a shared pointer
   declaration and impl in header files
 */

class Value : public std::enable_shared_from_this<Value> {

public:
    double _data;
    double _grad;
private:
    std::function<void()> _backward;
    set < shared_ptr<Value> > _prev;
    string _op;

public:

    // leaf node
    Value(double data, string op = "") : _data(data), _grad(0.0), _backward([](){}), _op(op) {

    }

    // non leaf node
    Value(double data, initializer_list<shared_ptr<Value> > children, string op = "")
            : _data(data), _grad(0.0), _backward([](){}), _prev(children.begin(), children.end()), _op(op) {

    }

    // RelU
    shared_ptr<Value> relu(){

        auto self = shared_from_this();
        auto out = make_shared<Value>(self -> _data > 0 ? self -> _data : 0, std::initializer_list<shared_ptr<Value>>{self}, "Relu");

        out -> _backward = [self, out]() {
            self -> _grad += (out -> _data > 0) * out -> _grad;
        };

        return out;
    }

    // power
    shared_ptr<Value> power(double p){


        auto self = shared_from_this();
        auto out = make_shared<Value>(pow(self -> _data, p), std::initializer_list<shared_ptr<Value>>{self}, "pow");


        out -> _backward = [self, out, p](){
            self -> _grad += out -> _grad * (p * pow(self -> _data, p - 1));
        };

        return out;
    }


    // exp
    shared_ptr<Value> exp(){

        auto self = shared_from_this();
        auto out = make_shared<Value>(std::exp(self -> _data), std::initializer_list<shared_ptr<Value>>{self}, "exp");

        out -> _backward = [self, out](){
            self -> _grad += (out -> _data) * out -> _grad;
        };

        return out;
    }

    // TanH
    shared_ptr<Value> tanh(){

        auto self = shared_from_this();
        auto out = make_shared<Value>(std::tanh(self -> _data), std::initializer_list<shared_ptr<Value>>{self}, "Tanh");

        out -> _backward = [self, out](){
            self -> _grad += (1 - out -> _data * out -> _data) * out -> _grad;
        };

        return out;
    }

    //topological ordering
    void build_topo(shared_ptr<Value> node, set<shared_ptr<Value>> &visited, vector <shared_ptr<Value>> &topo){

        visited.insert(node);

        for (auto &child : node -> _prev){
            if(visited.find(child) == visited.end()){
                build_topo(child, visited, topo);
            }
        }

        topo.push_back(node);
    }

    // back propagation
    void backward(){

        set<shared_ptr<Value>> visited;
        vector <shared_ptr<Value>> topo;

        build_topo(shared_from_this(), visited, topo);
        reverse(topo.begin(), topo.end());

        this -> _grad = 1.0;
        for(auto &node : topo){
            node -> _backward();
        }
    }

    friend shared_ptr<Value> operator+(const shared_ptr<Value> &lhs, const shared_ptr<Value> &rhs);

    friend shared_ptr<Value> operator*(const shared_ptr<Value> &lhs, const shared_ptr<Value> &rhs);

    // implement for subtraction and division
    friend shared_ptr<Value> operator-(const shared_ptr<Value> &lhs, const shared_ptr<Value> &rhs);

    friend shared_ptr<Value> operator/(const shared_ptr<Value> &lhs, const shared_ptr<Value> &rhs);

    //negation
   friend shared_ptr<Value> operator-(const shared_ptr<Value> &lhs);
};