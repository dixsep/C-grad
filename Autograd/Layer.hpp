
//
// Created by Saluvaji Vishal on 07/08/25.
//

#pragma once

#include "Neuron.hpp"

class Layer{
private:
    int n;
    vector < Neuron > neurons;
public:

    Layer(int inputs, int neuron_cnt){

        this -> n = neuron_cnt;

        for(int i = 0; i < neuron_cnt; ++i){
           // neurons.push_back(Neuron(inputs));
              neurons.emplace_back(inputs);
        }
    }

    vector<shared_ptr<Value> > operator()(vector < shared_ptr<Value> > &x){

        // len(x) = inouts
        vector < shared_ptr<Value> > outs(n);

        for(int i = 0; i < n; ++i){
            outs[i] = neurons[i](x);
        }

        return outs;
    }

    vector < shared_ptr<Value> > parameters(){

        vector<shared_ptr<Value> > params;

        for(int i = 0; i < n; ++i){
            auto neuron_params = neurons[i].parameters();
            params.insert(params.end(), neuron_params.begin(), neuron_params.end());
        }

        return params;
    }
};