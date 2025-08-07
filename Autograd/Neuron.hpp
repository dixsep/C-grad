
//
// Created by Saluvaji Vishal on 07/08/25.
//

#pragma once
#include "Value.hpp"

#include<random>

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dist(-1.0, 1.0);

class Neuron{
private:
    int n;
public:

    vector <shared_ptr<Value> > weights;
    shared_ptr<Value> bias;

    Neuron(int n){
        this -> n = n;
        weights.resize(n);

        for(int i = 0; i < n; ++i){
            weights[i] = make_shared<Value>(dist(gen));
        }

        bias = make_shared<Value>(dist(gen));

        // cout << "weights : \n";


        // for(auto ele : weights){
        //     cout << ele -> _data << ' ';
        // }

        // cout << "Bias : \n";

        // cout << bias -> _data << "\n";
    }

    shared_ptr<Value> operator()(const vector < shared_ptr<Value> > &x){

        // w[i] * x[i] + bias



        auto out = make_shared<Value>(0.0);

        for (int i = 0; i < n; ++i){
            out = out + this -> weights[i] * x[i];
        }

        out = out + this -> bias;
        out = out -> tanh();

        return out;
    }

    vector < shared_ptr<Value> > parameters(){

         vector < shared_ptr<Value> > params;

         for(int i = 0; i < n; ++i){
             params.push_back(weights[i]);
         }
         params.push_back(bias);

         return params;
    }
};
