
//
// Created by Saluvaji Vishal on 07/08/25.
//

#pragma once
#include "Layer.hpp"

using namespace std;

class MLP{
private:
    vector < Layer > layers;
    int n;
public:
    MLP(int input_size, vector < int > layers_size){

        this -> n = layers_size.size();    // including output layer

        vector < int > sz;
        sz.push_back(input_size);

        for(auto &layer_size : layers_size){
            sz.push_back(layer_size);
        }

        for(int i = 0; i < layers_size.size(); ++i){
            layers.emplace_back(sz[i], sz[i + 1]);
        }
    }

    vector<shared_ptr<Value> > operator()(vector < shared_ptr<Value> > x){

        for(int i = 0; i < n; ++i){
            x = layers[i](x);
        }

        return x;
    }

    vector<shared_ptr<Value> > parameters(){

        vector<shared_ptr<Value> > params;

        for(int i = 0;i < n; ++i){
            auto layer_params = layers[i].parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }

        return params;
    }
};