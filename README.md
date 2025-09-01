# 🧠 C-grad

A minimalistic, educational implementation of a scalar-based automatic differentiation engine and a small neural network library built on top of it — all in pure C++.
- Operates on **scalar values only**, meaning each neuron is broken down into primitive operations like tiny adds and multiplies.
- Despite its simplicity, this system can build and train full deep neural networks for classification / regression problems as well.
- Inspired from Andrej Karpathy's Micrograd

---

## ✨ Features

- **Autograd Engine**: Supports reverse-mode automatic differentiation.
- **Value Class**: Tracks operations and computes gradients.
- **Neural Network API**: Includes `Neuron`, `Layer`, and `MLP` abstractions.
- **Backpropagation**: Fully functional backward pass with graph traversal.
- **Educational**: Easy to understand and extend for learners.

---

## Example Usage
Below is a slightly contrived example showing a number of possible supported operations:

``` bash

    std::cout << "--- Simple Test ---" << std::endl;

    auto a = make_shared<Value>(-4.0);
    auto b = make_shared<Value>(2.0);

    auto c = a + b;
    auto d = a * b + b->power(3);

    c = c + c + 1;
    c = c + 1 + c + (-a);

    d = d + d * 2 + (b + a) -> relu();
    d = d + 3 * d + (b - a)->relu();

    auto e = c - d;
    auto f = e -> power(2);

    auto g = f / 2;
    g = g + 10/f;

    auto ee = a -> exp();
    auto tt = a -> tanh();
    auto rr = a -> relu();


    cout << "Forward Pass \n";
    cout <<"G : " <<  g -> _data << '\n';   // prints 24.7041,

    cout << "---------Backward Pass-----------\n";
    g -> backward();

    cout << "dg/da : " << a -> _grad << '\n';  // prints 138.8338, i.e. the numerical value of dg/da
    cout << "dg/db : " << b -> _grad << '\n';  // prints 645.5773, i.e. the numerical value of dg/db

```



## Training a Neural Net
The test.cpp provides a full demo of training an 2-layer neural network (MLP) with 2 hidden layers each of 4 nodes with sample inputs and desired outputs.This is achieved by initializing a neural net from MLP class and implementing a custom TanH activation function.

``` bash

        MLP n = MLP(3, {4, 4, 1});

    // sample training
    vector < vector < shared_ptr < Value> > > xs = {{make_shared<Value>(2.0), make_shared<Value>(3.0), make_shared<Value>(-1.0)}, 
{make_shared<Value>(3.0), make_shared<Value>(-1.0), make_shared<Value>(0.5)},
{make_shared<Value>(0.5), make_shared<Value>(1.0), make_shared<Value>(1.0)},
{make_shared<Value>(1.0), make_shared<Value>(1.0), make_shared<Value>(-1.0)}};

     vector < shared_ptr<Value> > ys = {make_shared<Value>(1.0), make_shared<Value>(-1.0), make_shared<Value>(-1.0), make_shared<Value>(1.0)};

     
    //optimisation
    double lr = 0.05;   // learning rate

    for (int step = 0; step < 5; ++step){

        //forward pass
        vector < shared_ptr<Value > > y_pred;

        for(int i = 0; i < xs.size(); ++i){
            auto x = xs[i];
            auto y = n(x)[0];
            y_pred.push_back(y);

        }

        auto loss = make_shared<Value>(0.0);

        for(int i = 0; i < xs.size(); ++i){
            loss = loss + (ys[i] - y_pred[i]) * (ys[i] - y_pred[i]);
        }

        //change grads to 0
        for(int i = 0; i < n.parameters().size(); ++i){
            n.parameters()[i] -> _grad = 0.0;
        }

        //backward pass
        loss -> backward();

        //update the parameters
        for(int i = 0; i < n.parameters().size(); ++i){
            n.parameters()[i] -> _data += lr * -n.parameters()[i] -> _grad;
        }

        cout << "Loss : " << loss -> _data << "\n";

        cout << "Predictions : \n";

        for(int i = 0; i < xs.size(); ++i){
            cout << y_pred[i] -> _data << " " << ys[i] -> _data << '\n';
        }
    }

```
