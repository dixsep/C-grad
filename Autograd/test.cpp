
#include "Value.hpp"

#include "MLP.hpp"

using namespace std;

// lhs + rhs
shared_ptr<Value> operator+(const shared_ptr<Value> &lhs, const shared_ptr<Value> &rhs){

    auto out = make_shared<Value>(lhs -> _data + rhs -> _data, initializer_list<shared_ptr<Value> >{lhs, rhs}, "+");

    out -> _backward = [lhs, rhs, out](){
        lhs -> _grad += out -> _grad;
        rhs -> _grad += out -> _grad;
    };

    return out;
}

// lhs + value
shared_ptr<Value> operator+(const shared_ptr<Value> &lhs, const double& value){

    auto rhs = make_shared<Value>(value);

    return operator+(lhs, rhs);
}

// value + lhs
shared_ptr<Value> operator+(const double& value, const shared_ptr<Value> &rhs){

    return operator+(rhs, value);
}

//lhs * rhs
shared_ptr<Value> operator*(const shared_ptr<Value> &lhs, const shared_ptr<Value> &rhs){

    auto out = make_shared<Value>(lhs -> _data * rhs -> _data, initializer_list<shared_ptr<Value> >{lhs, rhs}, "*");

    out -> _backward = [lhs, rhs, out](){
        lhs -> _grad += out -> _grad * rhs -> _data;
        rhs -> _grad += out -> _grad * lhs -> _data;
    };

    return out;
}

// lhs * value
shared_ptr<Value> operator*(const shared_ptr<Value> &lhs, const double& value){

    return operator*(lhs, make_shared<Value>(value));
}

// value * lhs
shared_ptr<Value> operator*(const double& value, const shared_ptr<Value> &rhs){

    return operator*(rhs, value);
}

//lhs - rhs
shared_ptr<Value> operator-(const shared_ptr<Value> &lhs, const shared_ptr<Value> &rhs){

    auto out = make_shared<Value>(lhs -> _data - rhs -> _data, initializer_list<shared_ptr<Value> >{lhs, rhs}, "-");

    out -> _backward = [lhs, rhs, out](){
        lhs -> _grad += out -> _grad;
        rhs -> _grad += out -> _grad * -1;
    };

    return out;
}

// lhs / rhs
shared_ptr<Value> operator/(const shared_ptr<Value> &lhs, const shared_ptr<Value> &rhs){

   // lhs / rhs
   // lhs * (power(rhs, -1))
   auto RHS = rhs->power(-1);
   return operator*(lhs, RHS);
}

// lhs / value
shared_ptr<Value> operator/(const shared_ptr<Value> &lhs, const double& value){

    return operator/(lhs, make_shared<Value>(value));
}

// value / rhs
shared_ptr<Value> operator/(const double& value, const shared_ptr<Value> &rhs){

    return operator/(make_shared<Value>(value),  rhs);
}

// negation
shared_ptr<Value> operator-(const shared_ptr<Value> &lhs){

    return operator*(lhs, -1);
}

void Value_check(){


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
    cout <<"G : " <<  g -> _data << '\n';

    cout << "---------Backward Pass-----------\n";
    g -> backward();

    cout << "dg/da : " << a -> _grad << '\n';
    cout << "dg/db : " << b -> _grad << '\n';
}

void mlp_check1(){

    MLP n = MLP(3, {4, 4, 1});   // layer1 : 4 , layer2 : 4, output : 1 (layer)

    vector < shared_ptr<Value> > x = {make_shared<Value>(2.0), make_shared<Value>(3.0), make_shared<Value>(5.0)};

    vector < shared_ptr<Value> > y = n(x);

    for(auto &ele : y){

        cout << ele -> _data << ' ';
    }
    
}

void mlp_check2(){

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

}

int main(){

    // Value_check();

    // mlp_check1();

    // mlp_check2();
}