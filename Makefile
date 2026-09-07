CXX := g++
CXXFLAGS := -std=c++17 -g -Wall -IAutograd

SRCS := Autograd/Value.cpp Autograd/Neuron.cpp Autograd/Layer.cpp Autograd/MLP.cpp Autograd/test.cpp
TARGET := Autograd/test

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)
