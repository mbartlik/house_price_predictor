# House Price Predictor

## Description
Given the price of a house and the values of a number of its attributes, the program can predict the prices of new houses given those attributes. The program is designed to take two text files as an input, one train file and one text file. By performing matrix operations on the train file the program produces a weight matrix that can be used to predict the price of the houses represented by the test matrix. 

## Calculation
Predicted Price : Y = W0 + W1*x1 + W2*X2 + W3*X3 + W4*X4
Weight Matrix : W = pseudoInv(X)*Y
pseudoInv(X) = inverse(transpose(X)*X) * transpose(X)  
weight(w) = pseudoInv(X) * Y
        where   X = Input data matrix
                Y = Target vector

## Input File Formatting
Example input files are given in the testcases folder. The first two lines of the train file should be the number of attributes and then the number of houses. Next each line are comma-separated values where all are the attribute values except the last, which is the price of that house. 

## Running
Here is an example of compiling and running the programs with some of the given testcases.
```bash
gcc -o example_compile price_predictor.c
./example_compile testcases/trainA.txt testcases/testA.txt
372862
446888
190019
606803
349252
230890
361327
260654
316616
497769
