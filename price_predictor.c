/* 
 * 
 * This code calculates the house price of a house by learning from
 * training data. It uses pseudo inverse of a given matrix to find the 
 * weight of different features.
 * 
 * Predicted Price : Y = W0 + W1*x1 + W2*X2 + W3*X3 + W4*X4
 * Weight Matrix : W = pseudoInv(X)*Y
 * pseudoInv(X) = inverse(transpose(X)*X) * transpose(X)  
 * 
 * weight(w) = pseudoInv(X) * Y
 *          where   X = Input data matrix
 *                  Y = Target vector
 * 
 */
 
#include<stdio.h>
#include<stdlib.h>

// all methods declarations
double** multiplyMatrix(double **matA, double **matB, int r1, int c1, int r2, int c2);
double** transposeMatrix(double** mat, int row, int col);
double** inverseMatrix(double **matA, int dimension);

// function to free the memory of a two dimensional matrix
// input is a pointer to the matrix and the number of rows
void freeMatrix(double** mat, int rows) {
    int i;
    for(i = 0; i < rows; i = i + 1) {
        free(mat[i]);
    }
    free(mat);
}

// function to multiply two matrix
// inputs are the two matrices along with the number of rows and cols in each
// returns pointer to the result matrix
double** multiplyMatrix(double **matA, double **matB, int r1, int c1, int r2, int c2)
{
    double** result=malloc(r1*sizeof(double*)); // size r1 x c2

    int i, j, k;

    // give the result matrix rows
    for(i = 0; i < r1; i = i + 1) {
        result[i] = malloc(c2*sizeof(double));
    }
    double currentEntry;

    // loops through the result array to calculate each entry of it
    for(i = 0; i < r1; i = i + 1) {
        for(j = 0; j < c2; j = j + 1) {
            currentEntry = 0;
            for(k = 0; k < c1; k = k + 1){
                currentEntry = currentEntry + matA[i][k] * matB[k][j];
            }
            result[i][j] = currentEntry; // write to result
        }
    }

    return result;
}

// function to transpose a matrix
// input is pointer to a 2d matrix and the number of rows and cols in it
// returns a new transposed matrix
double** transposeMatrix(double** mat, int row, int col)
{

    double** matTran=malloc(col*sizeof(double*)); // allocate for a transposed matrix
    
    int i, j;

    // give the transposed matrix rows
    for(i = 0; i < col; i = i + 1) {
        matTran[i] = malloc(row*sizeof(double));
    }

    // for each entry (j,i) of the new matrix, set equal to (i,j) of the old matrix
    for(i = 0; i < row; i = i + 1) {
        for(j = 0; j < col; j = j + 1) {
            matTran[j][i] = mat[i][j];
        }
    }

    return matTran;        
}

// function to inverse matrix
// inputs are pointer to a matrix and its dimension
// return a new matrix that is the inverse
double** inverseMatrix(double **matA, int dimension) {
    
    // allocate space for the identity matrix
    double** matI=malloc(dimension*sizeof(double*)); 
    int i, j, k;

    // give the identity matrix rows
    for(i = 0; i < dimension; i = i + 1) {
        matI[i] = malloc(dimension*sizeof(double));
    }

    // set the identity matrix
    for(i = 0; i < dimension; i = i + 1) {
        for(j = 0; j < dimension; j = j + 1) {
            if(i == j) {
                matI[i][j] = 1;
            }
            else {
                matI[i][j] = 0;
            }
        }
    }


    double f;
    
    // 1st layer main loop stops at each pivot point
    for(i = 0; i < dimension; i = i + 1) {
        f = matA[i][i];
        // 2nd layer loop multiplies the row by 1/f
        for(j = 0; j < dimension; j = j + 1) {

            matA[i][j] = matA[i][j] * 1/f;
            matI[i][j] = matI[i][j] * 1/f;

        }

        // 2nd layer loop goes through and does row operations to make the elements below the pivot zero
        for(j = i + 1; j < dimension; j = j + 1) {
            f = matA[j][i];
            // 3rd layer loop goes through the row itself doing each operation
            for(k = 0; k < dimension; k = k + 1) {

                matA[j][k] = matA[j][k] - f * matA[i][k];
                matI[j][k] = matI[j][k] - f * matI[i][k];

            }
        }
    }

    // 1st layer loop iterates through each pivot point going backwards
    for(i = dimension - 1; i >= 0; i = i - 1) { 
        // 2nd layer loop iterates row by row above the pivot point
        for(j = i - 1; j >= 0; j = j - 1) {
            f = matA[j][i];
            //3rd layer loop iterates through the row itself
            for(k = 0; k < dimension; k = k + 1) {

                matA[j][k] = matA[j][k] - f * matA[i][k];
                matI[j][k] = matI[j][k] - f * matI[i][k];
            }
        }
    }
    
    return matI;
}


int main(int argc, char** argv){
    
    // read file
    char* filename = argv[1];
    FILE* file = fopen(filename, "r");
    if(file == NULL) {
        printf("Error");
        return 0;
    }

    // read in number of attributes and training examples
    int attributesCount;
    int trainingCount;
    fscanf(file, "%d\n", &attributesCount);
    fscanf(file, "%d\n", &trainingCount);

    // define X matrix
    double** x = malloc(trainingCount * sizeof(double*));

    // vector Y
    double** y = malloc(trainingCount * sizeof(double*));

    int i;
    int j;
    
    // make first column of X all ones and makes the rows of the array
    for(i = 0; i < trainingCount; i = i + 1) {
        x[i] = malloc((attributesCount + 1) * sizeof(double));
        x[i][0] = 1;
    }
    // give y rows
    for(i = 0; i < trainingCount; i = i + 1) {
        y[i] = malloc(sizeof(double));
    }


    // iterate through the text file and scan into the array
    for(i = 0; i < trainingCount; i = i + 1) {
        for(j = 1; j < attributesCount + 1; j = j + 1) {
            fscanf(file, "%lf, ", &x[i][j]);
        }
        fscanf(file, "%lf ", &y[i][0]);
    }

    // the following calculations are detailed at the top of the file
    double** x_transposed = transposeMatrix(x, trainingCount, attributesCount+1);
    
    double** xtx = multiplyMatrix(x_transposed, x, attributesCount+1, trainingCount, trainingCount, attributesCount+1);
    
    double** xtxInv = inverseMatrix(xtx, attributesCount + 1);
    
    double** pseudoInv = multiplyMatrix(xtxInv, x_transposed, attributesCount+1, attributesCount+1, attributesCount+1, trainingCount); 

    double** w = multiplyMatrix(pseudoInv, y, attributesCount+1, trainingCount, trainingCount, 1); 
    
    // find number of test data points
    int testCount;
    FILE* file2 = fopen(argv[2], "r");
    fscanf(file2, "%d\n", &testCount);

    // make test data matrix
    double** testData = malloc(testCount*sizeof(double*));
    for(i = 0; i < testCount; i = i + 1) {
        testData[i] = malloc((attributesCount+1)*sizeof(double));
    }

    // scan in the test data
    for(i = 0; i < testCount; i = i + 1) {
        for(j = 0; j < attributesCount; j = j + 1) {
            if(j == 0) {
                testData[i][j] = 1;
            }
            else {
                fscanf(file2, "%lf,", &testData[i][j]);
            }
        }
        fscanf(file2, "%lf\n", &testData[i][attributesCount]);
    }

    // allocate for the result matrix
    double** final = malloc(testCount*sizeof(double*)); //multiplyMatrix(w, testData, testCount, attributesCount + 1, attributesCount + 1, 1);
    for(i = 0; i < testCount; i = i + 1) {
        final[i] = malloc(sizeof(double));
    }

    // write into the final result matrix by multiplying the testdata with the weight matrix
    for(i = 0; i < testCount; i = i + 1) {
        for(j = 0; j < attributesCount + 1; j = j + 1) {
            final[i][0] += w[j][0]*testData[i][j];
        }
    }

    // print results
    for(i = 0; i < testCount; i = i + 1) {
        printf("%0.0lf\n",final[i][0]);
    }



    // free allocated memory
    freeMatrix(x, trainingCount);
    freeMatrix(y, trainingCount);
    freeMatrix(pseudoInv, attributesCount + 1);
    freeMatrix(x_transposed, attributesCount + 1);
    freeMatrix(xtx, attributesCount + 1);
    freeMatrix(xtxInv, attributesCount + 1);
    freeMatrix(w, attributesCount + 1);
    freeMatrix(testData, testCount);
    freeMatrix(final, testCount);

    fclose(file);
    fclose(file2);
    
    return 0;


}




