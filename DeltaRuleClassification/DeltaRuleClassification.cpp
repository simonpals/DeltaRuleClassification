
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>

//Define constants
#define N 3
#define LEARN_RATE (double)20.0
#define MAX_ITERATION 1000000

struct TraintRange
{
    int begTrainClass1;
    int endTrainClass1;
    int begTrainClass2;
    int endTrainClass2;
    int begTestClass1;
    int endTestClass1;
    int begTestClass2;
    int endTestClass2;
};

class Perceptron
{
    //weights
    float weights[N];

public:
    Perceptron()
    {
        //Intialize weights
        weights[0] = randomFloat();
        weights[1] = randomFloat();
        weights[2] = randomFloat();
    }

    float randomFloat() {
        return ((float)rand() / (float)(RAND_MAX)) * 100.00;
    }

    void print()
    {
        printf("D = %5.3f + (%5.3fx) + (%5.3fy)\n", weights[0], weights[1], weights[2]);
    }

    int getClass(int x, int y)
    {
        return getSign(weights, x, y);
    }

    //Checks for misclassification.
    int getError(float weights[], int types[], int x[], int y[], int min, int max) 
    {
        int error = 0;
        for (int i = min; i < max; i++)
            if (types[i] != getSign(weights, x[i], y[i]))
                error++;

        return error;
    }

    //Returns sign of equation.
    int getSign(float weights[], int x, int y) 
    {
        float sum = weights[0] + x * weights[1] + y * weights[2];
        return (sum >= 0) ? 1 : -1;
    }

    //Updates the weights.
    void updateWeights(float weights[], int types[], int x[], int y[], int min, int max, int* updates) 
    {
        for (int i = min; i < max; i++)
            if (types[i] != getSign(weights, x[i], y[i])) {
                weights[0] += LEARN_RATE * types[i];
                weights[1] += LEARN_RATE * types[i] * x[i];
                weights[2] += LEARN_RATE * types[i] * y[i];
                (*updates)++;
            }

        return;
    }

    void train(int types[], int x[], int y[], TraintRange tr, int &updates, int &trainError, int &testError, int &itr)
    {
        //Check innitial training data error
        trainError = getError(weights, types, x, y, tr.begTrainClass1, tr.endTrainClass1);
        trainError += getError(weights, types, x, y, tr.begTrainClass2, tr.endTrainClass2);

        //PLA loop
        while (itr < MAX_ITERATION && trainError > 0) {
            //Update the weights
            updateWeights(weights, types, x, y, tr.begTrainClass1, tr.endTrainClass1, &updates);
            updateWeights(weights, types, x, y, tr.begTrainClass2, tr.endTrainClass2, &updates);

            //Update the training data error
            trainError = getError(weights, types, x, y, tr.begTrainClass1, tr.endTrainClass1);
            trainError += getError(weights, types, x, y, tr.begTrainClass2, tr.endTrainClass2);

            //Increment iterations
            itr++;
        }

        //Check test data error
        testError = getError(weights, types, x, y, tr.begTestClass1, tr.endTestClass1);
        testError += getError(weights, types, x, y, tr.begTestClass2, tr.endTestClass2);
    }
};

//Start of program
int main() {
    //Variables (int x1, y1... index 1 mean first perceptron and 2 second)
    Perceptron p1;
    Perceptron p2;
    TraintRange tr1;
    TraintRange tr2;

    int x1[80];
    int y1[80];
    int types1[80];
    float weights1[N];
    
    int x2[80];
    int y2[80];
    int types2[80];
    float weights2[N];

    int updates1 = 0;
    int trainError1 = 0;
    int testError1 = 0;
    int itr1 = 0;

    int updates2 = 0;
    int trainError2 = 0;
    int testError2 = 0;
    int itr2 = 0;

    //Seed number generator
    srand(time(NULL));

    //Data for first perceptron
    //Class A x[1:100] y[1:100]
    for (int i = 0; i < 40; i++) {
        x1[i] = rand() % (100) + 1;
        y1[i] = rand() % (100) + 1;
        types1[i] = 1;
    }

    //Class B x[-100:0] y[1:100]
    for (int i = 40; i < 80; i++) {
        x1[i] = rand() % (101) - 100;
        y1[i] = rand() % (100) + 1;
        types1[i] = -1;
    }

    //Data for second perceptron
    //Class A x[1:100] y[1:100]
    for (int i = 0; i < 40; i++) {
        x2[i] = x1[i];
        y2[i] = y1[i];
        types2[i] = 1;
    }

    //Class C x[101:200] y[101:200]
    for (int i = 40; i < 80; i++) {
        x2[i] = rand() % (100) + 101;
        y2[i] = rand() % (100) + 101;
        types2[i] = -1;
    }


    //Display starting weights
    printf("Perceptron 1 starting equation\n");
    p1.print();
    printf("Perceptron 2 starting equation\n");
    p2.print();

    tr1.begTrainClass1 = 0;
    tr1.endTrainClass1 = 25;
    tr1.begTrainClass2 = 40;
    tr1.endTrainClass2 = 65;
    tr1.begTestClass1 = 25;
    tr1.endTestClass1 = 40;
    tr1.begTestClass2 = 65;
    tr1.endTestClass2 = 80;

    tr2 = tr1;
   
    p1.train(types1, x1, y1, tr1, updates1, trainError1, testError1, itr1);
    p2.train(types2, x2, y2, tr2, updates2, trainError2, testError2, itr2);

    //Display final weights, iterations, updates, and error
    printf("\nFinal equation for perceptron 1\n");
    p1.print();
    printf("Iterations: %d\nTraining error: %d/%d\nTest error: %d/%d\n", itr1, trainError1, 50, testError1, 30);

    printf("\nFinal equation for perceptron 2\n");
    p2.print();
    printf("Iterations: %d\nTraining error: %d/%d\nTest error: %d/%d\n\n", itr2, updates2, trainError2, 50, testError2, 30);

    int xCoord;
    int yCoord;
    printf("Input x coordinate: ");
    scanf("%d", &xCoord);
    printf("Input y coordinate: ");
    scanf("%d", &yCoord);

    int c1 = p1.getClass(xCoord, yCoord);
    int c2 = p2.getClass(xCoord, yCoord);

    if (c1 == 1 && c2 == 1)
    {
        printf("It's class A x[1:100] y[1:100]");
    }
    else if (c1 == -1 && c2 == 1)
    {
        printf("It's class B x[-100:0] y[1:100]");
    }
    else if (c1 == 1 && c2 == -1)
    {
        printf("It's class C x[101:200] y[101:200]");
    }
    else
    {
        printf("Unknown class");
    }

    return 0;
}
