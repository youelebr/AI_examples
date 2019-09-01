#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float sigmoid (float x) {
	return 1/(1+exp(-x));
}

float sigmoid_der (float x) {
	return sigmoid(x)*(1-sigmoid(x));
}

int* transpose(int* matrix, int x_size, int y_size) {
	int * trans = (int*)malloc(x_size*y_size*sizeof(int));
	for (int i = 0; i < x_size; i++)
    	for(int j = 0 ; j < y_size ; j++)
         	trans[j*x_size+i] = matrix[i*y_size+j];

    return trans;
}

int main(int argc, char const *argv[])
{
	int* feature_set = (int*)malloc(5*3*sizeof(int));
	feature_set[0] = 0; feature_set[1] = 1; feature_set[2] = 0; 
	feature_set[3] = 0; feature_set[4] = 0; feature_set[5] = 1;
	feature_set[6] = 1; feature_set[7] = 0; feature_set[8] = 0;
	feature_set[9] = 1; feature_set[10]= 1; feature_set[11]= 0;
	feature_set[12]= 1; feature_set[13]= 1; feature_set[14]= 1;

	int* label = (int*)malloc(5*sizeof(int));
	label[0]=1;
	label[1]=0;
	label[2]=0;
	label[3]=1;
	label[4]=1;
	time_t t = 40000;
	srand(time(NULL));
	float * weights = (float*)malloc(3*sizeof(float));
	
	weights[0] = rand() / RAND_MAX; // 0.37454012
	weights[1] = rand() / RAND_MAX; // 0.95071431
	weights[2] = rand() / RAND_MAX; // 0.73199394

	// weights[0] = 0.37454012;
	// weights[1] = 0.95071431;
	// weights[2] = 0.73199394;

	float bias = rand() / RAND_MAX;
	// float bias = 0.59865848;
	float lr = 0.05;

  float * z = (float*)calloc(3,sizeof(float));
  float * error = (float*)calloc(3,sizeof(float));
	
	int * inputs = (int*)malloc(5*3*sizeof(int));
  float * XW = (float*)malloc(5 * sizeof(float));
	
	float* dcost_dpred;
	// float dpred_dz[5];
	float * dpred_dz = (float*)calloc(5, sizeof(float));
	// float z_delta[5];
	float * z_delta = (float*)calloc(5, sizeof(float));

	for (int nb_oftraining=0; nb_oftraining < 20000; nb_oftraining++) {
    	for (int i = 0; i < 5; i++) {
    			XW[i] = 0.0f;
        	for (int j = 0; j < 3; j++) {
        		XW[i] += feature_set[i*3+j]*weights[j];
        	}
        	XW[i] += bias;
	    	z[i] = sigmoid(XW[i]);
	    	error[i] = z[i] - label[i];
    	}

		// Error sum
		float sum =0.0f;
		for (int i=0; i < 5; i++) {
			sum += error[i];
		}
		printf("errorsum = [%f]\n", sum);


		dcost_dpred = error;
		for (int i=0; i < 5; i++) {
			dpred_dz[i] = sigmoid_der(z[i]);
			z_delta[i] = dcost_dpred[i] * dpred_dz[i];
		}
		inputs = transpose(feature_set, 5,3);

		for (int i=0; i < 3; i++) {
			float dot = 0.0f;
			for (int j=0; j < 5; j++) {
				dot += inputs[i*5+j]*z_delta[j];
			}
			weights[i] -= lr * dot;
		}

		for (int num = 0; num < 5; num++) {
			bias -= lr * z_delta[num];
		}

	}

// Let's suppose we have a record of a patient that comes in who smokes, is not obese, and doesn't exercise. 
// Let's find if he is likely to be diabetic or not. The input feature will look like this: [1,0,0].
// results will be [0.00707584]
// You can see that the person is likely not diabetic since the value is much closer to 0 than 1.
	int * single_point = (int*)malloc(3*sizeof(int));
	single_point[0]=1;single_point[1]=0;single_point[1]=0;
	
	float dot = 0.0f;
	for (int i=0; i < 3; i++) {
		dot += single_point[i]*weights[i];	
	}
	float result = sigmoid(dot + bias);
	printf("1st Result %f\n", result);

// Now let's test another person who doesn't, smoke, is obese, and doesn't exercises. 
// The input feature vector will be [0,1,0].
// result will be [0.99837029]
// You can see that the value is very close to 1, which is likely due to the person's obesity.
	single_point[0]=0;
	single_point[1]=1;
	single_point[2]=0;

	dot = 0.0f;
	for (int i=0; i < 3; i++) {
		dot += single_point[i]*weights[i];	
	}
	result = sigmoid(dot + bias);
	printf("2de Result %f\n", result);

printf("---------------------------------\n");

	return 0;
}