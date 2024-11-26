### In this repo I want to make my conspects about regressions



### Features
- Conspect about polynomial regression (this kata solution https://www.codewars.com/kata/591748b3f014a2593d0000d9 )








### How Each Element in Linear Regression Works

Linear regression is a method that models the relationship between an independent variable `x` and a dependent variable `y` using a linear equation of the form:


y = b_0 + b_1 * x



Where:
- `y` — predicted value (dependent variable),
- `x` — independent variable (the factor that influences `y`),
- `b_0` — intercept (the point where the line intersects the y-axis),
- `b_1` — slope coefficient (shows how `y` changes when `x` changes).

Now, let's look at each element of linear regression and its role.

### 1. **Independent Variable `x`**

This is the data we use to make predictions. It can be any quantity that influences the dependent variable `y`. Examples:
- House size (`x`) and its price (`y`),
- A person’s age (`x`) and their weight (`y`).

`x` is the input data that we observe or measure, and it’s used to predict `y`.

### 2. **Dependent Variable `y`**

This is the result we’re trying to predict based on `x`. For example, if `x` is the size of a house, `y` could be its price. The task of regression is to find the relationship between `x` and `y`, and use it to predict new values of `y` based on future values of `x`.

### 3. **Slope Coefficient `b_1`**

The coefficient `b_1` shows how much `y` changes when `x` changes. The formula is:



b_1 = Σ[(x_i - x̄)(y_i - ȳ)] / Σ[(x_i - x̄)^2]




- **Numerator**: `Σ[(x_i - x̄)(y_i - ȳ)]` — this is the covariance between `x` and `y`. It shows how `x` and `y` vary together. If the covariance is positive, it means that as `x` increases, `y` also tends to increase.
  
- **Denominator**: `Σ[(x_i - x̄)^2]` — this is the variance of `x`, which measures how much the values of `x` deviate from their mean. It normalizes the covariance, allowing us to get a coefficient that is proportional to the change in `y` with respect to `x`.

**Role**: `b_1` is the slope of the regression line. It defines how strongly `y` depends on `x`. A large value of `b_1` means a strong dependence of `y` on `x`.

### 4. **Intercept `b_0`**

The intercept `b_0` is the point where the regression line intersects the y-axis when `x = 0`. The formula is:



b_0 = ȳ - b_1 * x̄




- **Mean value `ȳ`** — this is the average of all observed values of `y`.
- **Mean value `x̄`** — this is the average of all observed values of `x`.

**Role**: `b_0` defines the starting point of the regression line. It is the value of `y` when `x` is zero. It is important for understanding the overall position of the line on the graph, but does not reflect how `y` changes relative to `x`.

### 5. **Error (residuals)**

The difference between the real values `y` and the predicted values `ŷ` is called the error (or residual):


Error = y_i - ŷ_i



The sum of the squared errors is minimized when constructing the regression line. Squaring the errors ensures that both positive and negative errors contribute to the total error equally.

**Role**: Errors show how well the model fits the data. The smaller the errors, the better the model. The method of least squares (used in linear regression) minimizes the sum of squared errors.

### 6. **Predicted Values `ŷ`**

Once we have found the coefficients `b_0` and `b_1`, we can use them to predict new values of `y` for any given `x` using the equation:


ŷ = b_0 + b_1 * x



**Role**: This equation allows us to predict values of `y` for new `x` values. We use the model for forecasting in new situations or on new data.

### Conclusion:

1. **Independent Variable `x`** — the data we use to make predictions.
2. **Dependent Variable `y`** — the value we are trying to predict.
3. **Slope Coefficient `b_1`** determines the strength and direction of the relationship between `x` and `y`.
4. **Intercept `b_0`** determines the starting point of the regression line.
5. **Error** shows the deviation between real and predicted values.
6. **Predicted Value `ŷ`** uses the found coefficients to compute new values of `y` based on `x`.

Thus, linear regression builds a straight line that best describes the relationship between the variables and is used for making predictions.






### Polynomial regression

I did it in another style, not like a previous conspect, sorry, if it's confused

### Class Constructor Datamining

def __init__(self, train_set):
    self.x_values = [point[0] for point in train_set]
    self.y_values = [point[1] for point in train_set]
    self.coefficients = self.fit_polynomial(5)
Description:
The constructor takes a training dataset (train_set), where each point consists of x and y values.
Arrays self.x_values and self.y_values are populated from the training set.
It calls the fit_polynomial(5) function to find the coefficients of a 5th-degree polynomial that will approximate the data.

### Method fit_polynomial(self, degree)


def fit_polynomial(self, degree):
    X = [[x ** d for d in range(degree + 1)] for x in self.x_values]
    Y = self.y_values
    X_T = self.transpose(X)
    X_T_X = self.matrix_multiply(X_T, X)
    X_T_Y = self.matrix_vector_multiply(X_T, Y)
    coefficients = self.solve_system(X_T_X, X_T_Y)
    return coefficients


It constructs a feature matrix X, where each row contains values of x raised from the 0th to the degree degree-th power.
Computes the transpose of matrix XT and performs matrix multiplications 𝑋𝑇 × X and XT ×Y 


### Method predict(self, x)

def predict(self, x):
    return sum(self.coefficients[i] * (x ** i) for i in range(len(self.coefficients)))

This method uses the computed polynomial coefficients to predict the y value for a given x.
It sums up values of coefficients[i] for each coefficient.

### Method transpose(self, matrix)

def transpose(self, matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

Transposes a matrix, which means it switches rows and columns



### Method matrix_multiply(self, A, B)

def matrix_multiply(self, A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result



Multiplies two matrices A and B and returns the result.
It uses loops over rows and columns to perform the matrix multiplication.



### Method matrix_vector_multiply(self, A, v)

def matrix_vector_multiply(self, A, v):
    result = [0 for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(v)):
            result[i] += A[i][j] * v[j]
    return result


Multiplies matrix 𝐴 by vector 𝑣 and returns the result.


### Method solve_system(self, A, b)

def solve_system(self, A, b):
    n = len(A)
    for i in range(n):
        factor = A[i][i]
        for j in range(i, n):
            A[i][j] /= factor
        b[i] /= factor
        for k in range(i + 1, n):
            factor = A[k][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
            b[k] -= factor * b[i]
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))
    return x


Solves a system of linear equations using Gaussian elimination (You can find more information about it here ---> [Tutorial](https://www.youtube.com/watch?v=eDb6iugi6Uk)
In the forward pass, it normalizes the current row and eliminates variables below the current row.
In the backward pass, it finds the solution of the system of equations.
