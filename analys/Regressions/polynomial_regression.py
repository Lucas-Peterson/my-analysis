class Datamining:
    def __init__(self, train_set):
        self.x_values = [point[0] for point in train_set]
        self.y_values = [point[1] for point in train_set]
        
        self.coefficients = self.fit_polynomial(5)

    def fit_polynomial(self, degree):
        X = [[x ** d for d in range(degree + 1)] for x in self.x_values]
        
        Y = self.y_values
        
        X_T = self.transpose(X)
    
        X_T_X = self.matrix_multiply(X_T, X)
        
        X_T_Y = self.matrix_vector_multiply(X_T, Y)
        
        coefficients = self.solve_system(X_T_X, X_T_Y)
        
        return coefficients

    def predict(self, x):
        return sum(self.coefficients[i] * (x ** i) for i in range(len(self.coefficients)))
    
    def transpose(self, matrix):
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

    def matrix_multiply(self, A, B):
        result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    def matrix_vector_multiply(self, A, v):
        result = [0 for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(v)):
                result[i] += A[i][j] * v[j]
        return result

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
