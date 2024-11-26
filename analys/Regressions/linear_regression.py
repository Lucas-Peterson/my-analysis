import random
import matplotlib.pyplot as plt


# 1. Generating random points (Refer to section "Independent Variable x")
def generate_data(num_points):
    # Generate random x values
    x = [random.uniform(0, 10) for _ in range(num_points)]
    # Generate y values based on the linear relationship y = 2x + 3 with added random noise
    y = [2 * xi + 3 + random.uniform(-1, 1) for xi in x]  # y = 2x + 3 + noise
    return x, y


# 2. Function to calculate the mean (Supporting "Dependent Variable y")
def mean(values):
    return sum(values) / len(values)


# 3. Calculating coefficients b0 (intercept) and b1 (slope)
# (Refer to sections "Slope Coefficient b1" and "Intercept b0")
def linear_regression(x, y):
    n = len(x)

    # Mean values for x and y
    mean_x = mean(x)
    mean_y = mean(y)

    # Calculating the numerator and denominator to find the slope coefficient b1
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

    # Slope coefficient b1
    b1 = numerator / denominator

    # Intercept b0
    b0 = mean_y - b1 * mean_x

    return b0, b1


# 4. Predicting y values based on the found coefficients
# (Refer to section "Predicted Values ŷ")
def predict(x, b0, b1):
    return [b0 + b1 * xi for xi in x]


# 5. Generating random data
# (Supporting "Independent Variable x" and "Dependent Variable y")
x, y = generate_data(100)

# 6. Finding the linear regression coefficients
# (Refer to sections "Slope Coefficient b1" and "Intercept b0")
b0, b1 = linear_regression(x, y)

# 7. Predicting the y values
# (Refer to section "Predicted Values ŷ")
predicted_y = predict(x, b0, b1)

# 8. Printing the coefficients
print(f"Found coefficients: b0 = {b0}, b1 = {b1}")

# 9. Plotting the real data points and the regression line
# (Refer to section "Predicted Values ŷ")
plt.scatter(x, y, color='blue', label='Data points')  # Real data points
plt.plot(x, predicted_y, color='red', label='Regression line')  # Regression line

# Adding labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()

# 10. Display the plot (Visualizing the relationship between x and y)
plt.show()

# Outputting the first 5 real and predicted values for clarity (Refer to "Error (Residuals)")
for i in range(5):
    print(f"x: {x[i]:.2f}, y (real): {y[i]:.2f}, y (predicted): {predicted_y[i]:.2f}")
