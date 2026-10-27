#Step	Code	কাজ
#1	X, y	Input ও Target
#2	W1, W2, b1, b2	Weight ও Bias
#3	np.dot()	Weighted Sum
#4	sigmoid()	Activation
#5	z1, a1, z2, a2	Forward Propagation
#6	loss	Error Calculation
#7	delta_output, delta_hidden	Backpropagation
#8	dW1, dW2	Gradient
#9	W1 -=, W2 -=	Weight Update
#10	for epoch	Repeated Training
#11	accuracy	Evaluation
#12	new_X	New Data Prediction


import numpy as np

# ==========================================
# STEP 1: Input Data
# ==========================================



X = np.array([[1.0, 0.0]])
y = np.array([[1.0]])

print("Input:", X)
print("Target:", y)


# ==========================================
# STEP 2: Weight and Bias Initialization
# ==========================================

# Input Layer -> Hidden Layer
W1 = np.array([
    [0.5, 0.3],
    [0.4, 0.2]
])

b1 = np.array([[0.1, 0.1]])

# Hidden Layer -> Output Layer
W2 = np.array([
    [0.6],
    [0.5]
])

b2 = np.array([[0.1]])

learning_rate = 0.1

print("\nInitial Weights:")
print("W1:\n", W1)
print("W2:\n", W2)


# ==========================================
# STEP 3: Weighted Sum Calculation
# ==========================================

def weighted_sum(X, W, b):
    return np.dot(X, W) + b


# ==========================================
# STEP 4: Activation Function
# ==========================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


# ==========================================
# STEP 5: Forward Propagation
# ==========================================

# Hidden Layer
z1 = weighted_sum(X, W1, b1)
a1 = sigmoid(z1)

# Output Layer
z2 = weighted_sum(a1, W2, b2)
a2 = sigmoid(z2)

prediction = a2

print("\nForward Propagation:")
print("Hidden Z:", z1)
print("Hidden Activation:", a1)
print("Output Z:", z2)
print("Prediction:", prediction)


# ==========================================
# STEP 6: Loss Calculation
# ==========================================

loss = 0.5 * np.mean((y - prediction) ** 2)

print("\nLoss:", loss)


# ==========================================
# STEP 7: Backpropagation
# ==========================================

# Output Layer Delta
delta_output = (
    (prediction - y) *
    sigmoid_derivative(prediction)
)

# Hidden Layer Delta
delta_hidden = (
    delta_output @ W2.T *
    sigmoid_derivative(a1)
)

print("\nBackpropagation:")
print("Output Delta:", delta_output)
print("Hidden Delta:", delta_hidden)


# ==========================================
# STEP 8: Gradient Calculation
# ==========================================

dW2 = a1.T @ delta_output
db2 = delta_output

dW1 = X.T @ delta_hidden
db1 = delta_hidden

print("\nGradients:")
print("dW1:\n", dW1)
print("db1:\n", db1)
print("dW2:\n", dW2)
print("db2:\n", db2)


# ==========================================
# STEP 9: Weight and Bias Update
# ==========================================

W1 = W1 - learning_rate * dW1
b1 = b1 - learning_rate * db1

W2 = W2 - learning_rate * dW2
b2 = b2 - learning_rate * db2

print("\nUpdated Weights:")
print("W1:\n", W1)
print("b1:\n", b1)
print("W2:\n", W2)
print("b2:\n", b2)


# ==========================================
# STEP 10: Iteration / Epoch
# ==========================================

epochs = 1000

for epoch in range(epochs):

    # Forward Propagation
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2
    prediction = sigmoid(z2)

    # Loss
    loss = 0.5 * np.mean((y - prediction) ** 2)

    # Backpropagation
    delta_output = (
        (prediction - y) *
        sigmoid_derivative(prediction)
    )

    delta_hidden = (
        delta_output @ W2.T *
        sigmoid_derivative(a1)
    )

    # Gradients
    dW2 = a1.T @ delta_output
    db2 = delta_output

    dW1 = X.T @ delta_hidden
    db1 = delta_hidden

    # Update Parameters
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}, Loss: {loss:.6f}"
        )


# ==========================================
# STEP 11: Model Evaluation
# ==========================================

final_prediction = sigmoid(
    sigmoid(X @ W1 + b1) @ W2 + b2
)

predicted_class = (
    final_prediction >= 0.5
).astype(int)

accuracy = np.mean(
    predicted_class == y
) * 100

print("\nModel Evaluation:")
print("Final Prediction:", final_prediction)
print("Predicted Class:", predicted_class)
print("Actual Class:", y)
print("Accuracy:", accuracy, "%")


# ==========================================
# STEP 12: Prediction / Inference
# ==========================================

new_X = np.array([[0.0, 1.0]])

hidden = sigmoid(new_X @ W1 + b1)

new_prediction = sigmoid(hidden @ W2 + b2)

new_class = (new_prediction >= 0.5).astype(int)

print("\nNew Data Prediction:")
print("New Input:", new_X)
print("Prediction Probability:", new_prediction)
print("Predicted Class:", new_class)
