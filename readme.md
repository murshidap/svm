 # confusion matrix
                 Predicted
                 Benign  Malignant
Actual Benign       72       0
Actual Malignant     3      39

# performance evaluation
accuracy: 97.37%
precision: 100%
recall: 92.86%
f1 score: 96.30%

# for different kernels
Kernel	Accuracy	Precision	Recall	F1
Linear	0.9648	0.9713	0.9353	0.9516
Poly	0.9011	1.0000	0.7353	0.8437
RBF	0.9736	0.9771	0.9529	0.9641

# best parameters:
 RBF, C=10, gamma=0.01


 ## C and Gamma:
    c:  controls the penalty for classification errors: lower values allow more errors
    gamma: controls how much influence each example has with RBF and polynomial kernels: lower values make smoother boundaries