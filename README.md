
# Naive Bayes

## 1. What is Naive Bayes?

Naive Bayes is a supervised Machine Learning algorithm primarily used for classification.

It is based on Bayes' theorem and assumes that all features are conditionally independent of each other given the target class.

It calculates the probability of a data point belonging to each class and predicts the class with the highest probability.

The independence assumption is called "naive" because real-world features are often related to each other.

## 2. Bayes' Theorem

Bayes' theorem calculates the probability of an event occurring given that another event has already occurred.

P(A|B) = P(B|A) × P(A) / P(B)

Where:

- P(A|B): Posterior probability — probability of A given B.
- P(B|A): Likelihood — probability of observing B given A.
- P(A): Prior probability — initial probability of A.
- P(B): Evidence — overall probability of observing B.

### Example

Suppose we want to determine whether an email is spam based on the presence of the word "offer".

Naive Bayes uses the probability of the word appearing in spam emails, the prior probability of spam, and the overall probability of the word appearing to estimate the probability that the email is spam.

## 3. How Naive Bayes Works

1. Calculate the prior probability of each class.
2. Calculate the likelihood of each feature given each class.
3. Apply Bayes' theorem to calculate the posterior probability.
4. Compare the probabilities across all classes.
5. Predict the class with the highest posterior probability.

For multiple features, Naive Bayes assumes conditional independence and multiplies their individual likelihoods.

## 4. Types of Naive Bayes

| Algorithm | Description | Common Use |
|---|---|---|
| Gaussian Naive Bayes | Assumes continuous features follow a Gaussian distribution | Numerical data |
| Multinomial Naive Bayes | Uses feature counts or non-negative frequencies | Text classification |
| Bernoulli Naive Bayes | Uses binary features indicating presence or absence | Spam detection |
| Complement Naive Bayes | Uses statistics from the complement of each class | Imbalanced text classification |

### Gaussian Naive Bayes

Assumes that continuous features follow a normal distribution within each class.

### Multinomial Naive Bayes

Commonly used for text classification, where features represent word counts or frequencies.

### Bernoulli Naive Bayes

Suitable for binary features, such as whether a particular word appears in an email.

## 5. Laplace Smoothing

Laplace smoothing is used to prevent zero probabilities when a feature has not appeared in the training data for a particular class.

Without smoothing, a single zero likelihood can make the entire probability product zero.

The smoothing parameter alpha controls the amount of smoothing.

- alpha = 1: Laplace smoothing.
- alpha > 0: Additive smoothing.
- alpha = 0: No smoothing.

In Scikit-learn, MultinomialNB uses alpha=1.0 by default.

## 6. Advantages and Disadvantages

### Advantages

- Simple and easy to implement.
- Fast training and prediction.
- Works well with high-dimensional data.
- Effective for text classification and spam detection.
- Requires relatively little training data.

### Disadvantages

- Assumes conditional independence between features.
- Performance may decrease when features are strongly related.
- Zero probabilities can occur without smoothing.
- Predicted probabilities may not always be well calibrated.

## 7. Python Implementation

### Gaussian Naive Bayes using the Iris Dataset

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create the model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## 8. Text Classification Example

Multinomial Naive Bayes is commonly used for spam detection and other text classification tasks.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample emails
emails = [
    "Win a free prize now",
    "Claim your free offer",
    "Meeting scheduled for tomorrow",
    "Please review the project report"
]

# Labels: 1 = Spam, 0 = Not Spam
labels = [1, 1, 0, 0]

# Create a text classification pipeline
model = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)

# Train the model
model.fit(emails, labels)

# Predict a new email
new_email = ["Win a free offer"]
prediction = model.predict(new_email)

print("Spam" if prediction[0] == 1 else "Not Spam")
```

## 9. Important Parameters

| Parameter | Description |
|---|---|
| alpha | Controls additive smoothing |
| fit_prior | Determines whether to learn class prior probabilities |
| class_prior | Allows manual specification of class priors |
| var_smoothing | Stabilizes variance calculations in GaussianNB |

## 10. Key Takeaways

- Naive Bayes is a probabilistic supervised learning algorithm.
- It is based on Bayes' theorem.
- It assumes conditional independence between features.
- It is commonly used for text classification and spam detection.
- GaussianNB is suitable for continuous numerical features.
- MultinomialNB is suitable for count-based features.
- Laplace smoothing helps prevent zero probabilities.

---

**Next:** Experiment with GaussianNB, MultinomialNB, and BernoulliNB to understand how different feature distributions affect classification performance.
