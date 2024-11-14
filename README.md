CODE -01
This code evaluates the performance of a spam detection model and visualizes its effectiveness. Here’s a summary of each step:
Imports Libraries: Loads necessary libraries for data manipulation, visualization, and model evaluation:
numpy, seaborn, and matplotlib.pyplot for numerical and visual tasks.
sklearn.metrics for calculating performance metrics like confusion matrix, accuracy, precision, recall, and F1 score.
Data Loading and Preparation:
Reads a dataset file (spam.csv) containing email messages and their categories (e.g., "ham" for non-spam and "spam").
Extracts the message content (Message) and category labels (Category).
Model Prediction:
Uses a previously trained model (classifier) to predict labels on a test dataset (x_test) and generates predicted labels (y_pred).
Confusion Matrix Visualization:
Computes a confusion matrix comparing actual labels (y_test) and predicted labels (y_pred).
Displays this matrix as a heatmap, helping to visualize the model’s performance by showing the count of true positives, true negatives, false positives, and false negatives.
Performance Metrics Calculation:

Calculates and displays common classification metrics:
Accuracy: Proportion of correct predictions.
Precision: Proportion of true spam messages among those predicted as spam.
Recall: Proportion of actual spam messages correctly identified.
F1 Score: Harmonic mean of precision and recall, providing a balance between the two.
This summary gives a snapshot of how well the spam classifier performs on the test dataset, with accuracy, precision, recall, and F1 score shown as percentages.


CODE -02
This code builds, trains, and evaluates a spam classification model using machine learning. Here's a breakdown of each part:

1. **Library Imports and NLTK Data Download**:
   - Imports necessary libraries, including NLTK for text processing and Scikit-learn for machine learning. It also downloads the "punkt" tokenizer to handle text tokenization.

2. **Data Loading and Preparation**:
   - Loads the dataset (`spam.csv`) containing a `Message` column with email text and a `Category` column with labels ("ham" for non-spam and "spam" for spam).

3. **Text Preprocessing**:
   - Defines a function `preprocess` that:
     - Filters out non-alphabetic characters from each message.
     - Tokenizes and stems each word using Lancaster stemming.
   - Applies the `preprocess` function to the email messages.

4. **Vectorization**:
   - Uses `TfidfVectorizer` to convert the preprocessed text into numerical vectors, making the data suitable for machine learning. It removes common English stop words.

5. **Label Encoding**:
   - Encodes "ham" as 0 and "spam" as 1 to prepare the labels for classification.

6. **Data Splitting**:
   - Splits the dataset into training and testing sets (80% for training and 20% for testing).

7. **Model Training**:
   - Trains a Support Vector Machine (SVM) classifier on the training data.

8. **Model and Vectorizer Saving**:
   - Saves the trained model and the TF-IDF vectorizer to a file (`training_data.pkl`) using `pickle` for future use.

9. **Message Classification Function**:
   - Defines `classify_message`, a function that preprocesses a new message, vectorizes it, and classifies it as either "spam" or "ham".

10. **Model Evaluation**:
    - Evaluates the model on the test set and prints out the accuracy, precision, recall, and F1 score to assess performance.

11. **Example Message Classification**:
    - Tests the model on a sample message and prints the classification result.

In summary, this code builds a complete spam detection pipeline, from data preprocessing to model training, evaluation, saving, and future message classification.


CODE -03
This code builds, trains, evaluates, and tests an SVM-based spam classifier. Here’s a breakdown of the major steps:

1. **Library Imports and Dataset Loading**:
   - Imports necessary libraries, including Pandas, Seaborn, and Scikit-learn for machine learning tasks.
   - Loads the `spam.csv` dataset, containing email messages and their labels (spam or ham).

2. **Data Preparation**:
   - Separates the features (`Message` column) and labels (`Category` column) and encodes labels as 0 for "ham" and 1 for "spam".

3. **Text Vectorization**:
   - Uses `TfidfVectorizer` to convert the text messages into TF-IDF vectors, removing common English stop words and setting a maximum document frequency of 0.9.

4. **Data Splitting**:
   - Splits the dataset into training and testing sets with an 80/20 split.

5. **Model Training**:
   - Initializes and trains a Support Vector Machine (SVM) classifier with a linear kernel on the training data.

6. **Saving the Model and Vectorizer**:
   - Saves the trained SVM classifier and TF-IDF vectorizer to a file (`svm_spam_model.pkl`) using `pickle` for future use.

7. **Model Evaluation**:
   - Predicts the labels for the test set and computes a confusion matrix.
   - Visualizes the confusion matrix as a heatmap, showing the true positives, true negatives, false positives, and false negatives.

8. **Performance Metrics**:
   - Calculates and prints the accuracy, precision, recall, and F1 score for the model on the test set.

9. **Message Classification Function**:
   - Defines `classify_message`, a function to classify a new message as "spam" or "ham". It preprocesses the message using the TF-IDF vectorizer and predicts its category using the trained SVM model.

10. **Example Classification**:
   - Classifies a sample message and prints the result.

In summary, this code provides a complete spam detection pipeline, from loading and preprocessing data to training, saving, evaluating the model, and classifying new messages.
