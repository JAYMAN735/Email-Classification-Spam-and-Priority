# Email Classification: Spam Detection and Priority Prediction

This project builds a two-stage email classification system using
Natural Language Processing (NLP) and machine learning.

The system first determines whether an email is **Spam** or **Ham (not
spam)**. If the email is classified as Ham, a second model predicts its
priority as **Low**, **Medium**, or **High**.

## Project Overview

The notebook implements the following workflow:

1.  Load and combine multiple email datasets for spam/ham
    classification.
2.  Clean and preprocess email text.
3.  Remove duplicate and highly similar emails.
4.  Convert email text into numerical features using TF-IDF.
5.  Train and evaluate several machine-learning classifiers for spam
    detection.
6.  Train and evaluate classifiers for three-class email priority
    prediction.
7.  Combine the selected classifiers into a single prediction pipeline.
8.  Provide an interactive email input interface using `ipywidgets`.

## System Workflow

``` text
Input Email
    |
    v
Text Preprocessing
(URL removal, special-character cleanup, lowercase,
 stopword removal, lemmatization)
    |
    v
TF-IDF Vectorization
    |
    v
Spam/Ham Classifier
    |
    +---- Spam ----> "Spam Email"
    |
    +---- Ham -----> Priority Classifier
                         |
                         +--> Low
                         +--> Medium
                         +--> High
```

## Datasets

The notebook loads the following files from `/content/` (the paths used
in Google Colab):

  File                         Role in the notebook
  ---------------------------- -----------------------------------------
  `completeSpamAssassin.csv`   Spam/ham email data
  `spam.xlsx`                  Additional spam/ham email data
  `spam_or_not_spam.csv`       Additional spam/ham email data
  `emails.csv`                 Email data used for priority prediction

The notebook combines the first three datasets for spam/ham
classification. It renames relevant columns to a common structure
(`Body` and `Label`) and maps the spam/ham labels to numeric values.

**Label convention used:** `0 = Ham`, `1 = Spam`.

For priority prediction, the notebook uses the `priority` column in
`emails.csv` and maps the labels as follows:

  Priority     Encoded value
  ---------- ---------------
  Low                      0
  Medium                   1
  High                     2

The dataset files are not embedded in this repository by the notebook
itself. Place the files in the expected location or update the notebook
paths before running it.

## Data Preprocessing

The notebook applies these text-cleaning operations:

-   Remove URLs.
-   Remove special characters and normalize whitespace.
-   Remove duplicate rows.
-   Remove emails containing fewer than 10 words.
-   Convert text to lowercase.
-   Remove English stopwords using NLTK.
-   Apply WordNet lemmatization.
-   Use cosine similarity on TF-IDF vectors to identify highly similar
    emails, with a similarity threshold greater than `0.8`.

## Feature Extraction

The notebook uses `TfidfVectorizer` with `max_features=5000` to convert
email text into TF-IDF features.

The same fitted vectorizer is then used to transform the email text for
both prediction tasks and for new email inputs.

## Machine Learning Models

### Spam/Ham classification

The notebook trains and evaluates:

-   Logistic Regression with balanced class weights
-   Standard Logistic Regression
-   Multinomial Naive Bayes
-   Decision Tree with balanced class weights
-   Random Forest with balanced class weights
-   XGBoost with a positive-class weight calculated from the training
    data

### Email priority classification

The notebook trains and evaluates:

-   Logistic Regression with balanced class weights
-   Standard Logistic Regression
-   Multinomial Naive Bayes
-   Decision Tree with balanced class weights
-   Random Forest with balanced class weights
-   XGBoost for multiclass classification

Both tasks use an 80/20 train-test split with `random_state=42` and
stratification. Model evaluation in the notebook uses accuracy and
`classification_report` (including precision, recall and F1-score).

The notebook does not declare a single best model based on a documented
model-selection rule. The integrated prediction pipeline uses: -
**Balanced Logistic Regression** for spam/ham classification. -
**Balanced Logistic Regression** for priority classification.

## Prediction Pipeline

The function `integrate_classification_pipeline(email_content)` performs
the complete prediction process:

1.  Preprocess the supplied email.
2.  Transform it with the fitted TF-IDF vectorizer.
3.  Predict Spam or Ham using the balanced Logistic Regression spam
    model.
4.  If the result is Spam, return `Spam Email`.
5.  If the result is Ham, predict its priority and return a result such
    as `Ham (Priority: High)`.

An `ipywidgets` text area and button are included in the notebook to let
a user enter an email and display the classification result. The final
`display(...)` calls are commented out in the notebook; uncomment them
to show the interface in a notebook environment.

## Requirements

The notebook uses Python and the following libraries:

-   `numpy`
-   `pandas`
-   `scikit-learn`
-   `nltk`
-   `xgboost`
-   `ipywidgets`
-   `openpyxl` (for reading the Excel dataset)

## How to Run

### Option 1: Google Colab

1.  Open the notebook in Google Colab.
2.  Upload the four dataset files to the Colab session, or place them in
    the paths expected by the notebook.
3.  Check that the file paths match the `/content/` paths in the code.
4.  Run the notebook cells from top to bottom.
5.  If NLTK resources are not already available, the notebook downloads
    `stopwords`, `wordnet`, and `punkt`.
6.  Run the final interface cell and uncomment the relevant
    `display(email_input, classify_button, output_widget)` line to use
    the interactive classifier.

### Option 2: Local Jupyter Notebook

1.  Clone or download this repository.
2.  Install the required libraries.
3.  Place the datasets in a local data folder.
4.  Update the notebook's `/content/...` dataset paths to the
    corresponding local paths.
5.  Open the notebook in Jupyter and run the cells from top to bottom.

Example installation command:

``` bash
pip install numpy pandas scikit-learn nltk xgboost ipywidgets openpyxl jupyter
```

## Repository Structure

A suggested repository structure is:

``` text
.
├── Email_Classification_Spam_And_Priority.ipynb
├── README.md
└── data/
    ├── completeSpamAssassin.csv
    ├── spam.xlsx
    ├── spam_or_not_spam.csv
    └── emails.csv
```

The `data/` directory is a suggested organization. The notebook
currently uses `/content/` paths, so those paths must be updated if you
adopt this structure.

## Important Notes and Limitations

-   The notebook expects external dataset files; it does not download
    them automatically.
-   The TF-IDF vectorizer is fitted before the train-test split in the
    notebook. For a leakage-safe evaluation, fit the vectorizer only on
    the training text (ideally inside a scikit-learn `Pipeline`) and
    transform the test text with that fitted vectorizer.
-   The cosine-similarity duplicate-removal step is performed before
    splitting. This can help remove near-duplicate records, but the
    similarity matrix can become memory-intensive for large datasets.
-   The priority model predicts labels learned from the `priority`
    column in `emails.csv`. Its reliability depends on the quality and
    consistency of those labels.
-   The notebook reports evaluation metrics when executed, but this
    README does not state numerical scores because fixed result values
    are not documented here.
-   The trained models and TF-IDF vectorizer are not saved as reusable
    model files in the notebook. The notebook trains them during
    execution.

## Future Improvements

-   Use a train-only preprocessing and TF-IDF fitting pipeline to
    prevent data leakage.
-   Compare models using class-wise precision, recall and F1-score, not
    accuracy alone.
-   Add confusion matrices for both classification tasks.
-   Tune model hyperparameters using cross-validation.
-   Save the fitted vectorizer and selected models with `joblib` or
    `pickle`.
-   Add a deployable interface, such as Streamlit or FastAPI, after
    validating the models.

## Author

**Jayman Prajapati**

------------------------------------------------------------------------

*This README describes the workflow implemented in the accompanying
Jupyter notebook. Dataset filenames, preprocessing steps, model families
and prediction behavior are based on the notebook code.*
