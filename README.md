# Toxic Comment Classification

This project aims to classify toxic comments using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The dataset is preprocessed, visualized, and used to train a classification model.

## Dataset
The dataset is loaded from a CSV file (`train.csv`) and contains the following key columns:
- `comment_text`: The text of the comment.
- `label`: The classification label (toxic or non-toxic).

## Steps in the Notebook
### 1. Installing Dependencies
The necessary libraries are installed, including:
- `contractions`
- `wordcloud`
- `xgboost`
- `tensorflow`

### 2. Importing Libraries
Essential Python libraries such as `numpy`, `pandas`, `matplotlib`, `seaborn`, `nltk`, `sklearn`, `xgboost`, and `tensorflow` are imported for data processing and modeling.

### 3. Data Exploration
- The dataset is loaded using Pandas (`pd.read_csv('train.csv')`).
- `df.info()` and `df.describe()` provide an overview of data types and summary statistics.
- The length of each comment is analyzed to understand text distribution.
- Missing values are checked (`df.isna().sum()`).

### 4. Data Preprocessing
A `clean_text` function is created to:
- Convert text to lowercase.
- Remove numbers and special characters.
- Tokenize words.
- Remove stopwords using NLTK.
- Store the cleaned text in a new column (`df['cleaned_text']`).

### 5. Data Visualization
A word cloud is generated from the cleaned text data to visualize the most frequent words.

### 6. Feature Extraction (TF-IDF)
Text data is transformed into numerical features using TF-IDF (`TfidfVectorizer(max_features=5000)`).

### 7. Machine Learning Model
- The dataset is split into training and testing sets (`train_test_split`).
- A `RandomForestClassifier` is trained.
- The model is evaluated using `accuracy_score`.

## Results
The accuracy of the model is printed after testing.

## Future Improvements
- Try deep learning models (LSTM, BERT) for better performance.
- Use additional NLP techniques (lemmatization, named entity recognition).
- Tune hyperparameters to improve model accuracy.

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
Run the Jupyter Notebook to execute the steps.
```bash
jupyter notebook code.ipynb
```

