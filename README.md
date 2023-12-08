# Automated News Categorization Using Natural Language Processing Techniques

kaggle link : https://www.kaggle.com/datasets/rmisra/news-category-dataset/code


In the era of digital information abundance, the proliferation of online news articles poses a significant challenge for users seeking specific and relevant content. The sheer volume of available news content often leads to information overload, making it difficult for individuals to navigate and locate articles aligned with their interests. To address this challenge, this project employs classification of news, a robust data mining technique, to efficiently categorize news articles.

The primary goal is to utilize classification algorithms to create distinct groups of news articles based on their content. This categorization approach aims to streamline and personalize the user experience by facilitating easier access to articles tailored to individual preferences. By effectively grouping news articles, this project endeavors to not only ensure accurate content categorization but also to enhance user interaction with digital news platforms.

## Installation

This project is in three files:

1. **Preprocessing:**
   - The 'News_Category_Dataset_v3.json' dataset undergoes comprehensive preprocessing: handling empty values, exploring and filtering data, balancing the dataset, and applying text preprocessing. The TF-IDF transformation prepares the data for model training. Dependencies include Python, Pandas, NLTK, and Scikit-learn. The 'Preprocessing.py' script executes these steps, producing model-ready data.
   - Important: Before running "Preprocessing.py," make sure to execute the `requirements.txt` file in sudo mode.

2. **Data Processing + Modelling:**
   - Same preprocessing steps have been performed for the below 
   - The "Modelling_Testing_Training.ipynb" file, within this folder, utilizes the "News_Category_Dataset_v3.json" inputs to construct various classifier models. Specifically, it generates or classifies into 10 classes with the train ansd test ratio 80:20.
   - The "Bayesian_tuned.ipynb" file, within this folder, utilizes the "News_Category_Dataset_v3.json" inputs to construct various classifier models. Specifically, it generates or classifies into 10 classes with the hyper parameter tuning
   - Note: The models have been pre-trained due to the time-consuming nature of building three models on around 200,000 articles, a process taking approximately 3-4 hours on a standard Colab Pro machine. The "combined_models.joblib" file in the results folder contains these pre-trained models, serving as a reference. No need to run these files for outputs; they're included for informational purposes.

3. **Results:**
   - This folder contains all the necessary files and packages for smooth execution during the evaluation phase.


## Code Run Sequence (it takes longer time to execute models So we  can directly run ## Testing for news a article below )

Follow these steps to run the code:

1. **Install Dependencies:**
   - Open the terminal and navigate to the "Preprocessing" directory.
   - Run the following command to install dependencies:
     ```bash
     pip3 install -r requirements.txt
     ```
2. **DATA PREPROCESSING**
   - Stay in the same directory and run the command
      ```bash
     python3 data_preprocessing.py
     ```
     you will get the tfidf_vectorizer.joblib file 

3. **Model Training:** (OPTIONAL STEP: BECAUSE IT TAKES LONGER TIME LIKE 3-4 HOURS)
   - Navigate to the "Modelling" directory.
   - Open and run the "Modelling_Testing_Training.ipynb" file, ensuring it has access to the "News_Category_Dataset_v3.json" file. So that you will get the models in the form of joblib files along with their TFIDF Vectorizer.

Note: The provided sequence assumes that you are using Python 3. Make sure your environment is set up correctly, and all required dependencies are installed before running the code.

## Testing for a News Article

To test a news article, follow these steps:

1. **Navigate to Results Directory:**
   - Open the terminal and go to the "Results" directory.

2. **Run Single_Domain_Prediction.sh:**
   - Execute the following command:
     ```bash
     ./shellfile.sh
     ```
   - This script will prompt you to enter the article where you can give any recent article with the message "Enter your news article:"

3. **Enter News Article:**
   - Input the article to test the category of the news.

4. **View Prediction:**
   - The script will provide the prediction using the stored model "combined_model.joblib" and give the prediction of the three models.

Note: Ensure that your environment is set up correctly, and you have followed the installation steps before testing a news article.

Note: Download the required files from Google Drive:

Download Models pickle file :  

Download  Models pickle file from drive: https://drive.google.com/file/d/1GSOi4jr37PDv9H_NxpLvYXfuibdn9mdj/view?usp=drive_link

Download TFIDF pickle file from drive : https://drive.google.com/file/d/1DELpHbAwwC8FeNMcZM1VsszTpTuztbX_/view?usp=drive_link 

and place these files in the Results folder


Ensure that both files are in your "Results" folder




Command : ./shellfile.sh

Note: it will run the script in the virtual environment


3. View Prediction
The model takes in news as an input and then classifies the given news category among the 10 different classes the model is trained on.


Work Distribution:

Dharnidhar Reddy banala –  Model Development, Feature Engineering, Report
Pavan Kumar – data Collection, Data Preprocessing, Presentation ppts
Nithish  -  Feature engineering, Model development , Presentation, Report

