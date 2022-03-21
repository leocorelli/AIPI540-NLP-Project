# AIPI 540 NLP Project
**Christine Park, Miranda Morris, and Leo Corelli for Duke AIPI 540 Spring 2022**

Predicting 1-5 Star Ratings on [100,000 Coursera Course Reviews Dataset](https://www.kaggle.com/septa97/100k-courseras-course-reviews-dataset)

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/FiveStarsInline5.svg" width="400" /> 
</p>

# Instructions:
Type in your terminal:
1. `git clone https://github.com/leocorelli/AIPI540-NLP-Project.git`

Navigate to this new directory. Then:

2. Go to https://drive.google.com/file/d/1Oxbn7Cigf3tqRlmQAZfeRNP5Ih76kzzF/view?usp=sharing and download the zip file
3. Unzip the zip file (make sure the folder is called "models") and then place it in the project's root directory

Then type the following in your terminal:

4. `python -m venv <VirtualEnvironmentName>`
5. `source <VirtualEnvironmentName>/bin/activate`
6. `make install`
7. `python -m spacy download en_core_web_sm`

and lastly, to run type:

8. `python3 main.py`

# About

In this project, we trained a deep-learning model (deberta-v3) and non-deep learning model (logistic regression with features from TF-IDF) to predict the number of stars a written course review. We achieved 80.79% accuracy with our deep learning model and 77.90% accuracy with our logistic regression model.
