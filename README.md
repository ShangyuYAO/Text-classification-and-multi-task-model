# Text-classification-and-multi-task-model
## Method: TF-IDF combined with neural network.

> This project aims to investigate the performance of single-head and multi-head classification models through a text classification task. The project employs the TF-IDF method combined with a neural network for text classification, where TF-IDF is used for feature vector extraction, and the neural network serves as the classifier. Specifically, we focus on two classification criteria: a primary criterion corresponding to the gender of the text author and a secondary criterion corresponding to the date of the first edition. Initially, a model with a single classification head is implemented to classify the text based on the gender of the author. Subsequently, a model with two classification heads is implemented to simultaneously classify the gender of both text authors and the date of the first edition. Finally, the performance of the two models will be compared.
>

## Table of Contents  
1. [Introduction](#introduction) 
2. [Methodology](#methodology) 
3. [Data](#data) 
4. [Result](#result) 
5. [Conclusion](#conclusion) 
6. [Code Usage Suggestions](#code-usage-suggestions) 
7. [Reference](#reference)

## Introduction
Nowadays, multi-task models are widely used in various fields such as natural language processing, computer vision, speech recognition. The demand for research and optimization of multi-task models is growing.

A multi-task model is a type of machine learning model designed to simultaneously address multiple related but potentially different tasks. Multi-head models can be considered a specific form of multi-task models. In a multi-head model, each head corresponds to a task, allowing the model to handle multiple tasks simultaneously. These models typically share some representations at their lower layers to enhance learning and generalization. Each head has its own set of parameters to adapt to the specific requirements of its corresponding task.

This project aims to explore the performance of multi-head models in text classification by constructing a relatively simple dual-head neural network model. During this process, a method called TF-IDF is employed to extract feature vectors, which are then connected to the single-head or dual-head neural networks we build to accomplish the text classification task.

## Rrinciple
In this section, the principles of TF-IDF and multi-task models will be briefly introduced.

### TF-IDF
TF-IDF, which stands for Term Frequency-Inverse Document Frequency, is a numerical statistic widely used in natural language processing and information retrieval. It quantifies the importance of a term (word) relative to a collection of documents. The idea behind TF-IDF is to assign a weight to each term based on how often it appears in a specific document (Term Frequency) and how unique it is across the entire document collection (Inverse Document Frequency). The definition of TF-IDF is the product of two statistics, term frequency and inverse document frequency.

**Term frequency(TF)**

Term Frequency (TF) measures how often a term appears in a document. The formula for Term Frequency is:

<font size=30>$$\mathrm{tf}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d}f_{t',d}}$$</font>

Where:
 - $f_{t,d}$ is the frequency of term $\mathbf{t}$ in document $\mathbf{d}$.
 - $\sum_{t' \in d}f_{t',d}$ is the sum of frequencies of all terms in document $\mathbf{d}$.

In a more intuitive form, the formula can be expressed as follows:

<font size=30>$$TF(t,d)=\frac{\text{Number of times term t appears in document d}}{\text{Total number of terms indocument d}}$$</font>

It's a normalized count representing the frequency of a term within a document. TF gives higher weights to terms that occur more frequently in a document.

**Inverse document frequency(IDF)**

Inverse Document Frequency (IDF) measures the importance of a term in the entire collection of documents. The formula for Inverse Document Frequency is:

<font size=30>$$\mathrm{idf}(t, D) =  \log \frac{N}{|\{d \in D: t \in d\}|+1}$$</font>

where
 - $\mathbf{N}$ is total number of documents in the corpus $N = {|D|}$.
 - $|\{d \in D: t \in d\}|$ is the number of documents where the term $\mathbf{t}$ appears. 

In a more intuitive form, the formula can be expressed as follows:

<font size=30>$$IDF(t,D)=log\left(\frac{\text{Number number of documents in the collection D}}{\text{Number of documents containing term t}+1}\right)$$</font>

 It penalizes terms that are common across many documents and gives higher weights to terms that are rare and appear in fewer documents.
 
**Term frequency–inverse document frequency (TF-IDF)**

Then tf–idf is calculated as:

<font size=30>$$\displaystyle \mathrm {tfidf} (t,d,D)=\mathrm {tf} (t,d)\cdot \mathrm {idf} (t,D)$$</font>

This combined score helps to identify the importance of a term in a specific document relative to its importance across the entire document collection.

### Multi-task model
Multi-task learning has two fundamental frameworks: Hard Parameter Sharing and Soft Parameter Sharing. Their graphical representations are as follows:

<p float="left">
  <img src="https://github.com/ShangyuYAO/Text-classification-and-multi-task-model/blob/main/Readme_hard_soft.png" width="50%" />
</p>

**Hard Parameter Sharing**
In the Hard Parameter Sharing framework, multiple tasks collectively leverage a shared neural network architecture. This shared network encompasses all layers and parameters, fostering the learning of a unified set of features that are applicable across all tasks. During training, the shared parameters are updated based on the combined error signals from all tasks. 

**Soft Parameter Sharing**
Conversely, the Soft Parameter Sharing framework introduces task-specific parameters alongside the shared ones. Each task possesses its unique set of parameters tailored to its specific requirements. These task-specific parameters are allowed to deviate from the shared ones, offering a degree of flexibility. To ensure a balance between task-specific adaptations and shared knowledge, a penalty is imposed on task-specific parameters if they deviate significantly from the shared parameters. 


## Methodology

This project utilizes TF-IDF combined with a neural network for text classification.

TF-IDF transforms textual data into a feature vector, where the vector's length corresponds to the number of features. In this project, the maximum number of features is set to 10,000, indicating that only the top 10,000 most significant features (words) are considered. After applying TF-IDF to the entire text corpus, a feature matrix is obtained with rows representing the number of files and columns set to 10,000.

This feature matrix is then fed into a neural network. For the neural network architecture, I opted for a relatively simple framework—a multilayer perceptron with a single hidden layer. The input layer consists of 10,000 neurons, corresponding to the columns of the TF-IDF feature matrix. The hidden layer contains 100 neurons. For a single-head model, the output layer has two neurons, suitable for a binary classification task. For a dual-head model, the output layer comprises four neurons, addressing two binary classification tasks. Each classification head utilizes cross-entropy loss as the loss function, and the loss of the dual-head model is the weighted sum of the losses from each head.

## Data
The dataset comprising 1052 texts (articles, books). The naming convention for these texts is consistent taking the form, for instance, of (BAZIN)(René)(l'isolée)(1)(1905)(1853)(1932)(fr)(z)(z)(V)(R)(T).txt. The first two sets of parentheses store the author's last name and first name, respectively. The third set of parentheses contains the title of the text. The fourth set of parentheses holds information about the author's gender, where 1 represents male and 2 represents female. The fifth set of parentheses stores the date of the first edition. In this dataset, the distribution of author genders and the date of the first edition is illustrated in the accompanying figures. For the project, I partitioned the dataset into training and testing sets with a ratio of 7:3, resulting in 737 texts for training and 315 texts for testing.

<p float="left">
  <img src="https://github.com/ShangyuYAO/Text-classification-and-multi-task-model/blob/main/readme_gender.jpg" width="30%" />
  <img src="https://github.com/ShangyuYAO/Text-classification-and-multi-task-model/blob/main/readme_year.jpg" width="30%" /> 
</p>

## Result
The confusion matrices for the single-head and dual-head models, obtained from classifying texts based on the author's gender, are shown in the figure below. Both matrices represent the best classification performance achieved after ten training epochs. Through experimentation, a slight improvement in the classification performance of the first head was observed upon introducing the second classification head.

Due to the shared underlying representation between the first and second classification heads, introducing the second head provides additional information to the first head, possibly contributing to the improvement in the performance of the first head. In fact, upon closer examination of the dataset, it is observed that works authored by the same individual often have the date of the first edition relatively close. This observation may offer valuable information for determining the author's identity and gender when classifying works.
<p float="left">
  <img src="https://github.com/ShangyuYAO/Text-classification-and-multi-task-model/blob/main/readme_before.jpg" width="30%" />
  <img src="https://github.com/ShangyuYAO/Text-classification-and-multi-task-model/blob/main/Readme_after.jpg" width="30%" /> 
</p>

## Conclusion
In this project, I employed a simple multilayer perceptron to construct both a single-head model and a dual-head model, comparing their performances in a text classification task. Through testing, a slight improvement in the classification performance of the original head was observed upon introducing the second classification head. This suggests that multi-head or multi-task models relying on shared underlying representations may achieve better results than single-head or single-task models. However, these results may not be generalizable because the model structure used in this project is singular and simplistic. To enhance the universality of the findings, it is necessary to explore and evaluate various model architectures.

Regarding TF-IDF, through experimentation, this method has proven to be simple, user-friendly, and capable of achieving satisfactory results. However, it still has some limitations. For instance, the straightforward structure of TF-IDF may not effectively reflect the importance of words and the distribution of key features. This limitation hinders its ability to adjust weights accurately, resulting in suboptimal precision. Additionally, TF-IDF does not consider the positional information of words. To achieve better text classification results, utilizing deep learning models such as BERT becomes essential.


## Code Usage Suggestions

This project comprises three code files: Data_processing.ipynb, single_head_model.ipynb, and dual_head_model.ipynb. All the code is written in Python and presented in Jupyter Notebook format. You can use them on Google Colab. A brief description of the contents of the three code files is provided below.

**Data_processing.ipynb**
This code needs to be run first.
This code file consists of the following main elements：
 - Data Collection and Preprocessing
 - Text Tokenization and Filtering
 - TF-IDF Vectorization
 - Saving Results

**single_head_model.ipynb**
The purpose of this code is to perform text classification using a single-head neural network model. The objective is to classify text based on the gender of the author (male or female). The code can be divided into the following sections:
 - Setting Up Environment
 - Loading and Preprocessing Data
 - Defining the Neural Network Model
 - Training the Neural Network
 - Testing the Neural Network

**dual_head_model.ipynb**
The purpose of this code is to perform text classification using a dual-head neural network model. The objective is to classify text based on both the gender of author (male or female) and the date of the first edition (before or after 1900) simultaneously. The code can be divided into the following sections:
 - Setting Up Environment
 - Loading and Preprocessing Data
 - Defining the Neural Network Model
 - Training the Neural Network
 - Testing the Neural Network

In addition to the three code files, there are also some files generated during the code execution process:

**sex.txt and year.txt**
These two text files respectively store the labels needed for author gender classification and date of the first edition classification.

**tf-idf.npy**
The file 'tf-idf.npy' stores the TF-IDF matrix in NumPy array format. This matrix contains the TF-IDF values for each term (word) in the corpus of documents. Each row in the matrix corresponds to a document, and each column corresponds to a unique term in the entire corpus.

## Reference

[1] Rajaraman, A., Ullman, J.D. (2011). "Data Mining". Mining of Massive Datasets. pp. 1–17. doi:10.1017/CBO9781139058452.002. ISBN 978-1-139-05845-2.

[2] Ruder, S. (2017). An Overview of Multi-Task Learning in Deep Neural Networks. ArXiv, abs/1706.05098.
