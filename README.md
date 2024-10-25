java c
School of Natural and Computing Sciences
Department of Computing Science
MSc in Artificial Intelligence
2024 – 2025
Assessment Item 1 of 2 Briefing Document – Individually Assessed (no teamwork)
Title: CS5062 – Machine Learning
Note: This assessment accounts for 50% of your total mark of the course.
Learning Outcomes
On successful completion of this component a student will have demonstrated competence in the following areas:
• Have knowledge  understanding of the core concepts of, and common practices, in Machine Learning.
• Have knowledge and understanding of fundamentals of machine learning, including a range of popular machine learning algorithms.
• Be able to use existing machine learning tools, frameworks, and libraries to build solutions for real-world or benchmark problem solving.
• Be able to perform. data pre-processing for machine learning.
• Be able to systematically evaluate the built machine learning solutions.
• Be able to critically examine the strengths and limitations of common machine learning algorithms when solving a specific problem.
• Be able to write reports for machine learning solutions.
Report Guidance  Requirements
Your report must conform. to the below structure and include the required content as outlined in each section. Each subtask has its own marks allocated. You must supply a written report, along with the corresponding code, containing all distinct sections/subtasks that provide a full critical and reflective account of the processes undertaken.
This assessment includes two tasks. The first task focuses on model selection. The main purpose of this task is to understand that when analysing a data set, there are often a number of machine learning models available. Evaluating these models and choosing the best possible models are the core efforts when using machine learning. The second task will focus on an image classification which provides you an opportunity to employ the state-of-the-art machine learning tools to analyse a relatively big data set, providing you a taste of using machine learning tools in real-world problems.
The following provides a detailed description over the two tasks. To complete these tasks, you are allowed to use any machine learning frameworks including TensorFlow and PyTorch.
Both datasets needed to fulfil the requirements of this assessment can be found in MyAberdeen.
Task 1: Model Selection (25 marks) [~ 1000 words]
Overview of the data: The purpose of this data is to use machine learning models to predict which passengers survived the Titanic shipwreck. The sinking of the Titanic is one of the most infamous shipwrecks in history. By looking at the data, it seems that some groups of people were more likely to survive than others although there was some element of luck involved in surviving. The purposes are to find a good classifier using this data as well as to find out what short of people were more likely to survive.
There are 12 variables in the table. Each data sample is given the ground truth of survival and so it can be used for evaluating your predictions. The data features are explained as follows (see the file titanic-data.csv):
• Survived: survival, 0=NO, and 1=Yes
• Pclass: Ticket class, 1=1st, 2=2nd, and 3=3rd
• Sex: sex
• Age: age in years
• SibSp: numbers of siblings or spouses aboard the Titanic
• Parch: numbers of parents or children aboard the Titanic
• Ticket: Ticket number
• Fare: Passenger fare
• Cabin: cabin number
• Embarked: Port of Embarkation, C = Cherbourg, Q = Queenstown, S = Sourthampton
You must accomplish the following subtasks using machine learning.
Subtasks:
1. Data import, summary, and preprocessing: As a first step, you need to load the data from the CSV file into Python environment that you have chosen. You should provide a summary of the data and preprocess the data. For example, what features would not be included in the analysis and why? Are there missing values? Are there any categorical variables? How would you deal with them and why? Shall we normalize or standardize the data before training any models and why? (5 marks)
2. Discussion on selecting an algorithm: Suppose you were a Machine Learning intern to design a machine learning model to predict the survival. You can choose many models for this purpose such as KNN, SVM, Neural Networks, Logistic Regression, etc. The intern used 70% of the data as training set, another 20% as validation set, and finally 10% as test set. The intern trained 10 different models and recorded the accuracy on the test set. The intern’s best performing model achieved 90% of accuracy. The intern concludes that this model is the best one to use. Would you agree? Why? Please explain and elaborate. (3 marks)
3. Designing algorithms: You will be using Logistic Regression (LR) for this task. One potential problem of us代 写CS5062 – Machine Learning 2024 – 2025 Assessment Item 1Python
代做程序编程语言ing Logistic Regression is overfitting. To reduce such issue, you will need to use LR with regulations. Therefore, this task is to write your own Python code to implement the regularized Logistic Regression algorithm. You must write your own code to implement the gradient descent algorithm and so the LR algorithm. However, you can choose to use Automatic Differentiation (AD) to compute the gradients in your code. For example, you can use TensorFlow, Pytorch, or Sympy to do the AD. Randomly choose 90% of the data as training set and the rest 10% as test set. Please report how you split the data into training and test sets. Please explain how you have code up the LR algorithm and report steps to train the model in detail and present accuracy results. (8 marks)
4. Choosing the best regularization parameter: Please explain the role of the regularization in the LR model. How is model controlled by the regularizer? You will need to choose the best regularizer for this data set. Explain the method you have chosen and elaborate the process using data, and graphs. (4 marks)
5. Decide the luck elements involved in surviving. As the best regularization parameter was chosen, what are the luck elements involved in surviving? Explain your results. (5 marks)
Task 2: Classification (25 marks) [~ 1000 words]
In this task, you are given a set of images which contain either a dog or a cat. The aim is to train machine learning classifiers to classify whether an image contains either a dog or a cat. This data was originally taken from the Kaggle competition (https://www.kaggle.com/c/dogs-vs-cats). Both training and test data sets will be made available on MyAberdeen (train_data_small.zip and test_data_small.zip). Note that the training data will be used to train the classifiers and the test data used for evaluations. The class labels are contained in the filenames with the words “cat” and “dog”.
To accomplish this task, you are expected to explore a range of machine learning classification algorithms introduced in the module and other materials from literature. You will then investigate which classifier would be recommended as the best for this classification task providing critical comparisons and justifications. As this is a binary classification problem, accuracy is chosen as the error metric to evaluate the performance of the classifier.
In this assignment you are free to choose any classifiers, however, the marking will be reasonably relying on the accuracy on the test data computed by your algorithms. Therefore, you need to report the best mean accuracy value over the test images, which will be ranked against other students.
When working on this assignment, you must analyze and report the points including but not limited to,
• Data preprocessing: What data preprocessing strategies have you applied to the data before applying classification models? Explain why or why not you have made data preprocessing. (3 marks)
• You may have to use convolutional neural network (CNN) for this task. Explain why CNN could be the appropriate model for this particular task. (6 marks)
• Explicitly demonstrate and justify the training process. For example, you may have to use and explain the early-stopping technique to monitor when to stop training when using deep learning models; you may want to monitor the convergence process by plotting the accuracy against the training iterations. (8 marks)
• Compare and report the performance results of those algorithms you have chosen. You may have to use tables and graphs to demonstrate the results. Report the mean accuracy values of the trained classifiers applied to the test data. (8 marks)
Useful Information
• Please describe and justify each step that is needed to reproduce your results by using code-snippets, screenshots and plots. When using screenshots or plots generated in Python please make sure they are clearly readable.
• If you use open source code, you must point out where it was obtained from (even if the sources are online tutorials or blogs) and detail any modifications you have made to it in your tasks. You should mention this in both your code and report. Failure to do so will result in zero marks being awarded on related (sub)tasks.
Marking Criteria
• Quality of the report, including structure, clarity, and brevity.
• Reproducibility. How easy is it for another MSc AI student to repeat your work based on your report and code?
• Quality of your experiments, including design and result presentation (use of figures and tables for better reporting).
• Configured to complete the task and the parameter tuning process (if needed).
• In-depth analysis of the results generated, including critical evaluation, insights into data, and significant conclusions.
• Quality of the source code, including the documentation of the code.





         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
