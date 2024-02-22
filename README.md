# Detection of Keratitis and Its Subtypes Using Deep Learning and Slitlamp Images
Code for the paper entitled "From the Diagnosis of Infectious Keratitis to Discriminating Fungal Subtypes: A Deep Learning-Based Study," submitted for review by the Nature journal Scientific Reports.

The code developed for the training and evaluation phases of the three main models introduced in this paper is provided in this repository.
By accessing the data and sending a request to the corresponding author, these codes will facilitate the development of the final model, capable of providing the same results.
Please note that more detailed results, including metrics such as precision and recall, can be achieved by using well-known packages for this matter, such as scikit-learn.

Important Note: The developer needs to manually split the data into 5 subsets to perform K-fold cross-validation with K=5. The code is adjusted to accommodate the possibility of ignoring this validation method, and therefore, 20% of the data is assumed to be used for the evaluation phase.
