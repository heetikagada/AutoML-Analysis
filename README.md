# Performance evaluation of AutoML frameworks vs Conventional strategies

Designing architectures for machine learning networks can be challenging. Our goal is to understand and evaluate the AutoML tools and compare them against the traditional machine learning model building methods. We have evaluated the efficacy of employing AutoML frameworks in a real world scenario and compared its results against the conventional strategy of a person of Data Science knowledge building the machine learning pipeline himself. We gain knowledge from the metrics and investigate the best method and tool to use for training any machine learning model. 

## Approaches
We have looked into 4 autoML systems namely, Auto-sklearn (Feurer et al., 2019), Google Cloud AutoML (Google), H20 (H2O.ai, 2017) and TPot and build the system to tackle Classification problems.  At each step of the machine learning pipeline we have focused on the metrics to look into while implementing an AutoML model and compare it against its corresponding counterpart in the conventional or traditional strategy.
Datasets used:  Wine dataset,  Bank marketing dataset, Spambase dataset

### Traditional Methods

We have built the Machine Learning models by tuning it by GridSearchCV for Random Forest, Gaussian Naive Bayes, Decision Tree and Logistic Regression. 
- Data Ingestion and Preparation: Manipulation of data is performed before it is ready for ingestion by the algorithms.
- Feature Engineering Automation: Estimation of the ability of the system to explore all available features and discover and evaluate features is done. 
- Machine Learning Algorithm: Decision to decide which ML parameters are to be selected by GridSearchCV is performed. 
- Evaluation: The algorithm is evaluated on test data and we recorded the performance.

The notebook that evaluates the traditional methods is: 

```
traditional_methods.ipynb
```


### Google AutoML Tables

Cloud AutoML is a suite of machine learning products that enables professionals with limited or no machine learning expertise to train high-quality models specific to their business needs. AutoML Tables is a supervised learning service. It uses structured data to train ML models for any data that the user gives. We select one column as the target dataset in AutoML. This column can be changed to create any number of models with the analyzed training data. We can also customize the training data by changing different hyperparameters like training/testing data split, node training hours and optimization parameters (AUC ROC, AUC PR etc. )

We implemented AutoML on the 3 datasets mentioned above. In Fig. 3,  we can see the results obtained from the Spambase Dataset. We can also check what parameters were used for the best performing model by viewing ‘Google Logging’ and record these parameters.

_Reference: https://cloud.google.com/automl-tables_ to implement Google AutoML Tables

### Auto sklearn

### TPOT 

### H2O

The notebook that evaluates the above AutoML methods is: 
```
automl_methods.ipynb
```

#### References: 

References: 
[1] ] L. Kotthoff, C. Thornton, H. H. Hoos, F. Hutter, and K. LeytonBrown, “Auto-weka 2.0: Automatic model selection and hyperparameter optimization in weka,” The Journal of Machine Learning Research, vol. 18, no. 1, pp. 826–830, Jan. 2017

[2] M. Feurer, A. Klein, K. Eggensperger, J. Springenberg, M. Blum, and F. Hutter, “Efficient and robust automated machine learning,” in Advances in Neural Information Processing Systems 28, 2015, pp. 2962–297

[3] “Mljar,” https://github.com/mljar/mljar-api-python, accessed: 2019-04-10.

[4] https://cloud.google.com/automl-tables




