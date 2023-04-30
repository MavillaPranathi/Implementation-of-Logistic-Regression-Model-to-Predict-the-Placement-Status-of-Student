# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

 1. Import the standard libraries.
 2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
 3. Import LabelEncoder and encode the dataset.
 4. Import LogisticRegression from sklearn and apply the model on the dataset.
 5. Predict the values of array.
 6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
 7. Apply new unknown values


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:M.Pranathi 
RegisterNumber:212222240064 
*/
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

![image](https://user-images.githubusercontent.com/118343610/235356053-df866f1c-efc8-4542-8e9f-71537e6eb8b3.png)

![image](https://user-images.githubusercontent.com/118343610/235356079-930e72a0-9b31-4f89-a2d1-5e0139ed890a.png)

![image](https://user-images.githubusercontent.com/118343610/235356100-d2e571b7-f151-41e3-8b21-4858b175e70a.png)

![image](https://user-images.githubusercontent.com/118343610/235356135-338054f8-61fd-49cf-a468-ff52aa02aaf1.png)

![image](https://user-images.githubusercontent.com/118343610/235356159-09423055-817d-456c-b111-e8acb3ef4b2f.png)

![image](https://user-images.githubusercontent.com/118343610/235356190-7016400f-b337-433b-8d96-9d749172c2e4.png)

![image](https://user-images.githubusercontent.com/118343610/235356220-8a98b5dd-553f-472e-85f5-f7d4546d41ec.png)

![image](https://user-images.githubusercontent.com/118343610/235356294-8fdc8fba-6968-40dd-b3d5-5f8ea9819a31.png)

![image](https://user-images.githubusercontent.com/118343610/235356319-a300dae2-fdcd-468b-97cd-0703db3e1ebc.png)

![image](https://user-images.githubusercontent.com/118343610/235356392-865af987-dbaf-4ba1-8064-0574b3e302bb.png)

![image](https://user-images.githubusercontent.com/118343610/235356436-65fd447c-024e-4ec0-a518-15d596346876.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
