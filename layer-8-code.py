# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %% [markdown]
# # main

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
LE=LabelEncoder()

# %%
# Load the dataset from CSV file
train = pd.read_csv('/kaggle/input/layer-8/train.csv')
valid = pd.read_csv('/kaggle/input/layer-8/valid.csv')
test = pd.read_csv('/kaggle/input/layer-8/test.csv')

# %%
train_original=train.copy()
valid_original=valid.copy()
test_original=test.copy()

# %%
train.head()

# %%
train.shape

# %%
TRAIN=[]

for i in range(4):
    TRAIN.append(train.dropna(subset=[train.columns[768+i]]))

TRAIN[0].shape, TRAIN[1].shape, TRAIN[2].shape, TRAIN[3].shape

# %%


# train_1=train.dropna(subset=[train.columns[768]])
# train_2=train.dropna(subset=[train.columns[769]])
# train_3=train.dropna(subset=[train.columns[770]])
# train_4=train.dropna(subset=[train.columns[771]])

# %%
# train_1.shape, train_2.shape, train_3.shape,train_4.shape,

# %%
VALID=[]

for i in range(4):
    VALID.append(valid.dropna(subset=[valid.columns[768+i]]))

VALID[0].shape, VALID[1].shape, VALID[2].shape, VALID[3].shape

# %%
# valid_1=valid.dropna(subset=[valid.columns[768]])
# valid_2=valid.dropna(subset=[valid.columns[769]])
# valid_3=valid.dropna(subset=[valid.columns[770]])
# valid_4=valid.dropna(subset=[valid.columns[771]])

# %%
train.head()

# %%
lable="y"
X_TRAIN=[]
Y_TRAIN=[]

for i in range(4):
    X_TRAIN.append(TRAIN[i].iloc[:, :768])
    Y_TRAIN.append(TRAIN[i].iloc[:, 768+i].to_frame(lable))

X_TRAIN[0].shape, X_TRAIN[1].shape, X_TRAIN[2].shape, X_TRAIN[3].shape


# %%
lable="y"
X_VALID=[]
Y_VALID=[]

for i in range(4):
    X_VALID.append(VALID[i].iloc[:, :768])
    Y_VALID.append(VALID[i].iloc[:, 768+i].to_frame(lable))

# %%
Y_TRAIN[0].head()

# %%
X_test = test.iloc[:, 1:769]

# %%
X_test.head()

# %%
for i in range(4):
  unique_classes, class_counts = np.unique(Y_TRAIN[i], return_counts=True)
  plt.bar(unique_classes, class_counts)
  plt.xlabel(f"Label {i+1}")
  plt.ylabel('Number of samples')
  plt.title('Label Distribution')
  plt.show()

# %%
# sampler = RandomOverSampler(random_state=45)
# k = sampler.fit_resample(X_TRAIN[3], Y_TRAIN[3][lable])
# X_TRAIN[3],Y_TRAIN[3]= k[0],k[1].to_frame(name=lable)
# X_TRAIN[3].shape

# %%
# from imblearn.under_sampling import RandomUnderSampler

# # Assuming you have the following variables:
# # X_TRAIN[3] is your feature data
# # Y_TRAIN[3] is your target variable
# # label is the label you want to balance

# # Create a RandomUnderSampler instance
# sampler = RandomUnderSampler(random_state=45)

# # Fit and transform your data using the sampler
# X_resampled, Y_resampled = sampler.fit_resample(X_TRAIN[3], Y_TRAIN[3][lable])
# X_TRAIN[3],Y_TRAIN[3]=X_resampled, Y_resampled.to_frame(name=lable)

# X_TRAIN[3].shape

# # X_resampled and Y_resampled now contain the randomly undersampled data


# %%

# unique_classes, class_counts = np.unique(X_TRAIN[3], return_counts=True)
# plt.bar(unique_classes, class_counts)
# plt.xlabel(f"Label {i+1}")
# plt.ylabel('Number of samples')
# plt.title('Label Distribution')
# plt.show()

# %%
#XGBoost Classifier
def xgBoostModel(X_train,Y_train):
  num_classes = len(Y_train[lable].unique())
  if num_classes == 2:
    objective = 'binary:logistic'
  else:
    objective = 'multi:softmax'
  # Create an XGBoost model
  model = xgb.XGBClassifier(objective=objective, random_state=39, tree_method='gpu_hist')
  Y_train_encoded = LE.fit_transform(Y_train[lable])
  # Train the model
  model.fit(X_train, Y_train_encoded)
  return model


# %%
#Support Vector Classifier
def svmModel(X_train,Y_train):
  # Create an Support Vector Classifier
  model = SVC(kernel='rbf', decision_function_shape='ovr', random_state=40, C=100)
  Y_train_encoded = LE.fit_transform(Y_train[lable])
  # Train the model
  model.fit(X_train, Y_train_encoded)
  return model

# %%
def evaluator(X_train,Y_train,X_valid,Y_valid,X_test,model,y_lable="1" ):
  model_name = model.__class__.__name__
  print(model_name)

  Y_pred_encoded = model.predict(X_valid)
  Y_pred = LE.inverse_transform(Y_pred_encoded)

  # Evaluate the y1 using valid data
  accuracy = accuracy_score(Y_valid, Y_pred)
  print(f"Valid Data Accuracy for y{y_lable}: {accuracy:.2f}")

  # Test data the model using test data
  Y_pred_test_encoded = model.predict(X_test)
  Y_pred_test = LE.inverse_transform(Y_pred_test_encoded)

  return Y_pred_test



# %%
# model_svm=svmModel(X_TRAIN[1],Y_TRAIN[1])
# y_pred_after=evaluator(X_TRAIN[1],Y_TRAIN[1],X_VALID[1],Y_VALID[1],X_test,model_svm,"1")

Y_TEST_PRED=[]

for i in range(4):
  print(f"y{i+1} :")
  pca_model_svm=svmModel(X_TRAIN[i],Y_TRAIN[i])
  y_pred_after=evaluator(X_TRAIN[i],Y_TRAIN[i],X_VALID[i],Y_VALID[i],X_test,pca_model_svm, f"{i + 1}")
  Y_TEST_PRED.append(y_pred_after)



# %% [markdown]
# # Hyper parameter

# %%
# # hyperparameters tuning
# from sklearn.model_selection import GridSearchCV
# # hyperparameters tuning using random search
# from sklearn.model_selection import RandomizedSearchCV

# param_grid = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
#     'gamma': ['scale', 'auto', 0.1, 1],
#     # Add more hyperparameters and their values as needed
# }

# grid_search = RandomizedSearchCV(SVC(random_state=42), param_grid, cv=5, n_jobs=-1)
# grid_search.fit(X_TRAIN[0],Y_TRAIN[0])

# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

# y_pred_1=evaluator(X_TRAIN[0],Y_TRAIN[0],X_VALID[0],Y_VALID[0],X_test,best_model)


# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluator1(X_train, Y_train, X_valid, Y_valid, X_test, model):
    # Fit the model on the training data
    model.fit(X_train, Y_train)

    # Make predictions on the training, validation, and test sets
    # y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)

    # Calculate evaluation metrics for each dataset
    # train_accuracy = accuracy_score(Y_train, y_train_pred)
    valid_accuracy = accuracy_score(Y_valid, y_valid_pred)
    # test_accuracy = accuracy_score(Y_test, y_test_pred)

    print(f"Valid Data Accuracy for y1: {valid_accuracy:.2f}")



    return y_test_pred

# Usage example:
# results = evaluator(X_TRAIN[0], Y_TRAIN[0], X_VALID[0], Y_VALID[0], X_test, best_model)


# %%
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.svm import SVC
# import numpy as np

# # Reduce the search space for hyperparameters
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'kernel': ['linear'],
#     'gamma': ['scale', 'auto', 0.1, 1],
# }

# # Reduce the number of iterations
# n_iter = 20

# # Use a smaller subset of data for initial search
# subset_indices = np.random.choice(len(X_TRAIN[0]), size=1000, replace=False)
# X_subset = X_TRAIN[0][:]
# Y_subset = Y_TRAIN[0][:]

# grid_search = RandomizedSearchCV(SVC(random_state=42), param_grid, cv=5, n_jobs=-1, n_iter=n_iter)
# grid_search.fit(X_subset, Y_subset)

# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

# # Perform final evaluation on the full dataset
# # y_pred_1 = evaluator1(X_TRAIN[0], Y_TRAIN[0], X_VALID[0], Y_VALID[0], X_test, best_model)


# %%
# y_pred_1 = evaluator1(X_TRAIN[0], Y_TRAIN[0], X_VALID[0], Y_VALID[0], X_test, best_model)

# %% [markdown]
# # PCA
# 

# %%
from sklearn.decomposition import PCA

def pca(X_train,X_valid,X_test,desired_variance_ratio=0.95):
  desired_variance_ratio = desired_variance_ratio
  pca = PCA(n_components=desired_variance_ratio, svd_solver='full')

  X_train_pca = pca.fit_transform(X_train)
  X_valid_pca = pca.transform(X_valid)
  X_test_pca = pca.transform(X_test)

  return X_train_pca ,X_valid_pca ,X_test_pca


# %%

X_TRAIN_PCA=[]
X_VALID_PCA=[]
X_TEST_PCA=[]

for i in range(4):
  X_TRAIN_PCA.append(pca(X_TRAIN[i],X_VALID[i],X_test)[0])
  X_VALID_PCA.append(pca(X_TRAIN[i],X_VALID[i],X_test)[1])
  X_TEST_PCA.append(pca(X_TRAIN[i],X_VALID[i],X_test)[2])

for i in range(4):
  print(X_TRAIN_PCA[i].shape, X_VALID_PCA[i].shape, X_TEST_PCA[i].shape)


# %%
# pca_model_svm_1=svmModel(X_TRAIN_PCA[0],Y_TRAIN[0])
# y_pred_after_1=evaluator(X_TRAIN_PCA[0],Y_TRAIN[0],X_VALID_PCA[0],Y_VALID[0],X_TEST_PCA[0],pca_model_svm_1,"1")
Y_TEST_PRED=[]

for i in range(4):
  print(f"y{i+1} :")
  pca_model_svm=svmModel(X_TRAIN_PCA[i],Y_TRAIN[i])
  y_pred_after=evaluator(X_TRAIN_PCA[i],Y_TRAIN[i],X_VALID_PCA[i],Y_VALID[i],X_TEST_PCA[i],pca_model_svm, f"{i + 1}")
  Y_TEST_PRED.append(y_pred_after)


# %%
#create the svm model for y1
# model_svm_1=svmModel(X_TRAIN[0],Y_TRAIN[0])
# y_pred_1=evaluator(X_TRAIN[0],Y_TRAIN[0],X_VALID[0],Y_VALID[0],X_test,model_svm_1)

#create the XGBoost model for y1
# model_XGBoost_1=xgBoostModel(X_train_1,Y_train_1)
# y_pred_low=evaluator(X_train_1,Y_train_1,X_valid_1,Y_valid_1,X_test,model_XGBoost_1)



# %% [markdown]
# # explainability

# %%
model_svm_1=svmModel(X_TRAIN_PCA[0],Y_TRAIN[0])
y_pred_after_1=evaluator(X_TRAIN_PCA[0],Y_TRAIN[0],X_VALID_PCA[0],Y_VALID[0],X_TEST_PCA[0],model_svm_1,f" {i + 1}")

coefficients = model_svm_1.coef_
absolute_coefficients = np.abs(coefficients)
absolute_coefficients

# %%
absolute_coefficients.shape

# %%
top_weights = []
num_classes = len(Y_TRAIN[0][lable].unique())

for class_X in range(num_classes):
    for class_Y in range(class_X + 1, num_classes):
        index = int(class_X * (2 * num_classes - class_X - 1) / 2 + class_Y - class_X - 1)
        for feature_index, weight in enumerate(coefficients[index]):
            absolute_weight = np.abs(weight)
            if len(top_weights) < 20:
                top_weights.append((absolute_weight, feature_index, class_X, class_Y, weight))
            else:
                min_absolute_weight = min(top_weights, key=lambda x: x[0])
                if absolute_weight > min_absolute_weight[0]:
                    min_index = top_weights.index(min_absolute_weight)
                    top_weights[min_index] = (absolute_weight, feature_index, class_X, class_Y, weight)

top_weights.sort(reverse=True)
# Loop through the top_weights list and print the top 20 weights with their details
for i, (absolute_weight, feature_index, class_X, class_Y, weight) in enumerate(top_weights):
    print(f"Top {i + 1} Weight: {weight:.2f} | Feature Index: {feature_index} | Class X: {class_X} | Class Y: {class_Y}")


# %% [markdown]
# # Lable 2 (Nural Network)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# %%
# print(f"data dataset shape {data.shape}")
# print(f"# of missing values {data['label_2'].isna().sum()}")
print(f"# of labels {Y_TRAIN[1].value_counts().shape[0]}")
print(f"label summary\n{Y_TRAIN[1].value_counts()}")

# data.dropna(subset=['label_2'], inplace=True)
# print(f"data dataset shape {data.shape}")

# %%


# %%
class AgeClassifier(nn.Module):
    def __init__(self, dropout_prob=0.5, weight_decay=1e-5):
        super(AgeClassifier, self).__init__()
        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 17)

        # Define dropout layers
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)

        # Define L2 regularization (weight decay) for linear layers
        self.l2_regularization = nn.Linear(1, 1)  # Initialize a linear layer with weight 1

        # Set weight_decay for regularization
        self.weight_decay = weight_decay

    def forward(self, tensors):
        output_l1 = torch.relu(self.linear1(tensors))
        output_l2 = torch.relu(self.linear2(output_l1))
        output_l3 = torch.relu(self.linear3(output_l2))
        output_l4 = self.linear4(output_l3)
        return output_l4

    def l2_regularization_loss(self):
        # Calculate L2 regularization loss for linear layers
        l2_loss = 0.0
        for param in self.parameters():
            if param.requires_grad:
                l2_loss += torch.norm(param, 2)
        return self.weight_decay * l2_loss


# %%
from sklearn.preprocessing import OneHotEncoder

label_2 = Y_TRAIN[1].values.reshape(-1, 1)
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(label_2)
print(ohe.categories_)

label_2 = ohe.transform(label_2)
print(label_2)

# %%
from sklearn.model_selection import train_test_split

X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_TRAIN[1], label_2, test_size=0.2, stratify=label_2, random_state=2023)

# X_train_tensors = torch.tensor(X_TRAIN[1].values, dtype=torch.float32)
# print(X_train_tensors.shape)

# X_test_tensors =  torch.tensor(X_VALID[1].values, dtype=torch.float32)
# print(X_test_tensors.shape)

X_train_tensors = torch.tensor(X_train_cv.iloc[:, :].values, dtype=torch.float32)
print(X_train_tensors.shape)

X_test_tensors = torch.tensor(X_test_cv.iloc[:, :].values, dtype=torch.float32)
print(X_test_tensors.shape)

# %%


# %%
# Create an instance of the FakeBERT model
ageClassifier = AgeClassifier()

# Define a cross-entropy loss function
criterion = nn.CrossEntropyLoss()

# Create a DataLoader for batching
batch_size = 128
dataset = TensorDataset(X_train_tensors, torch.tensor(y_train_cv))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define an optimizer
optimizer = torch.optim.Adam(ageClassifier.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):

    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = ageClassifier(inputs)

        ce_loss = criterion(outputs, targets)
        l2_loss = ageClassifier.l2_regularization_loss()

        total_loss = ce_loss + l2_loss

        total_loss.backward()
        optimizer.step()

    y_test_pred = ageClassifier(X_test_tensors)
    ce = criterion(y_test_pred, torch.tensor(y_test_cv))
    acc = (torch.argmax(y_test_pred, 1) == torch.argmax(torch.tensor(y_test_cv), 1)).float().mean()
    print(f"Epoch {epoch} validation: Cross-entropy={float(ce)}, Accuracy={float(acc)}")

# Save the trained model if needed
# torch.save(ageClassifier.state_dict(), 'ageClassifier_model.pth')

# %%
from sklearn.preprocessing import OneHotEncoder

label_2=torch.tensor(Y_VALID[1].values, dtype=torch.float32)
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(label_2)
print(ohe.categories_)

label_2 = ohe.transform(label_2)
print(label_2)
print(label_2.shape)


# Set your model to evaluation mode
ageClassifier.eval()

X_valid_tensors = torch.tensor(X_VALID[1].values, dtype=torch.float32)
print(X_valid_tensors.shape)

# label_2=torch.tensor(Y_VALID[1].values, dtype=torch.float32)
# print(label_2.shape)

y_pred = ageClassifier(X_valid_tensors)
print(y_pred.shape)

ce = criterion(y_pred, torch.tensor(label_2))
acc = (torch.argmax(y_pred, 1) == torch.argmax(torch.tensor(label_2), 1)).float().mean()
print(f"Cross-entropy={float(ce)}, Test Accuracy={float(acc)}")

# %% [markdown]
# # Generate CSV

# %%
# from sklearn.inspection import permutation_importance

# # Calculate permutation importances
# perm_importance = permutation_importance(model_svm_1,X_valid_1,Y_valid_1, n_repeats=30, random_state=42)
# feature_importances = perm_importance.importances_mean

# %%
Y_TEST_PRED[0].shape

# %%


# %%
def createCSVOutput(X_test_predict_array):
  IDs=[i for i in range(1, X_test_predict_array[0].shape[0]+1)]

  data = {
    'ID': IDs,
    'label_1': X_test_predict_array[0],
    'label_2': X_test_predict_array[1],
    'label_3': X_test_predict_array[2],
    'label_4': X_test_predict_array[3],
    # 'No of new features': [new_features.shape[1]] * len(y_pred),
  }
  # for i in range(new_features.shape[1]):
  #   data[f'new_feature_{i+1}'] = new_features[:, i]
  # for i in range(new_features.shape[1], 256):
  #       data[f'new_feature_{i+1}'] = [0] * len(y_pred)
  df = pd.DataFrame(data)
  filename = f'/kaggle/working/190420V_layer_8_results.csv'
  df.to_csv(filename, index=False)

# %%
createCSVOutput(Y_TEST_PRED)

# %% [markdown]
# 


