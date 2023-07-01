import sys
sys.path.append('.')
from lnn import *
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import pickle
import torch

# hyperparameters
num_predicates = 100
m = 10  # tunable hyperparameter m for the number of clauses in the m-CNF

# declare variables
x, y = Variables("x", "y")

# add predicates
predicate_array = []
not_predicate_array = []

for p in range(num_predicates):
    predicate_array.append(Predicate(f"R_{p}", 2))
    # R_p is the rectangle and S_p is the negation of the rectangle predicate
    not_predicate_array.append(Not(predicate_array[-1](x,y)))

# make the template formula
all_literals = predicate_array + not_predicate_array
clause_array = []
atoms = []
for p in all_literals:
    atoms.append(p(x,y))
for i in range(m):
    clause_array.append(And(*atoms))

formula = Or(*clause_array)  # FIXME: change the net structure if needed

# create model
# m = Model()
# m.add_knowledge(formula)

# add training data
df = pd.read_csv('output.csv').drop_duplicates(keep='first') # assume it has the columns for all the predicates and the formula predicate
df_train = df.sample(frac=0.8, random_state=0)
df_test = df.drop(df_train.index)
X, Y = df_train.iloc[:, :-1], df_train.iloc[:, -1]
print(X.shape, Y.shape)
print("value counts:", Y.value_counts())
# for idx in range(X.shape[0]):
#     for p, pred in enumerate(predicate_array):
#         my_fact = Fact.TRUE if bool(X.iloc[idx, p]) else Fact.FALSE
#         # print('pred', pred, 'fact', my_fact)
#         m.add_data({
#             pred: {(f'x_{idx}', f'y_{idx}'): my_fact}
#         })

#     my_label = Fact.TRUE if bool(Y.iloc[idx]) else Fact.FALSE
#     m.add_labels({
#         formula: {(f'x_{idx}', f'y_{idx}'): my_label}
#     })


# train with contradiction and supervised loss
# print('training')
# m.train(losses=[Loss.SUPERVISED], epochs=30)
# m.print()

# infer on the validation and test sets
m = pickle.load(open('m_cnf_lnn1.pkl', 'rb'))
print('-' * 80)
print('inference')
with torch.no_grad():
    m.infer(direction=Direction.UPWARD)
# print(formula.state())
# pickle.dump(m, open('m_cnf_lnn1.pkl', 'wb'))

print(formula.state())
my_pred = [0]*X.shape[0]
for k, v in formula.state().items():
    print(k, v)

    # if v == Fact.TRUE:
    #     my_pred[int(k[0].split('_')[1])] = 1
    
# exit(0)
     
print('training set results')   
print(classification_report(Y, my_pred))
print('accuracy:', accuracy_score(Y, my_pred))


train_set_size = X.shape[0]
print('testing set results')
X_test, Y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
print('testing set size:', X_test.shape, Y_test.shape)
print("value counts:", Y_test.value_counts())
for idx in range(X_test.shape[0]):
    for p, pred in enumerate(predicate_array):
        my_fact = Fact.TRUE if bool(X_test.iloc[idx, p]) else Fact.FALSE
        # print('pred', pred, 'fact', my_fact)
        m.add_data({
            pred: {(f'x_{idx+train_set_size}', f'y_{idx+train_set_size}'): my_fact}
        })
        
# m.infer(direction=Direction.UPWARD)
# my_test_pred = [0]*X_test.shape[0]
# for k,v in formula.state().items():
#     if int(k[0].split('_')[1]) < train_set_size:
#         continue
#     if v == Fact.TRUE:
#         my_test_pred[int(k[0].split('_')[1]) - train_set_size] = 1
     
# print('testing set results')   
# print(classification_report(Y_test, my_test_pred))
# print('accuracy:', accuracy_score(Y_test, my_test_pred))
