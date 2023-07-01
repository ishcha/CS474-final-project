import numpy as np
import sys
sys.path.append('.')
from lnn import *
from lnn.constants import _Fact
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import pickle
import sys
import random
import matplotlib.pyplot as plt
import argparse
random.seed(0)

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('file_name', help='file name for data csv file')
parser.add_argument('m', help='number of clauses in CNF of LNN', nargs='?', default=128, type=int)
args = parser.parse_args()
file_name = args.file_name  # 'single_predicate/R_9'
print('Training LNN for:', file_name)
num_predicates = 10
m = int(args.m)  # 128  # tunable hyperparameter m for the number of clauses in the m-CNF
print(m)
num_epochs = 300

# declare variables
x, y = Variables("x", "y")

# add predicates
predicate_array = []
not_predicate_array = []

for p in range(num_predicates):
    predicate_array.append(Predicate(f"R_{p}", 2))
    # R_p is the rectangle and S_p is the negation of the rectangle predicate
    not_predicate_array.append(Not(predicate_array[-1](x,y))) #, activation={"type": NeuralActivation.Lukasiewicz}))

# make the template formula

clause_array = []
atoms = []
for p in range(len(predicate_array)):
    # atoms.append(Or(predicate_array[p](x,y), not_predicate_array[p](x,y))) #, activation={"type": NeuralActivation.Godel}))
    atoms.append(predicate_array[p](x,y))

all_literals = atoms + not_predicate_array
# for p in range(len(not_predicate_array)): # predicate_array)):
#     atoms.append(not_predicate_array[p](x,y))
num_clause_predicates = len(all_literals)
num_clauses_done = 0
i = 0
while True:
    my_predicates = random.sample(all_literals, k=num_clause_predicates)
    clause_array.append(Or(*my_predicates)) #, activation={"type": NeuralActivation.Godel}))
    num_clauses_done += 1
    if num_clauses_done == 2**(i+1)-1:
        i += 1
        # num_clauses_done = 0
        num_clauses_predicates = num_clause_predicates - 1
    if num_clauses_done == m:
        break
    
# for i in range(m):
#     clause_array.append(And(*atoms)) #, activation={"type": NeuralActivation.Godel}))

# new_clause_array = []
# # m = m//2
# for i in range(m):
#     new_clause_array.append(And(*clause_array, activation={"type": NeuralActivation.Godel}))
# print('second or')    
# clause_array = []
# m = m//2
# for i in range(m):
#     clause_array.append(And(*new_clause_array, activation={"type": NeuralActivation.Godel}))
# print('second and')
formula = And(*clause_array)  #, activation={"type": NeuralActivation.Godel})  # FIXME: change the net structure if needed
# print('formula')
# shape_formula1 = Forall((x, y), 
#                         Not(
#                             And(predicate_array[0], predicate_array[2], activation={"type": NeuralActivation.Godel}), 
#                         activation={"type": NeuralActivation.Godel}))
# create model
model = Model()#w_max=random.randrange(2.0))
model.add_knowledge(formula)
# m.add_knowledge(shape_formula1, world=World.AXIOM)
# sys.stdout = open('model.txt','wt')
# # with open('model.txt', 'w') as f:
# m.print(params=True)
# sys.stdout = sys.__stdout__
# model.print()

# add training data
df = pd.read_csv(f'data/{file_name}.csv').drop_duplicates(keep='first') # assume it has the columns for all the predicates and the formula predicate
# df_train = df.sample(frac=1., random_state=0)
# df_test = df.drop(df_train.index)
# X, X_test, Y, Y_test = train_test_split(df.iloc[:, :-1], df.iloc[:,-1], test_size=0.2, random_state=42, stratify=df.iloc[:,-1])
X, Y = df.iloc[:, :-1], df.iloc[:,-1]
# print(X.shape, Y.shape)
for idx in range(X.shape[0]):
    for p, pred in enumerate(predicate_array):
        my_fact = Fact.TRUE if bool(X.iloc[idx, p]) else Fact.FALSE
        # print('pred', pred, 'fact', my_fact)
        model.add_data({
            pred: {(f'x_{idx}', f'y_{idx}'): my_fact}
        })

    my_label = Fact.TRUE if bool(Y.iloc[idx]) else Fact.FALSE
    model.add_labels({
        formula: {(f'x_{idx}', f'y_{idx}'): my_label}
    })


# train with contradiction and supervised loss
# print('training')
(running_loss, loss_history), inference_history = model.train(losses=[Loss.SUPERVISED, Loss.CONTRADICTION], epochs=num_epochs, learning_rate=0.05, direction=Direction.UPWARD)
# model.print()

# infer on the validation and test sets
print('-' * 80)
print('inference')
model.infer(direction=Direction.UPWARD)
# print(formula.state())
pickle.dump(m, open('m_cnf_lnn2.pkl', 'wb'))
sys.stdout = open('model1.txt','wt')
# with open('model.txt', 'w') as f:

model.print(params=True)
sys.stdout = sys.__stdout__

my_pred = [0]*X.shape[0]
true = 0
false = 0
others = 0
contra = 0
for k, v in formula.state().items():
    if v == Fact.TRUE or v == _Fact.APPROX_TRUE:
        my_pred[int(k[0].split('_')[1])] = 1
        true += 1
    elif v == Fact.FALSE or v == _Fact.APPROX_FALSE:
        my_pred[int(k[0].split('_')[1])] = 0
        false += 1
    else:
        my_pred[int(k[0].split('_')[1])] = -1
        others += 1
     
print('training set results')   
print(classification_report(Y, my_pred))
print('accuracy:', accuracy_score(Y, my_pred))
print('true', true, 'false', false, 'others', others)


# train_set_size = X.shape[0]
# print('testing set results')
# # X_test, Y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
# print('testing set size:', X_test.shape, Y_test.shape)
# for idx in range(X_test.shape[0]):
#     for p, pred in enumerate(predicate_array):
#         my_fact = Fact.TRUE if bool(X_test.iloc[idx, p]) else Fact.FALSE
#         # print('pred', pred, 'fact', my_fact)
#         model.add_data({
#             pred: {(f'x_{idx+train_set_size}', f'y_{idx+train_set_size}'): my_fact}
#         })
        
# model.infer(direction=Direction.UPWARD)
# my_test_pred = [0]*X_test.shape[0]
# for k,v in formula.state().items():
#     if int(k[0].split('_')[1]) < train_set_size:
#         continue
#     if v == Fact.TRUE:
#         my_test_pred[int(k[0].split('_')[1]) - train_set_size] = 1
     
# print('testing set results')   
# print(classification_report(Y_test, my_test_pred))
# print('accuracy:', accuracy_score(Y_test, my_test_pred))

# print training loss plot

# print(loss_history)
# print(running_loss)
# print(inference_history)
# print()
# loss_sup = [loss_history[i][0] for i in range(len(loss_history))]
# loss_contra = [loss_history[i][1] for i in range(len(loss_history))]
plt.plot(np.arange(len(running_loss)), running_loss, 'r')
# plt.plot(np.arange(num_epochs), loss_contra, 'g', '--', label='Contradiction')
plt.title('Variation of loss with training epochs')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.ylim((0, 0.7))
plt.xlim((0, 300))
# plt.legend()
plt.savefig(f'figures/{file_name}.png')
plt.show()
