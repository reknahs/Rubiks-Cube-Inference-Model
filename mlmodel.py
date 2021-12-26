from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

X = np.empty(shape=(1620000,3), dtype=int)
y = np.empty(shape=1620000, dtype=int)
info = []
colors = ["blue","orange","yellow","white","red","green"]
for i in range(6):
    f = open(colors[i]+"_rgb","r")
    info.append(f.readlines())
    f.close()
spot = 0
f.close()
for i in range(270000):
    for j in range(6):
        X[spot] = np.array(list(map(int,info[j][i].split(" "))))
        y[spot] = j
        spot += 1

forest = RandomForestClassifier(n_estimators = 100, random_state = 0)
forest.fit(X, y)


f = open("model","wb")
pickle.dump(forest,f)
f.close()
