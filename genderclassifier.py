from sklearn import tree
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)
a = []
b = input("What is the height of this person?\n")
c = input("What is the weight if this person?\n")
d = input("What is the shoe size if this person?\n")
a.append(b)
a.append(c)
a.append(d)
prediction = clf.predict([a])
print(prediction)