im


x1 = list()
for i in range(100):
    x1.append(i+51)

print(x1)

x2 = []
for i in range(11):
    x2.append(1 - (i/10))

print(x2)

x3 = [0 if i < 0.5 else 1 for i in x2]

print(x3)