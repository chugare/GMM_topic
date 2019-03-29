

a = range(10)
b = range(11,20)
c = []
for i in range(10):
    c.append((i*2,i*3))
k = zip(a,b)
for i in k:
    print(i)
for i in zip(*c):
    print(list(i))