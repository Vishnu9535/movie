x=list()
x=[0,1,2,3,4,5]
z=list()
if isinstance(x,list):
    z=[i for i in x]
print(z)
if len(z)>2:
    z=z[:3]
print(z)