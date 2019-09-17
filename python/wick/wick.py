from fullctr import fullctr

orb='iapqrsbj'
pm=[-1,1,0,0,0,0,1,-1]
#dag=[1,-1,1,-1,1,-1,1,-1]

ctrlist = fullctr(orb,pm)

for i in ctrlist:
    print(i)
