import sympy as sp
import numpy as np

thi,thL1,thT1,thL2,thT2,cL1,cT1,cL2,cT2,rho1,rho2,RL,RT,TL,TT,A,w,x,y = sp.symbols('thi,thL1,thT1,thL2,thT2,cL1,cT1,cL2,cT2,rho1,rho2,RL,RT,TL,TT,A,w,x,y')


# SNi,CSi,SNL1,CSL1,SNT1,CST1,SNL2,CSL2,SNT2,CST2,cL1,cT1,cL2,cT2,rho1,rho2,RL,RT,TL,TT,A,w,x,y = sp.symbols('SNi,CSi,SNL1,CSL1,SNT1,CST1,SNL2,CSL2,SNT2,CST2,cL1,cT1,cL2,cT2,rho1,rho2,RL,RT,TL,TT,A,w,x,y')

kL1 = w/cL1
kL2 = w/cL2
kT1 = w/cT1
kT2 = w/cT2

p1 = A*sp.exp(sp.I*(kL1*sp.sin(thi)*x + kL1*sp.cos(thi)*y)) + RL*sp.exp(sp.I*(kL1*sp.sin(thL1)*x - kL1*sp.cos(thL1)*y))


P1 = RT*sp.exp(sp.I*(kT1*sp.sin(thT1)*x - kT1*sp.cos(thT1)*y))

p2 = TL*sp.exp(sp.I*(kL2*sp.sin(thL2)*x + kL2*sp.cos(thL2)*y))

P2 = TT*sp.exp(sp.I*(kT2*sp.sin(thT2)*x + kT2*sp.cos(thT2)*y))


# p1 = A*sp.exp(sp.I*(kL1*SNi*x + kL1*CSi*y)) + RL*sp.exp(sp.I*(kL1*SNL1*x - kL1*CSL1*y))
#
# P1 = RT*sp.exp(sp.I*(kT1*SNT1*x - kT1*CST1*y))
#
# p2 = TL*sp.exp(sp.I*(kL2*SNL2*x + kL2*CSL2*y))
#
# P2 = TT*sp.exp(sp.I*(kT2*SNT2*x + kT2*CST2*y))


u1 = [sp.diff(p1,x) + sp.diff(P1,y), sp.diff(p1,y) - sp.diff(P1,x)]

e1 = [sp.diff(u1[0],x), sp.diff(u1[1],y), sp.diff(u1[0],y) + sp.diff(u1[1],x)]

s1 = [rho1*(cL1**2-2*cT1**2)*e1[0]+(rho1*cL1**2)*e1[1],(rho1*cT1**2)*e1[2]]

# s1 = [(rho1*cL1**2)*e1[1],(rho1*cT1**2)*e1[2]]


u2 = [sp.diff(p2,x) + sp.diff(P2,y), sp.diff(p2,y) - sp.diff(P2,x)]

e2 = [sp.diff(u2[0],x), sp.diff(u2[1],y), sp.diff(u2[0],y) + sp.diff(u2[1],x)]

s2 = [rho2*(cL2**2-2*cT2**2)*e2[0]+(rho2*cL2**2)*e2[1],(rho2*cT2**2)*e2[2]]

# s2 = [(rho2*cL2**2)*e2[1],(rho2*cT2**2)*e2[2]]



RT = [RL,RT,TL,TT]

eq = [u2[0]-u1[0],u2[1]-u1[1],s2[0]-s1[0],s2[1]-s1[1]]

eq = [(eeq.subs(y,0)).subs(x,0) for eeq in eq]


a = [[sp.simplify(sp.diff(eq[n],RT[m])/(sp.I*w)) if n<2 else sp.simplify(sp.diff(eq[n],RT[m])/(w**2)) for m in range(4)] for n in range(4)]

b = [-sp.simplify(sp.diff(eq[n],A)/(sp.I*w)) if n<2 else -sp.simplify(sp.diff(eq[n],A)/(w**2)) for n in range(4)]


for n in range(4):
    for m in range(4):

        print('a['+str(n)+','+str(m)+'] = '+str(a[n][m]))

    # print(a[n])

print('\n')

for n in range(4):

    print('b['+str(n)+'] = '+str(b[n]))


# e1 = [sp.simplify((ee1.subs(y,0)).subs(x,0)) for ee1 in e1]
#
# e2 = [sp.simplify((ee2.subs(y,0)).subs(x,0)) for ee2 in e2]
#
# s1 = [rho1*(cL1**2)*e1[0]+rho1*(cL1**2 - 2*cT1**2)*e1[1],rho1*(cL1**2-2*cT1**2)*e1[0]+(rho1*cL1**2)*e1[1],(rho1*cT1**2)*e1[2]]
#
#
# s2 = [rho2*(cL2**2)*e2[0]+rho2*(cL2**2 - 2*cT2**2)*e2[1],rho2*(cL2**2-2*cT2**2)*e2[0]+(rho2*cL2**2)*e2[1],(rho2*cT2**2)*e2[2]]
#
#
# # s1 = [sp.simplify((ss1.subs(y,0)).subs(x,0)) for ss1 in s1]
# #
# # s2 = [sp.simplify((ss2.subs(y,0)).subs(x,0)) for ss2 in s2]
# #
# #
# si = [sp.simplify((ss1.subs(RL,0)).subs(RT,0)) for ss1 in s1]
#
# sr = [sp.simplify(ss1.subs(A,0)) for ss1 in s1]
#
#
# TLpr = sp.simplify(((s2[0]+s2[1])/(si[0]+si[1])).subs(A,1.))
#
# # TLpr = sp.simplify((TLpr.subs(cT1,0)).subs(cT2,0))
#
#
# TTpr = sp.simplify(((s2[2])/((1/2)*(si[0]+si[1]))).subs(A,1))
#
# # TTpr = sp.simplify(TTpr.subs(cL2,0))
#
# print('\n')
#
# print(TLpr)
#
# print('\n')
#
# print(TTpr)
