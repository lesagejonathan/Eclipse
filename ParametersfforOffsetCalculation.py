from sympy import *
import numpy as np

n = Symbol('n')
m = Symbol('m')
p = Symbol('p')
h = Symbol('h')
d = Symbol('d')
Th = Symbol('Th')
sphi = Symbol('sphi')
cphi = Symbol('cphi')

xn = Symbol('xn')
yn = Symbol('yn')
xn = n*p*cphi
yn = h + n*p*sphi

xm = Symbol('xm')
ym = Symbol('ym')
xm = d - m*p*cphi
ym = h + m*p*sphi

x0 = Symbol('x0')
x1 = Symbol('x1')
x2 = Symbol('x2')

cw = Symbol('cw')
c0 = Symbol('c0')
c1 = Symbol('c1')

t = Symbol('t')
dtdx0 = Symbol('dtdx0')
dtdx1 = Symbol('dtdx1')
dtdx2 = Symbol('dtdx2')

t = sqrt((xn-x0)**2+yn**2)/cw + sqrt((x1-x0)**2+Th**2)/c0 + sqrt((x2-x1)**2+Th**2)/c1 + sqrt((xm-x2)**2+ym**2)/cw
dtdx0 = simplify(diff(t,x0))
dtdx1 = simplify(diff(t,x1))
dtdx2 = simplify(diff(t,x2))
