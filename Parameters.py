from sympy import *
import numpy as np

xn = Symbol('xn')
yn = Symbol('yn')

n = Symbol('n')
p = Symbol('p')
h = Symbol('h')
R = Symbol('R')
L = Symbol('L')
Th = Symbol('Th')
teta = Symbol('teta')

sphi = Symbol('sphi')
cphi = Symbol('cphi')

x0 = Symbol('x0')
y0 = Symbol('y0')
y0 = sqrt(R**2-x0**2)

x1 = Symbol('x1')
y1 = Symbol('y1')
y1 = sqrt((R-Th)**2-x1**2)

x2 = Symbol('x2')
y2 = Symbol('y2')
y2 = sqrt(R**2-x2**2)

X = Symbol('X')
Y = Symbol('Y')

cw = Symbol('cw')
c = Symbol('c')
c0 = Symbol('c0')
c1 = Symbol('c1')
c2 = Symbol('c2')

xn = -(R+h)*sin(teta) + n*p*cphi
yn =  (R+h)*cos(teta) + n*p*sphi

t1 = Symbol('t1')
dt1dx0 = Symbol('dt1dx0')

t1 = sqrt((x0-xn)**2+(y0-yn)**2)/cw + sqrt((X-x0)**2+(Y-y0)**2)/c
dt1dx0 = simplify(diff(t1,x0))

t2 = Symbol('t2')
dt2dx0 = Symbol('dt2dx0')
dt2dx1 = Symbol('dt2dx1')

t2 = sqrt((x0-xn)**2+(y0-yn)**2)/cw + sqrt((x1-x0)**2+(y1-y0)**2)/c0 + sqrt((X-x1)**2+(Y-y1)**2)/c1
dt2dx0 = simplify(diff(t2,x0))
dt2dx1 = simplify(diff(t2,x1))

t3 = Symbol('t3')
dt3dx0 = Symbol('dt3dx0')
dt3dx1 = Symbol('dt3dx1')
dt3dx2 = Symbol('dt3dx2')

t3 = sqrt((xn-x0)**2+(yn-y0)**2)/cw + sqrt((x1-x0)**2+(y1-y0)**2)/c0 + sqrt((x2-x1)**2+(y2-y1)**2)/c1 + sqrt((X-x2)**2+(Y-y2)**2)/c2
dt3dx0 = simplify(diff(t3,x0))
dt3dx1 = simplify(diff(t3,x1))
dt3dx2 = simplify(diff(t3,x2))
