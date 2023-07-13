import scipy.optimize, scipy.integrate
import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs('out', exist_ok=True)

alpha = 1
def q(x, p):
  if p == 2:
    z = x/alpha**2
    return 2-(z**2+4)**.5 + z*np.arcsinh(z/2)
  else:
    maxx = 1e9
    T = alpha**(2-p)
    limT = 1-maxx**(2/p-1)/T
    minT = maxx**(2/p-1)/T

    def h(t):
      return (1-t)**(p/(2-p)) - (1+t)**(p/(2-p))
    def hinv(x):
      return scipy.optimize.brentq(lambda t : h(t)-x, -limT, limT)

    assert(abs(x) < maxx)
    return scipy.integrate.quad(hinv, 0, x/alpha**p, epsabs=0, epsrel=1e-10)[0]

def h(t, p):
  if p == 2:
    return 2*np.sinh(t)
  else:
    return (1-t)**(p/(2-p)) - (1+t)**(p/(2-p))

def g(x, p):
  if p == 2:
    return np.abs(x)*np.log(np.abs(x)/np.exp(1))
  else:
    return np.abs(x) - (p/2)*np.abs(x)**(2/p);

linewidth = 2
fontsize = 14


for func in 'qgh':
  plt.clf()
  if func == 'q':
    bd = 5
    f = q
    plt.axis([-bd, bd, 0,3]);
  elif func == 'g':
    bd = 8
    f = g
    plt.axis([-bd, bd, -1.2,3]);
  elif func == 'h':
    bd = 1-1e-2
    f = h
    plt.axis([-bd, bd, -20,20]);
  else:
    assert(func in 'qgh')

  t = np.linspace(-bd,bd,501)

  ax = plt.gca()
  ax.tick_params(axis='both', which='major', labelsize=fontsize)
  plt.locator_params(nbins=5)

  for i,p in enumerate([2,2.5,3,4]):
    style = "k"+"- -. : --".split()[i]
    label = f'p = {p}'

    ax.plot(t, [f(x,p) for x in t], style, label=label, lw=linewidth)


  plt.legend(fontsize=fontsize)
  plt.savefig('out/%s_p.pdf'%func)
  #plt.show()
