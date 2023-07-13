import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs('out', exist_ok=True)

def gradFlowScipy(D, alpha, A, y):
  m,n = A.shape

  def f(t,w):
    a,b = w.reshape(2,n)
    x = a**D-b**D
    G = D*A.transpose()@(y-A@x)
    return np.concatenate((a**(D-1)*G, -b**(D-1)*G))

  def df(t,w):
    a,b = w.reshape(2,n)
    x = a**D-b**D
    G = D*A.transpose()@(y-A@x)
    H = np.zeros((n*2,n*2))
    H[:n,:n] = (D-1)*np.diag( a**(D-2)*G)
    H[n:,n:] = (D-1)*np.diag(-b**(D-2)*G)
    P = np.concatenate((A@np.diag(a**(D-1)), -A@np.diag(b**(D-1))),axis=1)
    H -= D**2*P.transpose()@P
    return H

  y0 = np.ones(n*2)*alpha
  res = scipy.integrate.LSODA(f, 0, y0, 1e9*alpha**(2-D), jac=df, rtol=1e-12, atol=0)
  while res.status == 'running': res.step()
  assert(res.status == 'finished')
  a,b = res.y.reshape(2,n)
  x = a**D-b**D
  assert(np.linalg.norm(A@x-y) < 1e-10)
  return x

Ds = [3,4,5]
alphaDs = [10**-i for i in range(10)]
epses = [0.3,0.1,0.03,0.01]

for D in Ds:
  plt.clf()
  plt.xlabel('$\\alpha^p$', fontsize=20)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  for i,eps in enumerate(epses):
    A = np.array([[1,1-eps]])
    y = np.array([1])
    limit = np.array([1,0])
    dists = []
    for alphaD in alphaDs:
      print(D, eps, alphaD)
      alpha = alphaD**(1/D)
      x = gradFlowScipy(D,alpha,A,y)
      dists.append( np.linalg.norm(x-limit) )

    style = "k"+"- -- : -.".split()[i]
    label = f'$\epsilon = {eps}$'
    plt.loglog(alphaDs, dists, style, label=label, lw=2)

  plt.legend(fontsize=15)
  plt.savefig(f'out/knekk_d{D}.pdf', bbox_inches='tight')
  #plt.show()
