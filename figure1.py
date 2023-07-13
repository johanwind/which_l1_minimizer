import matplotlib.pyplot as plt
import numpy as np
np.float = np.float64 # sklearn uses deprecated np.float
import sklearn.linear_model
import scipy.integrate
np.random.seed(0)

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
  res = scipy.integrate.Radau(f, 0, y0, 1e9*alpha**(2-D), jac=df, rtol=2.5e-14, atol=0)
  while res.status == 'running': res.step()
  assert(res.status == 'finished')
  a,b = res.y.reshape(2,n)
  x = a**D-b**D
  assert(np.linalg.norm(A@x-y) < 1e-10)
  return x


Ds = [2,2.5,3,4]
alphaDs = [10**-i for i in range(10)]

Ays = [([[1,1,1],
         [3,0,1]], 
        [3,3]),
       ([[2,-1,0,1],
         [0,3,2,0]],
        [0,6]), 
       ([[-0.111,  0.120, -0.370, -0.240, -1.197],
         [ 0.209, -0.972, -0.755,  0.324, -0.109],
         [ 0.210, -0.391,  0.235,  0.665,  0.353]],
        [0.973, -0.039, -0.886])]

Ays = [(np.array(A,dtype=np.float64),np.array(y,dtype=np.float64)) for A,y in Ays]


def analyticLimit(D,A,y):
  case = [c for c in range(len(Ays)) if (A.shape==Ays[c][0].shape)][0]
  r = []
  if case == 2:
    clf = sklearn.linear_model.LassoLars(alpha=0, fit_intercept=False)
    clf.fit(A,y)
    r = clf.coef_
  elif case in [0,1]:
    if D == 2:
      t = (4-6*2**(1/3)+9*2**(2/3))/31
    else:
      assert(D > 2)
      t = 1/(1+(3**(2/D)/(2**(2/D)+1))**(-D/(D-2)))
    r = [1-t,(1-t)*2,t*3]
    if case == 1: r.append(0)

  assert(len(r) == A.shape[1])

  return np.array(r)


for (A,y),name in zip(Ays,'line plane random'.split()):
  plt.xlabel('$\\alpha^p$', fontsize=20)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  for i,D in enumerate(Ds):
    limit = analyticLimit(D,A,y)
    dists = []
    for alphaD in alphaDs:
      print(list(A.shape), D, alphaD)
      alpha = alphaD**(1/D)
      x = gradFlowScipy(D,alpha,A,y)
      dists.append( np.linalg.norm(x-limit) )

    style = "k"+"- -- : -.".split()[i]
    plt.loglog(alphaDs, dists, style, label=f'$p = {D}$', lw=2)
  #plt.loglog(alphaDs, alphaDs, '--', label='y=x')
  plt.legend(fontsize=15)
  plt.savefig('out/%s_convergence.pdf'%name, bbox_inches='tight')
  #plt.show()
  plt.clf()
