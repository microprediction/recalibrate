# Grabbed from https://github.com/siebenrock/activation-functions/blob/master/activation_functions.ipynb to avoid keras dependency or whatever
import math


ACTIVATIONS = {'id':lambda x: x,
               'pw':lambda x: 2*(1 if x > 3 else 0 if x < -3 else 1/6*x+1/2)-1.0,
               'hs':lambda x: 1 if x > 0 else -1,
               'sg':lambda x: 2*(1 / (1 + math.exp(-x)))-1.0,
               'bs':lambda x: (1 - math.exp(-x)) / (1 + math.exp(-x)),
               'ht':lambda x: 2 / (1 + math.exp(-2 * x)) -1,
               'at':lambda x: (2/math.pi)*math.atan(x),
               'st':lambda x: (2/math.pi)*math.atan(max(0.1 * x, x)),
               'et':lambda x: (2/math.pi)*math.atan(x) if x > 0 else 0.5 * (math.exp(x) - 1)}

INVERSE_ACTIVATIONS = {'id': lambda y: y,
                       'ht': lambda y: -0.5*math.log(2/(y+1)-1) if y<1 else 100000000,
                       'ln': lambda y: 5*(y-0.5) }

if __name__=='__main__':
    import matplotlib.pyplot as plt
    x = [0.01*i for i in range(-500,500)]
    xf_interleaved = list()
    for ac,f in ACTIVATIONS.items():
        xf_interleaved.append(x)
        xf_interleaved.append([f(xi) for xi in x])
    plt.plot(*xf_interleaved)
    plt.legend(list(ACTIVATIONS.keys()))
    plt.show()
    print(math.atan(41))

    def one(x):
        y = ACTIVATIONS['ht'](x)
        return  INVERSE_ACTIVATIONS['ht'](y)

    print(one(0.8))








