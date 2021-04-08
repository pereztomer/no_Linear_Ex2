import numpy as np


def generic_bisect(f, df, l, u, eps, k):
    """
    :param f: a function (lambda)
    :param df: the derivative of the function - f' (lambda)
    :param l: lower bound (double)
    :param u: upper bound (double)
    :param eps: mistake
    :param k: maximum iteration number (int)
    """
    h = 0
    fv = []
    if df(l) * df(u) >= 0:
        raise RuntimeError("df(u)*df(l)=>0")
    if eps < 0:
        raise RuntimeError("Epsilon must be a semi positive number")
    if k <= 0:
        raise RuntimeError("K must be a positive number")

    while abs(l - u) >= eps and k > 0:
        k -= 1
        fv.append(tuple([f(l), f(u)]))
        h = (l + u) / 2
        if df(u) * df(h) > 0:
            u = h
        else:
            l = h

    x = (l + u) / 2
    return x, fv


def generic_newton(f, df, ddf, x0, eps, k):
    """

    :param f: a fucntion
    :param df: f'
    :param ddf: f''
    :param x0: a beginning point for the newton method (double)
    :param eps: a minimal error (double)
    :param k: number of iterations (int)
    """
    if eps < 0:
        raise RuntimeError("Epsilon must be a semi positive number")
    if k <= 0:
        raise RuntimeError("K must be a positive number")
    x = []
    fv = []
    x.append(x0)
    counter = 1
    flag = True
    newton_equation = lambda x: x - df(x) / ddf(x)
    while flag and counter <= k - 1:
        x.append(newton_equation(x[counter - 1]))
        fv.append(f(x[counter]))

        if abs(x[counter] - x[counter - 1]) < eps:
            flag = False
        counter += 1
    return x[-1], fv


def generic_hybrid(f, df, ddf, l, u, eps1, eps2, k):
    if u <= l:
        raise RuntimeError("u must me larger than l")
    if eps1 < 0 or eps2 < 0:
        raise RuntimeError("epsilon must be semi positive")
    if k <= 0:
        raise RuntimeError("K must be a positive number")
    fv = []
    x = []
    fv.append(f(u))
    x.append(u)
    counter = 0
    while abs(u - l) > eps1 and k > 0 and abs(df(x[-1])) > eps1:
        x_newt = x[-1] - df(x[-1]) / ddf(x[-1])
        counter += 1
        if l <= x_newt <= u and abs(df(x_newt)) < 0.99 * abs(df(x[-1])):
            x.append(x_newt)
        else:
            x.append((u + l) / 2)

        if df(x[-1]) * df(u) > 0:
            u = x[-1]
        else:
            l = x[-1]
        k -= 1
        fv.append(f(x[-1]))
    return x[-1], fv


def generic_gs(f, l, u, eps, k):
    if u <= l:
        raise RuntimeError("u must me larger than l")
    if eps < 0:
        raise RuntimeError("epsilon must be semi positive")
    fv = []
    c = (-1 + np.sqrt(5)) / 2
    x2 = c * l + (1 - c) * u
    fx2 = f(x2)
    x3 = (1 - c) * l + c * u
    fx3 = f(x3)
    for i in range(k - 2):
        if fx2 < fx3:
            u = x3
            x3 = x2
            fx3 = fx2
            x2 = c * l + (1 - c) * u
            fx2 = f(x2)
        else:
            l = x2
            x2 = x3
            fx2 = fx3
            x3 = (1 - c) * l + c * u
            fx3 = f(x3)
        fv.append(tuple([fx2, fx3]))
        if np.abs(u - l) < eps:
            break
    x = (x2 + x3) / 2
    return x, fv


if __name__ == '__main__':
    f_x = lambda x: x ** 2 + (x ** 2 - 3 * x + 10) / (2 + x)
    df = lambda x: 2 * x + (x ** 2 + 4 * x - 16) / ((x + 2) ** 2)
    ddf = lambda x: x + 40 / (x + 2) ** 3
    eps = 10 ** -6
    lb = -1
    ub = 5
    k = 50
    x0 = (lb + ub) / 2
    # x, fv = generic_bisect(f_x, df,lb, ub, eps , k)
    # x,fv = generic_newton(f_x,df,ddf,x0,eps,k)
    # x, fv = generic_hybrid(f_x, df, ddf, lb, ub, eps, eps, k)
    x, fv = generic_gs(f_x, lb, ub, eps, k)
    print("The result is " + str(x))
    print()
    for obj in fv:
        print(obj)
    # print('lb={:10.10f} ub={:10.10f}'.format(obj[0],obj[1]))
