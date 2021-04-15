import numpy as np
import matplotlib.pyplot as plot

real_root = 3.58254399930370


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
    fv_fv1 = [((x[0] + x[1]) / 2) - real_root for x in fv]
    plot.semilogy(range(len(fv_fv1)), fv_fv1)
    plot.legend(['bisect'])
    plot.show()
    return x, fv


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
    fv_fv1 = [x - real_root for x in fv]
    plot.semilogy(range(len(fv_fv1)), fv_fv1)
    plot.legend(['hybrid'])
    plot.show()
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
    fv_fv1 = [((x[0] + x[1]) / 2) - real_root for x in fv]
    plot.semilogy(range(len(fv_fv1)), fv_fv1)
    plot.legend(['golden section'])
    plot.show()
    return x, fv


def ex2(l, u, eps):
    g = lambda x: -3.55 * x ** 3 + 1.1 * x ** 2 + 0.765 * x - 0.74
    dg = lambda x: -10.65 * x ** 2 + 2.2 * x + 0.765
    r = generic_hybrid(g, g, dg, l, u, eps, eps, 100)
    print("The result is " + str(r))
    return r


def gs_denoise_step(mu, a, b, c):
    f = lambda t: mu * (t - a) ** 2 + abs(t - b) + abs(t - c)
    t, _ = generic_gs(f, min(a, b, c) - 1, max(a, b, c) + 1, 10 ** -10, 100000)
    # print("The result is " + str(t))
    return t


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
    xn = x0
    fv = []

    for i in range(k - 2):
        fv.append(f(xn))
        dfxn = df(xn)
        ddfxn = ddf(xn)
        xn = xn - dfxn / ddfxn
        if ddfxn == 0:
            return None
        if np.abs(dfxn) < eps:
            break
    fv_fv1 = [x - real_root for x in fv]
    plot.semilogy(range(len(fv_fv1)), fv_fv1)
    plot.legend(['Newton'])
    plot.show()
    return xn, fv


def gs_denoise(s, alpha, N):
    import copy
    x = copy.deepcopy(s)
    for i in range(N):
        for k in range(len(s)):
            if k == 0 or k == len(s) - 1:
                bc = 0.5
                x[k] = gs_denoise_step(alpha, 2 * s[k], bc * x[k], bc * x[k])
            else:
                x[k] = gs_denoise_step(alpha, s[k], x[k + 1], x[k - 1])
    return x


if __name__ == '__main__':
    f_x = lambda x: x ** 2 + (x ** 2 - 3 * x + 10) / (2 + x)
    df = lambda x: (2 * x ** 3 + 9 * x ** 2 + 12 * x - 16) / ((2 + x) ** 2)
    ddf = lambda x: (2 * x ** 4 + 16 * x ** 3 + 48 * x ** 2 + 104 * x + 112) / ((x + 2) ** 4)

    eps = 10 ** (-6)
    lb = -1
    ub = 5
    k = 50
    x0 = (lb + ub) / 2
    x, fv = generic_bisect(f_x, df,lb, ub, eps , k)
    # x, fv = generic_newton(f_x, df, ddf, x0, eps, k)
    # x, fv = generic_hybrid(f_x, df, ddf, lb, ub, eps, eps, k)
    # x, fv = generic_gs(f_x, lb, ub, eps, k)
    # print("The result is " + str(x))
    # r = ex2(-1, 2, 10 ** -5)
    # print("The result is " + str(r))
    # plot.semilogy(indicies, fvGraph)
    # plot.show()
    # gs_denoise_step(2, 4, 8, 2)

    # plotting the real discrete signal
    # real_s_1 = [1.] * 40
    # real_s_0 = [0.] * 40
    #
    # plot.plot(range(40), real_s_1, 'black', linewidth=0.7)
    # plot.plot(range(41, 81), real_s_0, 'black', linewidth=0.7)
    #
    # # solving the problem
    # s = np.array([[1.] * 40 + [0.] * 40]).T + 0.1 * np.random.randn(80, 1)  # noised signal
    # x1 = gs_denoise(s, 0.5, 100)
    # x2 = gs_denoise(s, 0.5, 1000)
    # x3 = gs_denoise(s, 0.5, 10000)
    #
    # plot.plot(range(80), s, 'cyan', linewidth=0.7)
    # plot.plot(range(80), x1, 'red', linewidth=0.7)
    # plot.plot(range(80), x2, 'green', linewidth=0.7)
    # plot.plot(range(80), x3, 'blue', linewidth=0.7)
    # print("X1:", x1)
    # print("X2:", x2)
    # print("X3:", x3)
    # plot.show()
