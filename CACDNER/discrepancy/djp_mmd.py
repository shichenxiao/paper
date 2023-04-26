# -*- coding: utf-8 -*-
# @Time    : 2020/7/4 19:18
# @Author  : wenzhang
# @File    : djp_mmd.py


import torch as tr


def _primal_kernel(Xs, Xt):
    Z = tr.cat((Xs.T, Xt.T), 1)  # Xs / Xt: batch_size * k
    return Z


def _linear_kernel(Xs, Xt):
    Z = tr.cat((Xs, Xt), 0)  # Xs / Xt: batch_size * k
    K = tr.mm(Z, Z.T)
    return K


def _rbf_kernel(Xs, Xt, sigma):
    Z = tr.cat((Xs, Xt), 0)
    ZZT = tr.mm(Z, Z.T)  #Z乘以Z的转置
    diag_ZZT = tr.diag(ZZT).unsqueeze(1) # tr.diag:取矩阵对角线元素
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.T
    K = tr.exp(-exponent / (2 * sigma ** 2))
    return K


# functions to compute the marginal MMD with rbf kernel
def rbf_mmd(Xs, Xt, sigma):
    K = _rbf_kernel(Xs, Xt, sigma)
    m = Xs.size(0)  # assume Xs, Xt are same shape
    e = tr.cat((1 / m * tr.ones(m, 1), -1 / m * tr.ones(m, 1)), 0)
    M = e * e.T
    tmp = tr.mm(tr.mm(K.cpu(), M.cpu()), K.T.cpu())
    loss = tr.trace(tmp).cuda()
    return loss


# functions to compute rbf kernel JMMD
def rbf_jmmd(Xs, Ys, Xt, Yt0, sigma):
    K = _rbf_kernel(Xs, Xt, sigma)
    n = K.size(0)
    m = Xs.size(0)  # assume Xs, Xt are same shape
    e = tr.cat((1 / m * tr.ones(m, 1), -1 / m * tr.ones(m, 1)), 0)
    C = len(tr.unique(Ys))
    M = e * e.T * C
    for c in tr.unique(Ys):
        e = tr.zeros(n, 1)
        e[:m][Ys == c] = 1 / len(Ys[Ys == c])
        if len(Yt0[Yt0 == c]) == 0:
            e[m:][Yt0 == c] = 0
        else:
            e[m:][Yt0 == c] = -1 / len(Yt0[Yt0 == c])
        M = M + e * e.T
    M = M / tr.norm(M, p='fro')  # can reduce the training loss only for jmmd
    tmp = tr.mm(tr.mm(K.cpu(), M.cpu()), K.T.cpu())
    loss = tr.trace(tmp).cuda()
    return loss


# functions to compute rbf kernel JPMMD
def rbf_jpmmd(Xs, Ys, Xt, Yt0, sigma):
    Xs = tr.reshape(Xs, (-1, 768))
    Xt = tr.reshape(Xt, (-1, 768))
    Ys = Ys.view(-1)
    Yt0 = Yt0.view(-1).cuda()
    K = _rbf_kernel(Xs, Xt, sigma)
    n = K.size(0)
    m = Xs.size(0)  # assume Xs, Xt are same shape
    M = 0
    for c in tr.unique(Ys):
        e = tr.zeros(n, 1)
        e[:m] = 1 / len(Ys)
        if len(Yt0[Yt0 == c]) == 0:
            e[m:] = 0
        else:
            e[m:] = -1 / len(Yt0)
        M = M + e * e.T
    K = K.cuda()
    M = M.cuda()

    tmp = tr.mm(tr.mm(K, M), K.T)
    # tmp = tr.mm(tr.mm(K.cpu(), M.cpu()), K.T.cpu())
    loss = tr.trace(tmp).cuda()
    return loss


# functions to compute rbf kernel DJP-MMD
def rbf_djpmmd(Xs, Ys, Xt, Yt0, sigma):
    Xs = tr.reshape(Xs, (-1, 768))
    Xt = tr.reshape(Xt, (-1, 768))
    Xs = Xs.tolist()
    Xt = Xt.tolist()
    Ys = Ys.view(-1)
    Ys = Ys.tolist()
    Yt0 = Yt0.tolist()
    Ys_len = len(Ys)
    Ys_temp = []
    Xs_temp = []
    for i in range(Ys_len):
        if Ys[i] != -1 and Ys[i] != 8:
            Ys_temp.append(Ys[i])
            Xs_temp.append(Xs[i])
    Yt0_temp = []
    Xt_temp = []
    for i in range(len(Yt0)):
        if Yt0[i] != -1 and Yt0[i] != 8:
            Yt0_temp.append(Yt0[i])
            Xt_temp.append(Xt[i])
    Xs = tr.tensor(Xs_temp)
    Xt = tr.tensor(Xt_temp)
    Ys = tr.tensor(Ys_temp)
    Yt0 = tr.tensor(Yt0_temp)
    # Yt0 = Yt0.view(-1)
    K = _rbf_kernel(Xs, Xt, sigma)  # (x.size(0)+y.size(0),x.size(0)+y.size(0))
    # K = _linear_kernel(Xs, Xt)  # bad performance
    m = Xs.size(0)
    n = Xt.size(0)
    C = 8  # len(tr.unique(Ys))

    # For transferability
    Ns = 1 / m * tr.zeros(m, C).scatter_(1, Ys.unsqueeze(1).cpu(), 1)
    Nt = tr.zeros(n, C)
    if len(tr.unique(Yt0)) == 1:
        Nt = 1 / n * tr.zeros(n, C).scatter_(1, Yt0.unsqueeze(1).cpu(), 1)
    # Ns.cuda()
    # Nt.cuda()
    Rmin_1 = tr.cat((tr.mm(Ns, Ns.T), tr.mm(-Ns, Nt.T)), 1)
    Rmin_2 = tr.cat((tr.mm(-Nt, Ns.T), tr.mm(Nt, Nt.T)), 1)
    Rmin = tr.cat((Rmin_1, Rmin_2), 0)

    # For discriminability
    Ms = tr.empty(m, (C - 1) * C)
    Mt = tr.empty(n, (C - 1) * C)
    for i in range(0, C):
        idx = tr.arange((C - 1) * i, (C - 1) * (i + 1))
        Ms[:, idx] = Ns[:, i].repeat(C - 1, 1).T
        tmp = tr.arange(0, C)
        Mt[:, idx] = Nt[:, tmp[tmp != i]]
    Rmax_1 = tr.cat((tr.mm(Ms, Ms.T), tr.mm(-Ms, Mt.T)), 1)
    Rmax_2 = tr.cat((tr.mm(-Mt, Ms.T), tr.mm(Mt, Mt.T)), 1)
    Rmax = tr.cat((Rmax_1, Rmax_2), 0)
    M = Rmin - 0.1 * Rmax
    # tmp = tr.mm(tr.mm(K.cpu(), M.cpu()), K.T.cpu())
    K = K.cuda()
    M = M.cuda()

    tmp = tr.mm(tr.mm(K, M), K.T)
    loss = tr.trace(tmp.cuda())

    return loss

# def rbf_djpmmd(Xs, Ys, Xt, Yt0, sigma):
#     Xs = tr.reshape(Xs,(-1,768))
#     Xt = tr.reshape(Xt, (-1, 768))
#     Ys = Ys.view(-1)
#     K = _rbf_kernel(Xs, Xt, sigma) #(x.size(0)+y.size(0),x.size(0)+y.size(0))
#     # K = _linear_kernel(Xs, Xt)  # bad performance
#     m = Xs.size(0)
#     # n = Xt.size(0)
#     C =20  # len(tr.unique(Ys))
#
#     # For transferability
#     Ns = 1 / m * tr.zeros(m, C).scatter_(1, Ys.unsqueeze(1).cpu(), 1)
#     Nt = tr.zeros(m, C)
#     if len(tr.unique(Yt0)) == 1:
#         Nt = 1 / m * tr.zeros(m, C).scatter_(1, Yt0.unsqueeze(1).cpu(), 1)
#     Rmin_1 = tr.cat((tr.mm(Ns, Ns.T), tr.mm(-Ns, Nt.T)), 0)
#     Rmin_2 = tr.cat((tr.mm(-Nt, Ns.T), tr.mm(Nt, Nt.T)), 0)
#     Rmin = tr.cat((Rmin_1, Rmin_2), 1)
#
#     # For discriminability
#     Ms = tr.empty(m, (C - 1) * C)
#     Mt = tr.empty(m, (C - 1) * C)
#     for i in range(0, C):
#         idx = tr.arange((C - 1) * i, (C - 1) * (i + 1))
#         Ms[:, idx] = Ns[:, i].repeat(C - 1, 1).T
#         tmp = tr.arange(0, C)
#         Mt[:, idx] = Nt[:, tmp[tmp != i]]
#     Rmax_1 = tr.cat((tr.mm(Ms, Ms.T), tr.mm(-Ms, Mt.T)), 0)
#     Rmax_2 = tr.cat((tr.mm(-Mt, Ms.T), tr.mm(Mt, Mt.T)), 0)
#     Rmax = tr.cat((Rmax_1, Rmax_2), 1)
#     M = Rmin - 0.1 * Rmax
#     K=K.cuda()
#     M=M.cuda()
#     # tmp = tr.mm(tr.mm(K.cpu(), M.cpu()), K.T.cpu())
#     tmp = tr.mm(tr.mm(K, M), K.T)
#     loss = tr.trace(tmp.cuda())
#
#     return -loss

