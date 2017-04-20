using TimeSchemes
using Base.Test
##
M = 1
B = 1
K = 1
v0 = 1
v1 = 0
t = linspace(0, 1, 1001)
F = zeros(t)
v = TimeSchemes.newmark(M, B, K, F, v0, v1, t)
