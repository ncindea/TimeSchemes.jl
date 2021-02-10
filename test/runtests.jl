using TimeSchemes
using Test

##
M = 1
B = 1
K = 1
v0 = [1.]
v1 = [0.]
t = convert(Array{Float64},range(0, stop=1, length=10001))'
F = zeros(length(t))'
v, cV = TimeSchemes.newmark(M, B, K, F, v0, v1, t)

sol = 1 / 3 * exp.(- t / 2) .* (sqrt(3) * sin.(sqrt(3) / 2 * t) + 3  * cos.(sqrt(3) / 2 * t))
sold = - 1 / 6 * exp.(- t / 2) .* (sqrt.(3) * sin.(sqrt(3) / 2 * t) + 3  * cos.(sqrt(3) / 2 * t)) +
    1 / 3 * exp.(- t / 2) .* (3 / 2 * cos.(sqrt(3) / 2 * t) - 3 * sqrt(3) / 2 * sin.(sqrt(3) / 2 * t))
@test (maximum(abs.(v[1, :]' - sol)) < 1e-9) && (maximum(abs.(v[2, :]' - sold)) < 1e-9)
println("All tests passed.")
