module TimeSchemes

"
    function newmark(M, B, K, F, v0, v1, t)
    Solve a second order in time system of ODES using a Newmark discretization scheme.

    The form of the system is as follows:
    M * v''(t) + B * v'(t) + K * v(t) = F(t)
    v(0) = v_0,   v'(0) = v_1
"
function newmark(M, B, K, F, v0, v1, t)
  # β and γ are the parameters of the scheme
  β = 0.25
  γ = 0.5


  N = length(v0) # number of equations in the system
  T = length(t)  # number of discretization times

  # ASSUMPTION : t is a equidistributed vector
  dt = t[2] - t[1]

  V = zeros(2 * N, T) # solution to return

  p = 1:N               # first N components of the solution v
  q = (N + 1):(2 * N)   # last  N components of the solution v'

  # initialization with initial contion
  V[p, 1] = v0                       # initial position
  V[q, 1] = v1                       # initial velocity
  a = M \ (F[:,1] - B * v1 - K * v0) # initial acceleration

  MI = M + γ * dt * B + β * dt^2 * K

  # for details about this scheme see [Hughes, p. 490]
  for n = 2:T
    d = V[p, n - 1] + dt * V[q, n - 1] + dt^2 / 2 * (1 - 2 * β) * a
    v = V[q, n - 1] + (1 - γ) * dt * a
    a = MI \ (F[:, n] - B * v - K * d)
    V[p, n] = d + β * dt^2 * a
    V[q, n] = v + γ * dt * a
  end
  V
end


end # module
