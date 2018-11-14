module TimeSchemes

"""
  function newmark(M, B, K, F, v0, v1, t; β = 0.25, γ = 0.5, outputNodes = [])

Solve a second order in time system of ODES using a Newmark discretization scheme.

The form of the system is as follows:
    v(0) = v_0,   v'(0) = v_1
"""
function newmark(M, B, K, F, v0, v1, t; β = 0.25, γ = 0.5, outputNodes = [])
  # β and γ are the parameters of the scheme


  N = length(v0) # number of equations in the system
  T = length(t)  # number of discretization times

  # ASSUMPTION : t is a equidistributed vector
  dt = t[2] - t[1]
  V = []
  if length(outputNodes) != 0
    V = zeros(length(outputNodes), T)
  else
    V = zeros(2 * N, T) # solution to return
    outputNodes = 1:(2*N)
  end

  p = 1:N               # first N components of the solution v
  q = (N + 1):(2 * N)   # last  N components of the solution v'

  cV = zeros(2 * N, 1)
  # initialization with initial contion
  cV[p] = v0                       # initial position
  cV[q] = v1                       # initial velocity
  a = M \ (F[:,1] - B * v1 - K * v0) # initial acceleration
  V[:, 1] = cV[outputNodes]

  MI = M + γ * dt * B + β * dt^2 * K

  # for details about this scheme see [Hughes, p. 490]
  for n = 2:T
    d = cV[p] + dt * cV[q] + dt^2 / 2 * (1 - 2 * β) * a
    v = cV[q] + (1 - γ) * dt * a
    a = MI \ convert(Array{Float64, 1}, F[:, n] - B * v - K * d)
    cV[p] = d + β * dt^2 * a
    cV[q] = v + γ * dt * a
    V[:, n] = cV[outputNodes]
  end
  V, cV
end


end # module
