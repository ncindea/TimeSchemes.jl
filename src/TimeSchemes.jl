module TimeSchemes
using LinearAlgebra
using SparseArrays

"""
  function Crank-nicolson(M, K, F, v0, t; outputNodes = [])

Solve a linear first order in time system of ODES using the crank-nicolson discretization scheme.

The form of the system is as follows:
    M v'(t) + K v(t) = F(t)
    v(0) = v_0
"""
function crank_nicolson(M, K, F, v0, t; outputNodes = [])
    N = length(v0) # number of equations in the system
    T = length(t)  # number of discretization times
    
    # ASSUMPTION : t is a equidistributed vector
    dt = t[2] - t[1]
    V = []
    if length(outputNodes) != 0
        V = zeros(length(outputNodes), T)
    else
        V = zeros(N, T) # solution to return
        outputNodes = 1:N
    end
    cV = v0
    V[outputNodes, 1] = cV[outputNodes]
    MI = M + dt/2 * K
    MO = M - dt/2 * K
    for n = 2:T
        cV = MI \ convert(Array{Float64, 1}, dt*(F[:, n-1]+F[:,n])/2 + MO* cV)
        V[outputNodes, n] = cV[outputNodes]
    end
    V, cV
end



"""
  function implicit_euler(M, K, F, v0, t; outputNodes = [])

Solve a linear first order in time system of ODES using an implicit Euler discretization scheme.

The form of the system is as follows:
    M v'(t) + K v(t) = F(t)
    v(0) = v_0
"""
function implicit_euler(M, K, F, v0, t; outputNodes = [])
    N = length(v0) # number of equations in the system
    T = length(t)  # number of discretization times
    
    # ASSUMPTION : t is a equidistributed vector
    dt = t[2] - t[1]
    V = []
    if length(outputNodes) != 0
        V = zeros(length(outputNodes), T)
    else
        V = zeros(N, T) # solution to return
        outputNodes = 1:N
    end
    cV = v0
    V[outputNodes, 1] = cV[outputNodes]
    MI = M + dt * K
    for n = 2:T
        cV = MI \ convert(Array{Float64, 1}, dt*F[:, n] + M * cV)
        V[outputNodes, n] = cV[outputNodes]
    end
    V, cV
end

"""
  function explicit_euler(M, K, F, v0, t; outputNodes = [])

Solve a linear first order in time system of ODES using an implicit Euler discretization scheme.

The form of the system is as follows:
    M v'(t) + K v(t) = F(t)
    v(0) = v_0
"""
function explicit_euler(M, K, F, v0, t; outputNodes = [])
    N = length(v0) # number of equations in the system
    T = length(t)  # number of discretization times
    
    # ASSUMPTION : t is a equidistributed vector
    dt = t[2] - t[1]
    V = []
    if length(outputNodes) != 0
        V = zeros(length(outputNodes), T)
    else
        V = zeros(N, T) # solution to return
        outputNodes = 1:N
    end
    cV = v0
    V[outputNodes, 1] = cV[outputNodes]
    MI = M - dt * K
    for n = 2:T
        cV = M \ convert(Array{Float64, 1}, dt*F[:, n-1] + MI * cV)
        V[outputNodes, n] = cV[outputNodes]
    end
    V, cV
end

"""
  function newmark(M, B, K, F, v0, v1, t; β = 0.25, γ = 0.5, outputNodes = [])

Solve a second order in time system of ODES using a Newmark discretization scheme.

The form of the system is as follows:
    M v''(t) + K v(t) = F(t)
    v(0) = v_0,   v'(0) = v_1
"""
function newmark(M, B, K, F, v0, v1, t; β = 0.25, γ = 0.5, outputNodes = [], B0 = B)
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
  a = M \ (F[:,1] - B0 * v1 - K * v0) # initial acceleration
  V[:, 1] = cV[outputNodes]

  MI = M + γ * dt * B + β * dt^2 * K
  FF = lu(MI)  

  # for details about this scheme see [Hughes, p. 490]
  for n = 2:T
    d = cV[p] + dt * cV[q] + dt^2 / 2 * (1 - 2 * β) * a
    v = cV[q] + (1 - γ) * dt * a
    a = FF \ convert(Array{Float64, 1}, F[:, n] - B * v - K * d)
    cV[p] = d + β * dt^2 * a
    cV[q] = v + γ * dt * a
    V[:, n] = cV[outputNodes]
  end
  V, cV
end


"""
  function explicit2(K, F, v0, v1, t; outputNodes = [])

Solve a second order in time system of ODES using an explicit discretization scheme.

The form of the system is as follows:
    v''(t) + K v(t) = F(t)
    v(0) = v_0,   v'(0) = v_1
"""
function explicit2(K, F, v0, v1, t; outputNodes = [])


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
    a = F[:,1] - K * v0 # initial acceleration
    V[:, 1] = cV[outputNodes]
    V[p, 2] = v0 + dt * v1 + dt^2 / 2 * a
    a = F[:, 2] - K*V[p, 2]
    V[q, 2] = (V[p, 2] - V[p, 1]) / dt + dt / 2 * a


    for n = 2:T-1
        V[p, n+1] = 2*V[p, n] - V[p, n - 1] + dt^2 * a
        a = F[:, n+1] - K*V[p, n+1]
        V[q, n+1] = (V[p, n+1] - V[p, n]) / dt + dt / 2 * a
        
        cV = V[:, n+1]
    end
    V, cV
end


end # module
