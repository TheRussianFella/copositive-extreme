import networkx as nx
import numpy as np
from collections import defaultdict
import itertools
import functools
from functools import reduce
import operator

def product(iterable):
    return reduce(operator.mul, iterable, 1)

## Graph processing functions

def find_stable_sets(adj_matrix):
    
    n = adj_matrix.shape[0]
    I = []
    
    for i in range(0, n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                
                if not ( adj_matrix[i,j] or adj_matrix[j,k] 
                        or adj_matrix[i,k]):
                    I.append(set([i,j,k]))
    return I

def has_4_ss(adj_matrix):

  n = adj_matrix.shape[0]

  for i in range(0, n):
        for j in range(i+1, n):
            for k in range(j+1, n):
              for l in range(k+1, n):
                
                if not (adj_matrix[i,j] or adj_matrix[j,k] 
                        or adj_matrix[i,k] or adj_matrix[i,l] 
                        or adj_matrix[j,l] or adj_matrix[k,l]):
                  return True


def is_copirr(adj_matrix):
  
  n = adj_matrix.shape[0]

  # check if it's symmetrical
  if (adj_matrix != adj_matrix.T).sum() != 0:
    return False

  # check if there is no stable set of size 4
  if has_4_ss(adj_matrix):
    return False
  
  # check alpha critical and alpha covered
  for i in range(0, n):
        for j in range(i+1, n):

          if adj_matrix[i,j]:
            adj_matrix[i,j] = adj_matrix[j,i] = 0

            if not has_4_ss(adj_matrix):
              adj_matrix[i,j] = adj_matrix[j,i] = 1
              return False

            adj_matrix[i,j] = adj_matrix[j,i] = 1

          else:
            ok = False
            for k in range(n):
              if (not adj_matrix[i,k]) and (not adj_matrix[j,k]):
                ok = True
            
            if not ok:
              return False
    
  return True

## Graph creation

def get_graphs():
    
    ALL_GRAPHS = {}
    
    cycle = np.array([
    [1, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 1, 1],
    [1, 0, 0, 1, 0, 0, 1],
    [1, 1, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 1]
    ])
    ALL_GRAPHS['cycle'] = cycle
    
    dupl = np.array([
    [1, 0, 0, 1, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1],
    [1, 0, 0, 1, 0, 0, 1, 1]
    ])
    ALL_GRAPHS['dupl'] = dupl
    
    zigzag = np.array([
    [1, 1, 0, 0, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 1, 0, 1, 1]
    ])
    ALL_GRAPHS['zigzag'] = zigzag
    
    fork = np.array([
    [1, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 1, 1]
    ])
    ALL_GRAPHS['fork'] = fork
    
    big_zig = np.array([
    [1, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1]
    ])
    ALL_GRAPHS['big_zig'] = big_zig
    
    big_triag = np.array([
    [1, 1, 0, 0, 1, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 1, 1, 1]
    ])
    ALL_GRAPHS['big_triag'] = big_triag
    
    ya_big_triag = np.array([
    [1, 1, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 1, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 1]
    ])
    ALL_GRAPHS['ya_big_triag'] = ya_big_triag
    
    ya_big_triag_2 = np.array([
    [1, 1, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 1]
    ])
    ALL_GRAPHS['ya_big_triag_2'] = ya_big_triag_2
    
    tang_triag = np.array([
    [1, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 1, 1, 1]
    ])
    ALL_GRAPHS['tang_triag'] = tang_triag
    
    square = np.array([
    [1, 1, 0, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1]
    ])
    ALL_GRAPHS['square'] = square
    
    center = np.array([
    [1, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 1]
    ])
    ALL_GRAPHS['center'] = center
    
    ship = np.array([
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 1, 1, 1]
    ])
    ALL_GRAPHS['ship'] = ship
    
    sinking_ship = np.array([
    [1, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 1, 0, 1, 1]
    ])
    ALL_GRAPHS['sinking_ship'] = sinking_ship
    
    storm = np.array([
      [1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
      [1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
      [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
      [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
      [1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
      [0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
      [1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
      [0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
      [0, 0, 1, 0, 1, 0, 0, 1, 1, 1]
    ])
    ALL_GRAPHS['storm'] = storm
    
    cell = np.array([
    [1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 0, 1, 1]
    ])
    ALL_GRAPHS['cell'] = cell
    
    return ALL_GRAPHS

## Functions for building equalities+inequalities systems for specific graphs

def build_triple_equalities(I, n):

  # Give numbers to all cells in a matrix
  var_matr   = np.ones([n,n], dtype=np.int)*(-1)
  num_params = int((n*(n-1))/2)
  c = 0

  for i in range(n):
    for j in range(i+1, n):
      var_matr[i][j] = c
      var_matr[j][i] = c
      c += 1
  
  # Now make equalities
  param_eq = []

  for ss in I:
    s = ss.copy()
    i, j, k = s.pop(), s.pop(), s.pop()
    param_eq.append([var_matr[i,j], var_matr[i,k], var_matr[j,k]])
  
  # And put them in a matrix
  A = []
  for p in param_eq:
    ai = np.zeros(num_params)
    ai[p[0]] = ai[p[1]] = ai[p[2]] = 1
    A.append(ai)
  
  return A, var_matr, num_params 

def total_num_of_combinations(I, var_matr, num_params, only_nonequivalent=True):
    
    # Find parametrizations for elements of a matrix that correspond to an edge
    variants = defaultdict(list)

    for x in range(len(I)):
        for y in range(x+1, len(I)):
            if len(I[x].intersection(I[y])) == 2:
                inter = I[x].intersection(I[y])
                i = I[x].difference(I[y]).pop()
                j = inter.pop()
                k = inter.pop()
                l = I[y].difference(I[x]).pop()

                variants[var_matr[i,l]].append([var_matr[i,j], var_matr[j,k], var_matr[k,l]])
    
    if only_nonequivalent:
        return product([2**(len(var)-1) for element, var in variants.items()])
    else:
        return product([2**(len(var)) for element, var in variants.items()])
    
def one_param_iterator(k):
  for x in itertools.product([-1,1], repeat=k-1):
    yield (1,) + x

def build_variance_equalities_iterator(I, var_matr, num_params, only_nonequivalent=True):

  # Find parametrizations for elements of a matrix that correspond to an edge
  variants = defaultdict(list)

  for x in range(len(I)):
    for y in range(x+1, len(I)):
      if len(I[x].intersection(I[y])) == 2:

        inter = I[x].intersection(I[y])
        i = I[x].difference(I[y]).pop()
        j = inter.pop()
        k = inter.pop()
        l = I[y].difference(I[x]).pop()

        variants[var_matr[i,l]].append([var_matr[i,j], var_matr[j,k], var_matr[k,l]])
  
  variant_keys = list(variants.keys())

  # Give away equalities that represent different module openings

  if only_nonequivalent:
    comb_iter = itertools.product(*[one_param_iterator(len(variants[x])) for x in variant_keys])
  else:
    comb_iter = itertools.product(*[itertools.product([-1,1], repeat=len(variants[x])) for x in variant_keys])


  for p in comb_iter:

    perm = functools.reduce(lambda x, y: x+y, p)
    A = []; b = []
    c = 0
    openings = defaultdict(list)
    
    for edge_key in variant_keys:

      var = variants[edge_key]

      for k in range(len(var)):

        ai = np.zeros(num_params)
        ai[edge_key] = 1
        bi = 0

        if perm[c] == 1:
          ai[var[k]] = -1; bi = 0
          openings[edge_key].append((var[k], 1))
        elif perm[c] == -1:
          ai[var[k]] = 1; bi = 2
          openings[edge_key].append((var[k], -1))
        
        c += 1
        A.append(ai)
        b.append(bi)
    
    
    yield A, b, openings

def build_inequalities(num_params):

  B = np.concatenate([np.identity(num_params), -1*np.identity(num_params)])
  c = np.array([1 for _ in range(num_params)] + [0 for _ in range(num_params)])

  return B, c