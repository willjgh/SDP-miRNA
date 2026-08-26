'''
Module implementing optimized constructor for base model with updated compute_A() and compute_B() functions.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import math
import time
import numpy as np
import sympy as sp
from SDP_miRNA import utils
from SDP_miRNA import optimization_utils
import gurobipy as gp
from gurobipy import GRB
from functools import lru_cache
from collections import OrderedDict
from scipy.sparse import coo_matrix

# ------------------------------------------------
# Utils
# ------------------------------------------------

def compute_order(alpha):
    """Sum of elements of a power."""
    order = 0
    for alpha_i in alpha:
        order += alpha_i
    return order

def compute_Nd(S, d):
    """Number of moments of order <= d (S species)."""
    return math.factorial(S + d) // (math.factorial(d) * math.factorial(S))

def compute_powers_exact_order(S, d):
    """
    Compute the Nd powers of order <= d (S species), preserving the exact
    ordering of the original implementation while avoiding expensive repeated
    list-membership checks.
    """
    powers = [tuple(0 for _ in range(S))]
    powers_prev = [tuple(0 for _ in range(S))]

    for order in range(1, d + 1):
        seen = OrderedDict()
        for alpha in powers_prev:
            for i in range(S):
                alpha_new = list(alpha)
                alpha_new[i] += 1
                seen[tuple(alpha_new)] = None
        powers_current = list(seen.keys())
        powers.extend(powers_current)
        powers_prev = powers_current

    return powers

@lru_cache(None)
def stirling2(n, k):
    """Stirling number of the second kind."""
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0 or k > n:
        return 0
    if k == 1 or k == n:
        return 1
    return k * stirling2(n - 1, k) + stirling2(n - 1, k - 1)

@lru_cache(None)
def stirling1_signed(n, k):
    """Signed Stirling number of the first kind."""
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0 or k > n:
        return 0
    if n == k:
        return 1
    return stirling1_signed(n - 1, k - 1) - (n - 1) * stirling1_signed(n - 1, k)

# ------------------------------------------------
# B Functions
# ------------------------------------------------

@lru_cache(None)
def observed_species_terms(l):
    """
    Expand E[X^l] for X ~ Bin(n, p) as a polynomial in p and n:

        E[X^l] = sum_{k,m} coeff * p^k * n^m

    Returns a list of triples:
        (beta_power=k, x_power=m, coeff)
    """
    coeffs = {}
    for k in range(l + 1):
        s2 = stirling2(l, k)
        if s2 == 0:
            continue
        for m in range(k + 1):
            s1 = stirling1_signed(k, m)
            if s1 == 0:
                continue
            coeffs[(k, m)] = coeffs.get((k, m), 0) + s2 * s1

    return [(beta_pow, x_pow, coeff) for (beta_pow, x_pow), coeff in coeffs.items()]

def compute_B_optimized(beta, S, U, d):
    """
    Capture efficiency moment scaling matrix.

    This version preserves the exact powers ordering of the original
    compute_powers implementation, while avoiding SymPy polynomial algebra.
    It returns a SciPy CSR sparse matrix.

    Args:
        beta: per-cell capture efficiency sample
        S: number of species
        U: unobserved species indices
        d: maximum moment order used

    Returns:
        B: (Nd, Nd) sparse CSR matrix of coefficients
    """
    U = set(U)

    powers = compute_powers_exact_order(S, d)
    Nd = len(powers)
    power_to_idx = {alpha: i for i, alpha in enumerate(powers)}

    y_beta = np.array([np.mean(beta ** l) for l in range(d + 1)], dtype=float)

    zero_power = (0,) * S
    rows = []
    cols = []
    data = []

    for row, alpha in enumerate(powers):
        term_map = {(0, zero_power): 1.0}

        for i in range(S):
            a_i = alpha[i]

            if i in U:
                species_terms = [(0, a_i, 1.0)]
            else:
                species_terms = observed_species_terms(a_i)

            new_map = {}
            for (b0, xs0), c0 in term_map.items():
                for b1, x1, c1 in species_terms:
                    xs_new = list(xs0)
                    xs_new[i] += x1
                    xs_new = tuple(xs_new)

                    key = (b0 + b1, xs_new)
                    new_map[key] = new_map.get(key, 0.0) + c0 * c1

            term_map = new_map

        row_accum = {}
        for (beta_power, xs_power), coeff in term_map.items():
            col = power_to_idx[xs_power]
            value = coeff * y_beta[beta_power]
            if value != 0:
                row_accum[col] = row_accum.get(col, 0.0) + value

        for col, value in row_accum.items():
            if value != 0:
                rows.append(row)
                cols.append(col)
                data.append(value)

    B = coo_matrix((data, (rows, cols)), shape=(Nd, Nd), dtype=float).tocsr()
    B.sum_duplicates()
    B.eliminate_zeros()

    return B

# ------------------------------------------------
# A Functions
# ------------------------------------------------

@lru_cache(None)
def multinomial_poly_terms(alpha_tuple, shift_tuple):
    """
    Expand prod_i (x_i + shift_i)^alpha_i as a polynomial in x.

    Returns a dict:
        xs_power_tuple -> coefficient
    """
    S = len(alpha_tuple)
    term_map = {tuple(0 for _ in range(S)): 1}

    for i in range(S):
        a_i = alpha_tuple[i]
        v_i = shift_tuple[i]

        species_terms = []
        for k in range(a_i + 1):
            coeff = math.comb(a_i, k) * (v_i ** (a_i - k))
            if coeff != 0:
                power = [0] * S
                power[i] = k
                species_terms.append((tuple(power), coeff))

        new_map = {}
        for p0, c0 in term_map.items():
            for p1, c1 in species_terms:
                p_new = tuple(p0[j] + p1[j] for j in range(S))
                new_map[p_new] = new_map.get(p_new, 0) + c0 * c1
        term_map = new_map

    return term_map

@lru_cache(None)
def monomial_exponent_tuple(expr, xs_tuple):
    """Return exponent tuple for a monomial expression in xs_tuple."""
    powers = expr.as_powers_dict()
    return tuple(int(powers.get(x, 0)) for x in xs_tuple)

def parse_propensity_polynomial(expr, xs):
    """
    Parse a SymPy polynomial expression into a dict:
        xs_power_tuple -> coefficient
    """
    poly = sp.Poly(sp.expand(expr), *xs)
    return {tuple(mon): coeff for mon, coeff in zip(poly.monoms(), poly.coeffs()) if coeff != 0}

def compute_A_optimized(alpha, reactions, vrs, db, R, S, d):
    """
    Moment equation coefficient matrix.
    NOTE: must have order(alpha) <= d.

    This version preserves the exact powers ordering of the original
    compute_powers implementation, uses sparse output, replaces repeated
    linear searches by hash lookup, and avoids expanding the shifted monomial
    term separately for each reaction term.

    Args:
        alpha: moment order for equation (d/dt mu^alpha = 0)
        reactions: list of strings detailing a_r(x) for each reaction r
        vrs: array detailing v_r for each reaction r
        db: largest order a_r(x)
        R: number of reactions
        S: number of species
        d: maximum moment order used

    Returns:
        A: (R, Nd) sparse CSR matrix of coefficients
    """
    if compute_order(alpha) > d - db + 1:
        raise NotImplementedError(
            f"Maximum moment order {d} too small for moment equation of alpha = {alpha}: involves moments of higher order."
        )

    alpha = tuple(alpha)
    vrs = np.asarray(vrs)

    xs = sp.symbols([f'x{i}' for i in range(S)])

    powers = compute_powers_exact_order(S, d)
    Nd = len(powers)
    power_to_idx = {p: i for i, p in enumerate(powers)}

    x_alpha = tuple(alpha)

    rows = []
    cols = []
    data = []

    for r in range(R):
        prop_expr = sp.parse_expr(reactions[r], {'xs': xs})
        prop_terms = parse_propensity_polynomial(prop_expr, xs)

        shift = tuple(int(vrs[r, i]) for i in range(S))
        shifted_terms = multinomial_poly_terms(alpha, shift)

        delta_terms = dict(shifted_terms)
        delta_terms[x_alpha] = delta_terms.get(x_alpha, 0) - 1
        if delta_terms[x_alpha] == 0:
            del delta_terms[x_alpha]

        row_accum = {}
        for prop_power, prop_coeff in prop_terms.items():
            for delta_power, delta_coeff in delta_terms.items():
                total_power = tuple(prop_power[i] + delta_power[i] for i in range(S))
                col = power_to_idx[total_power]
                value = prop_coeff * delta_coeff
                row_accum[col] = row_accum.get(col, 0) + value

        for col, value in row_accum.items():
            if value != 0:
                rows.append(r)
                cols.append(col)
                data.append(float(value) if getattr(value, 'is_number', False) else value)

    A = coo_matrix((data, (rows, cols)), shape=(R, Nd)).tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()

    return A

# ------------------------------------------------
# General base model
# ------------------------------------------------

def base_model(opt, model, OB_bounds):
    '''
    Construct 'base model' with semidefinite constraints removed to give NLP

    Args:
        opt: Optimization class (or subclass), see relevant attributes
        model: empty gurobi model object
        OB_bounds: confidence intervals on observed moments up to order d (at least)

        Relevant class attributes

        beta: capture efficiency vector
        reactions: list of strings detailing a_r(x) for each reaction r
        vrs: array containing v_r for each reaction r
        db: largest order a_r(x)
        R: number of reactions
        S: number of species
        U: indices of unobserved species
        d: maximum moment order used
            d_bd: _ for moment bounds
            d_me: _ for moment equations
            d_sd: _ for semidefinite constraints
        fixed: list of pairs of (reaction index r, value to fix k_r to)
        time_limit: optimization time limit

        constraint options

        moment_bounds: CI bounds on moments
        moment_matrices: 
        moment_equations
        factorization
        factorization_telegraph
        telegraph_moments

    Returns:
        model: gurobi model object with NLP constraints (all but semidefinite)
        variables: dict for model variable reference
    '''

    ts = time.time()

    # model settings
    model.Params.TimeLimit = opt.time_limit

    # helpful values
    Nd = utils.compute_Nd(opt.S, opt.d)
    O = [i for i in range(opt.S) if i not in opt.U]
    SO = len(O)

    # variables
    y = model.addMVar(shape=Nd, vtype=GRB.CONTINUOUS, name="y", lb=0)
    if opt.K is None:
        k = model.addMVar(shape=opt.R, vtype=GRB.CONTINUOUS, name="k", lb=0)
    else:
        k = model.addMVar(shape=opt.R, vtype=GRB.CONTINUOUS, name="k", lb=0, ub=opt.K)

    # variable dict
    variables = {
        'y': y,
        'k': k
    }

    if opt.print_times: print(f"Variables: {time.time() - ts}")

    # moment matrices
    if opt.constraints.moment_matrices:

        ts = time.time()

        # for each species
        for s in range(opt.S + 1):

            # up to order d_sd
            M_s = optimization_utils.construct_M_s(y, s, opt.S, opt.d_sd)
            variables[f'M_{s}'] = M_s

        if opt.print_times: print(f"Moment Matrices {time.time() - ts}")
    
    # moment bounds
    if opt.constraints.moment_bounds:

        ts = time.time()
        to = 0
        tc = 0
        ta = 0

        so = time.time()

        # only explicitly bound observed, leave unobserved unbounded
        # avoids issues with e+100 upper bounds on unobserved moments

        so1 = time.time()

        # B scaling matrix
        B = compute_B_optimized(opt.dataset.beta, opt.S, opt.U, opt.d)

        to1 = time.time() - so1

        so2 = time.time()

        # downsampled moments
        y_D = B @ y

        to2 = time.time() - so2

        so3 = time.time()

        # powers up to order d_bd for all species & observed species
        powers_S = utils.compute_powers(opt.S, opt.d_bd)
        powers_SO = utils.compute_powers(SO, opt.d_bd)

        to3 = time.time() - so3

        to = time.time() - so

        # for all species powers
        for i, alpha_S in enumerate(powers_S):

            sc = time.time()

            # skip if contains unobserved species (non-zero power)
            unobserved = False
            for j in opt.U:
                if alpha_S[j] > 0:
                    unobserved = True
            if unobserved:
                continue

            # otherwise: bound

            # find corresponding index of observed species power
            alpha_SO = [alpha_S[i] for i in O]
            j = powers_SO.index(alpha_SO)

            tc += time.time() - sc

            sa = time.time()
    
            # bound
            model.addConstr(y_D[i] <= OB_bounds[1, j], name=f"y_{i}_UB")
            model.addConstr(y_D[i] >= OB_bounds[0, j], name=f"y_{i}_LB")

            ta += time.time() - sa

        if opt.print_times:

            print(f"Moment Bounds {time.time() - ts}")
            print(f"    Overhead: {to} ({to1}, {to2}, {to3})")
            print(f"    Code: {tc}")
            print(f"    Adding: {ta}")

    # moment equations
    if opt.constraints.moment_equations:

        ts = time.time()
        tc = 0
        ta = 0

        # moment equations of order up to d_me - db + 1
        # means d_me highest order moment invovled
        moment_powers = utils.compute_powers(opt.S, opt.d_me - opt.db + 1)
        for alpha in moment_powers:

            sc = time.time()

            # compute A as R x N_d, so no need to subset to d_me for product
            A_alpha_d = compute_A_optimized(alpha, opt.reactions, opt.vrs, opt.db, opt.R, opt.S, opt.d)

            tc += time.time() - sc

            sa = time.time()

            model.addConstr(k.T @ A_alpha_d @ y == 0, name=f"ME_{alpha}_{opt.d}")

            ta += time.time() - sa

        if opt.print_times:

            print(f"Moment Equations {time.time() - ts}")
            print(f"    Code: {tc}")
            print(f"    Adding: {ta}")

    # moment factorization: currently only for S = 2
    if opt.constraints.factorization:

        # check S = 2
        if opt.S != 2:
            print("Factorization of more than 2 species not supported")
            return None

        powers = utils.compute_powers(opt.S, opt.d)
        for i, alpha in enumerate(powers):

            # E[X1^a1 X2^a2] = E[X1^a1] E[X2^a2]
            if (alpha[0] > 0) and (alpha[1] > 0):
                j = powers.index([alpha[0], 0])
                l = powers.index([0, alpha[1]])
                model.addConstr(y[i] == y[j] * y[l], name=f"Moment_factorization_{alpha[0]}_({alpha[1]})")

    # constraints assuming unobserved species have state space {0, 1}
    if opt.constraints.unobserved_constraints:

        # for each power
        powers = utils.compute_powers(opt.S, opt.d)
        for i, alpha in enumerate(powers):

            # for each unobserved species
            for j in opt.U:

                # if unobserved species has power >= 1
                if alpha[j] > 1:

                    # set equal to moment with power = 1
                    alpha_reduced = alpha
                    alpha_reduced[j] = 1
                    l = powers.index(alpha_reduced)
                    model.addConstr(y[i] == y[l], name="U_eq")

                # if unobserved species has power 1
                elif alpha[j] == 1:

                    # constrain to less than moment with power 0
                    alpha_reduced = alpha
                    alpha_reduced[j] = 0
                    l = powers.index(alpha_reduced)
                    model.addConstr(y[i] <= y[l], name="U_ineq")

    # fixed moment
    model.addConstr(y[0] == 1, name="y0_base")

    # rate parameter constraints
    for r, val in opt.rate_fixed:
        model.addConstr(k[r] == val, name=f"k{r}_fixed")
    for r, val in opt.rate_lower:
        model.addConstr(k[r] >= val, name=f"k{r}_lower")
    for r, val in opt.rate_upper:
        model.addConstr(k[r] <= val, name=f"k{r}_upper")
 
    return model, variables
