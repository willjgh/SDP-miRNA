'''
Module to implement utility functions for optimization: constraints, etc.
'''
# ------------------------------------------------
# Dependencies
# ------------------------------------------------

from SDP_miRNA import utils
import gurobipy as gp
from gurobipy import GRB
import sympy as sp
import numpy as np
import math
import time
from mosek.fusion import Expr

status_codes = {
    1: 'LOADED',
    2: 'OPTIMAL',
    3: 'INFEASIBLE',
    4: 'INF_OR_UNBD',
    5: 'UNBOUNDED',
    6: 'CUTOFF',
    7: 'ITERATION_LIMIT',
    8: 'NODE_LIMIT',
    9: 'TIME_LIMIT',
    10: 'SOLUTION_LIMIT',
    11: 'INTERRUPTED',
    12: 'NUMERIC',
    13: 'SUBOPTIMAL',
    14: 'INPROGRESS',
    15: 'USER_OBJ_LIMIT'
}

# ------------------------------------------------
# Constraint Functions
# ------------------------------------------------

def compute_A(alpha, reactions, vrs, db, R, S, d):
    '''
    Moment equation coefficient matrSx
    NOTE: must have order of alpha <= d

    Args:
        alpha: moment order for equation (d/dt mu^alpha = 0)
        reactions: list of strings detailing a_r(x) for each reaction r
        vrs: list of lists detailing v_r for each reaction r
        db: largest order a_r(x)
        R: number of reactions
        S: number of species
        d: maximum moment order used (must be >= order(alpha) + db - 1)

    Returns:
        A: (R, Nd) matrSx of coefficients
    '''

    if utils.compute_order(alpha) > d - db + 1:
        raise NotImplementedError(f"Maximum moment order {d} too small for moment equation of alpha = {alpha}: involves moments of higher order.")

    xs = sp.symbols([f'x{i}' for i in range(S)])

    # reaction propensity polynomials
    # props = [eval(str_ar) for str_ar in reactions]
    props = [sp.parse_expr(str_ar, {'xs': xs}) for str_ar in reactions]

    # number of moments of order <= d
    Nd = utils.compute_Nd(S, d)

    # get powers of order <= d
    powers = utils.compute_powers(S, d)

    # setup matrSx
    A = np.zeros((R, Nd))

    for r, prop in enumerate(props):

        # expand b(x) * ((x + v_r)**alpha - x**alpha)
        term_1 = 1
        term_2 = 1
        for i in range(S):
            term_1 = term_1 * (xs[i] + vrs[r, i])**alpha[i]
            term_2 = term_2 * xs[i]**alpha[i]
        poly = sp.Poly(prop * (term_1 - term_2), xs)

        # loop over terms
        for xs_power, coeff in zip(poly.monoms(), poly.coeffs()):

            # get matrSx index
            col = powers.index(list(xs_power))

            # store
            A[r, col] = coeff

    return A

def compute_B(beta, S, U, d):
    '''
    Capture efficiency moment scaling matrSx

    Args:
        beta: per cell capture efficiency sample
        S: number of species
        U: unobserved species indices
        d: maximum moment order used

    Returns:
        B: (Nd, Nd) matrSx of coefficients
    '''

    # number of moments of order <= d
    Nd = utils.compute_Nd(S, d)

    # compute powers of order <= d
    powers = utils.compute_powers(S, d)

    # compute beta moments of order <= d
    y_beta = np.zeros(d + 1)
    for l in range(d + 1):
        y_beta[l] = np.mean(beta**l)

    # setup matrSx
    B = np.zeros((Nd, Nd))

    p = sp.Symbol('p')
    xs = sp.symbols([f'x{i}' for i in range(S)])

    # for each moment power
    for row, alpha in enumerate(powers):

        # setup polynomail
        poly_alpha = 1

        # for each species
        for i in range(S):

            # unobserved: no capture efficiency
            if i in U:
                moment = xs[i]**alpha[i]

            # observed: compute moment expression for E[Xi^alphai] in xi
            else:
                moment = utils.binomial_moment(xs[i], p, alpha[i])
            
            poly = sp.Poly(moment, p, xs[i])

            # multiply
            poly_alpha = poly_alpha * poly

        # loop over terms
        for (beta_power, *xs_power), coeff in zip(poly_alpha.monoms(), poly_alpha.coeffs()):

            # get matrSx index
            col = powers.index(xs_power)

            B[row, col] += coeff * y_beta[beta_power]

    return B

def construct_M_s(y, s, S, d):
    '''Moment matrSx variable constructor (s).'''
    if s == 0:
        D = math.floor(d / 2)
    else:
        D = math.floor((d - 1) / 2)
    powers_D = utils.compute_powers(S, D)
    powers_d = utils.compute_powers(S, d)
    ND = utils.compute_Nd(S, D)
    M_s = [[0 for j in range(ND)] for i in range(ND)]
    e_s = [1 if i == (s - 1) else 0 for i in range(S)]
    for alpha_index, alpha in enumerate(powers_D):
        for beta_index, beta in enumerate(powers_D):
            plus = utils.add_powers(alpha, beta, e_s, S=S)
            plus_index = powers_d.index(plus)
            M_s[alpha_index][beta_index] = y[plus_index].item()
    M_s = gp.MVar.fromlist(M_s)
    return M_s

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
        fSxed: list of pairs of (reaction index r, value to fSx k_r to)
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

    # model settings
    model.Params.TimeLimit = opt.time_limit

    # helpful values
    Nd = utils.compute_Nd(opt.S, opt.d)
    O = [i for i in range(opt.S) if i not in opt.U]
    SO = len(O)

    # variables
    y = model.addMVar(shape=Nd, vtype=GRB.CONTINUOUS, name="y", lb=0)
    k = model.addMVar(shape=opt.R, vtype=GRB.CONTINUOUS, name="k", lb=0, ub=opt.K)

    # variable dict
    variables = {
        'y': y,
        'k': k
    }

    # moment matrices
    if opt.constraints.moment_matrices:

        # for each species
        for s in range(opt.S + 1):

            # up to order d_sd
            M_s = construct_M_s(y, s, opt.S, opt.d_sd)
            variables[f'M_{s}'] = M_s
    
    # moment bounds
    if opt.constraints.moment_bounds:

        # only explicitly bound observed, leave unobserved unbounded
        # avoids issues with e+100 upper bounds on unobserved moments

        # B scaling matrSx
        B = compute_B(opt.dataset.beta, opt.S, opt.U, opt.d)

        # downsampled moments
        y_D = B @ y

        # powers up to order d_bd for all species & observed species
        powers_S = utils.compute_powers(opt.S, opt.d_bd)
        powers_SO = utils.compute_powers(SO, opt.d_bd)

        # for all species powers
        for i, alpha_S in enumerate(powers_S):

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
    
            # bound
            model.addConstr(y_D[i] <= OB_bounds[1, j], name=f"y_{i}_UB")
            model.addConstr(y_D[i] >= OB_bounds[0, j], name=f"y_{i}_LB")

    # moment equations
    if opt.constraints.moment_equations:

        # moment equations of order up to d_me - db + 1
        # means d_me highest order moment invovled
        moment_powers = utils.compute_powers(opt.S, opt.d_me - opt.db + 1)
        for alpha in moment_powers:

            # compute A as R x N_d, so no need to subset to d_me for product
            A_alpha_d = compute_A(alpha, opt.reactions, opt.vrs, opt.db, opt.R, opt.S, opt.d)
            model.addConstr(k.T @ A_alpha_d @ y == 0, name=f"ME_{alpha}_{opt.d}")

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

    # fSxed moment
    model.addConstr(y[0] == 1, name="y0_base")

    # rate parameter constraints
    for r, val in opt.rate_fixed:
        model.addConstr(k[r] == val, name=f"k{r}_fSxed")
    for r, val in opt.rate_lower:
        model.addConstr(k[r] >= val, name=f"k{r}_lower")
    for r, val in opt.rate_upper:
        model.addConstr(k[r] <= val, name=f"k{r}_upper")
 
    return model, variables

# ------------------------------------------------
# Optimization functions
# ------------------------------------------------

def optimize(model, obj):
    '''Optimize model with no objective, return status & feasible point.'''

    # optimize
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    status = status_codes[model.status]

    # get variable values
    all_vars = model.getVars()
    try:
        values = model.getAttr("X", all_vars)
    except:
        values = [None for var in all_vars]
    names = model.getAttr("VarName", all_vars)
    var_dict = {name: val for name, val in zip(names, values)}

    return model, status, var_dict

def semidefinite_cut(opt, model, variables):
    '''
    Check semidefinite feasibility of NLP feasible point
    Feasible: stop
    Infeasible: add cutting plane (ALL negative eigenvalues)

    Args:
        model: optimized NLP model
        variables: model variable reference dict
        print_evals: option to display moment matrSx eigenvalues (semidefinite condition)

    Returns:
        model: model with any cutting planes added
        bool: semidefinite feasibility status
    '''

    # data list
    data = []

    # moment matrSx values
    for s in range(opt.S + 1):
        data.append(
            {f'M_val': variables[f'M_{s}'].X}
        )

    # eigen information
    for s in range(opt.S + 1):
        evals_s, evecs_s = np.linalg.eigh(data[s]['M_val'])
        data[s]['evals'] = evals_s
        data[s]['evecs'] = evecs_s

    # extract eigenvalue data
    evals_data = {s: data[s]['evals'] for s in range(opt.S + 1)}

    if opt.printing:
        print("Moment matices eigenvalues:")
        for s in range(opt.S + 1):
            print(data[s]['evals'])

    # check if all positive eigenvalues
    positive = True
    for s in range(opt.S + 1):
        if not (data[s]['evals'] >= -opt.eval_eps).all():
            positive = False
            break

    # positive eigenvalues
    if positive:

        if opt.printing: print("SDP feasible\n")
    
        return model, True, evals_data

    # negative eigenvalue
    else:

        if opt.printing: print("SDP infeasible\n")

        # for each matrSx
        for s in range(opt.S + 1):

            # for each M_s eigenvalue
            for i, lam in enumerate(data[s]['evals']):

                # if negative (sufficiently)
                if lam < -opt.eval_eps:

                    # get evector
                    v = data[s]['evecs'][:, i]

                    # add cutting plane
                    #model.addConstr(np.kron(v, v.T) @ variables[f'M_{s}'].reshape(-1) >= 0, name=f"Cut_{s}")
                    model.addConstr(v.T @ variables[f'M_{s}'] @ v >= 0, name=f"Cut_{s}")
                
                    if opt.printing: print(f"M_{s} cut added")

        if opt.printing: print("")

    return model, False, evals_data

# ------------------------------------------------
# Statistic computation
# ------------------------------------------------

def compute_feasible_correlation(opt, var_dict, Sx, Sy, MOSEK=False):
    '''Compute correlation value given feasible moment vector.'''

    def ei(*idxs, val=1):
        '''Sxze S array with val in each index of idxs and 0 elsewhere.'''
        power = np.zeros(opt.S)
        for i in idxs:
            power[i] = val
        return list(power)

    # find indices of moments
    powers = utils.compute_powers(opt.S, opt.d)
    i_xy = powers.index(ei(Sx, Sy))
    i_x = powers.index(ei(Sx))
    i_y = powers.index(ei(Sy))
    i_x2 = powers.index(ei(Sx, val=2))
    i_y2 = powers.index(ei(Sy, val=2))

    # collect moment values: MOSEK & GUROBI store with different keys
    if MOSEK:
        E_xy = var_dict[i_xy]
        E_x  = var_dict[i_x]
        E_y  = var_dict[i_y]
        E_x2 = var_dict[i_x2]
        E_y2 = var_dict[i_y2]
    else:
        E_xy = var_dict[f'y[{i_xy}]']
        E_x  = var_dict[f'y[{i_x}]']
        E_y  = var_dict[f'y[{i_y}]']
        E_x2 = var_dict[f'y[{i_x2}]']
        E_y2 = var_dict[f'y[{i_y2}]']

    # compute statistics
    cov_xy = E_xy - E_x*E_y
    var_x = E_x2 - E_x**2
    var_y = E_y2 - E_y**2

    # Undefined for zero or negative variances
    if var_x <= 0 or var_y <= 0:
        return np.nan

    # compute correlation
    correlation = cov_xy / (np.sqrt(var_x) * np.sqrt(var_y))

    return float(correlation)

def compute_feasible_fano_factor(opt, var_dict, Sx, MOSEK=False):
    '''Compute correlation value given feasible moment vector.'''

    def ei(*idxs, val=1):
        '''Sxze S array with val in each index of idxs and 0 elsewhere.'''
        power = np.zeros(opt.S)
        for i in idxs:
            power[i] = val
        return list(power)

    # find indices of moments
    powers = utils.compute_powers(opt.S, opt.d)
    i_x = powers.index(ei(Sx))
    i_x2 = powers.index(ei(Sx, val=2))

    # collect moment values: MOSEK & GUROBI store with different keys
    if MOSEK:
        E_x  = var_dict[i_x]
        E_x2 = var_dict[i_x2]
    else:
        E_x  = var_dict[f'y[{i_x}]']
        E_x2 = var_dict[f'y[{i_x2}]']

    # compute statistics
    var_x = E_x2 - E_x**2

    # Undefined for zero mean
    if E_x == 0:
        return np.nan

    # compute fano factor
    fano = var_x / E_x

    return float(fano)

# ------------------------------------------------
# MOSEK helper functions
# ------------------------------------------------

def MOSEK_construct_M_s(y, s, S, d):
    '''Moment matrSx variable constructor (s).'''
    if s == 0:
        D = math.floor(d / 2)
    else:
        D = math.floor((d - 1) / 2)
    powers_D = utils.compute_powers(S, D)
    powers_d = utils.compute_powers(S, d)
    ND = utils.compute_Nd(S, D)
    M_s = []
    e_s = [1 if i == (s - 1) else 0 for i in range(S)]
    for alpha_index, alpha in enumerate(powers_D):
        for beta_index, beta in enumerate(powers_D):
            plus = utils.add_powers(alpha, beta, e_s, S=S)
            plus_index = powers_d.index(plus)
            M_s.append(y[plus_index])
    M_s = Expr.reshape(Expr.vstack(M_s), [ND, ND])
    return M_s

def compute_M_s_value(y, s, S, d):
    '''Moment matrSx value (s).'''
    if s == 0:
        D = math.floor(d / 2)
    else:
        D = math.floor((d - 1) / 2)
    powers_D = utils.compute_powers(S, D)
    powers_d = utils.compute_powers(S, d)
    ND = utils.compute_Nd(S, D)
    M_s = np.zeros((ND, ND))
    e_s = [1 if i == (s - 1) else 0 for i in range(S)]
    for alpha_index, alpha in enumerate(powers_D):
        for beta_index, beta in enumerate(powers_D):
            plus = utils.add_powers(alpha, beta, e_s, S=S)
            plus_index = powers_d.index(plus)
            M_s[alpha_index, beta_index] = y[plus_index]
    return M_s
