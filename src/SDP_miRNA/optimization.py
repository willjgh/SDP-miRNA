'''
Module implementing classes to handle optimization inference method.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

from SDP_miRNA import optimization_utils
from SDP_miRNA import utils
from SDP_miRNA.constraints import Constraint
import json
import tqdm
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import traceback
import time

# ------------------------------------------------
# Constants
# ------------------------------------------------

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
# General Optimization class
# ------------------------------------------------

class Optimization():
    def __init__(
        self,
        dataset,
        constraints,
        reactions,
        vrs,
        db,
        R,
        S,
        U,
        rate_fixed=[],
        rate_lower=[],
        rate_upper=[],
        d=None,
        d_bd=None,
        d_me=None,
        d_sd=None,
        license_file=None,
        time_limit=300,
        total_time_limit=300,
        eval_eps=10**-6,
        cut_limit=100,
        K=np.inf,
        custom_constraint=None,
        objective_function=None,
        save_model=False,
        load_model=False,
        silent=True,
        printing=False,
        tqdm_disable=False
        ):
        '''Initialize analysis settings and result storage.'''
        
        # store reference to dataset
        self.dataset = dataset

        # model settings
        self.constraints = constraints
        self.reactions = reactions
        self.vrs = vrs
        self.db = db
        self.R = R
        self.S = S
        self.U = U
        self.rate_fixed = rate_fixed
        self.rate_lower = rate_lower
        self.rate_upper = rate_upper

        # moment order settings
        if d is not None:
            self.d = d
            self.d_bd = d
            self.d_me = d
            self.d_sd = d
        elif (d_bd is not None) and (d_me is not None) and (d_sd is not None):
            self.d = max(d_bd, d_me, d_sd)
            self.d_bd = d_bd
            self.d_me = d_me
            self.d_sd = d_sd
        else:
            raise Exception(f"No moment order specified")

        # optimization settings
        self.license_file = license_file
        self.time_limit = time_limit
        self.total_time_limit = total_time_limit
        self.eval_eps = eval_eps
        self.cut_limit = cut_limit
        self.K = K
        self.custom_constraint = custom_constraint
        self.objective_function = objective_function

        # file settings
        self.save_model = save_model
        self.load_model = load_model

        # display settings
        self.silent = silent
        self.printing = printing
        self.tqdm_disable = tqdm_disable

        # results
        self.result_dict          = {}
        self.eigenvalues_dict     = {}
        self.optim_times_dict     = {}
        self.cut_times_dict       = {}
        self.feasible_values_dict = {}

    def analyse_dataset(self):
        '''Analyse given dataset using method settings and store results.'''

        # loop over gene queries of dataset
        for i in tqdm.tqdm(range(self.dataset.total_gene_queries), disable=self.tqdm_disable):

            try:

                # test feasibility of sample i
                solution, eigenvalues, optim_times, cut_times, feasible_values = self.feasibility_test(i)

                # store results
                self.result_dict[i]          = solution
                self.eigenvalues_dict[i]     = eigenvalues
                self.optim_times_dict[i]     = optim_times
                self.cut_times_dict[i]       = cut_times
                self.feasible_values_dict[i] = feasible_values

            # if exception
            except Exception as e:

                # display exception and traceback
                print(f"Optimization failed: {e}")
                traceback.print_exception(e)

                # store None as default result
                self.result_dict[i] = {
                    'status': None,
                    'time': None,
                    'cuts': None
                }
                self.eigenvalues_dict[i]     = None
                self.optim_times_dict[i]     = None
                self.cut_times_dict[i]       = None
                self.feasible_values_dict[i] = None

    def feasibility_test(self, i):
        '''
        Test feasibility of optimization model for gene query i

        Cutting Plane Algorithm:
        - Optimize NLP
            - Infeasible: stop
            - Feasible: check SDP feasibility
                - Feasible: stop
                - Infeasible: add cutting plane and return to NLP step

        Arguments:
            i: index of dataset gene query to test
            
        Returns:
            solution: dictionary of feasibility status, optimization time, cuts
            eigenvalues: moment matrix eigenvalues (*)
            optim_times: NLP optimization time (*)
            feasible_values: feasible value dictionary (*)
            (* list of values, one at each cut algorithm iteration)
        '''

        # store information from each cut algorithm iteration
        eigenvalues = []
        optim_times = []
        cut_times = []
        feasible_values = []

        # get moment bounds for gene query i
        OB_bounds = self.dataset.moment_bounds[:, i, :]

        # raise exception if moments not available
        if self.d_bd > self.dataset.d:
            raise Exception(f"Optimization uses bounds on d_bd = {self.d_bd} too high for dataset d = {self.dataset.d}")

        # if provided load WLS license credentials
        if self.license_file:
            environment_parameters = json.load(open(self.license_file))

        # otherwise use default environment (e.g Named User license)
        else:
            environment_parameters = {}
 
        # silence output
        if self.silent:
            environment_parameters['OutputFlag'] = 0

        # environment context
        with gp.Env(params=environment_parameters) as env:

            # model context
            with gp.Model('test-SDP', env=env) as model:

                # if provided: load model
                if self.load_model:

                    # get model
                    model = gp.read(self.load_model, env)

                    # get variables
                    Nd = utils.compute_Nd(self.S, self.d)
                    variables = {
                        'y': gp.MVar([model.getVarByName(f'y[{i}]') for i in range(Nd)]),
                        'k': gp.MVar([model.getVarByName(f'k[{i}]') for i in range(self.R)])
                    }

                    # construct moment matrix variables from variable y
                    if self.constraints.moment_matrices:
                        for s in range(self.S + 1):
                            M_s = optimization_utils.construct_M_s(variables['y'], s, self.S, self.d)
                            variables[f'M_{s}'] = M_s

                    # general setup
                    model.Params.TimeLimit = self.time_limit

                # otherwise: construct base model
                else:
                    model, variables = optimization_utils.base_model(self, model, OB_bounds)

                # additional constraints
                if self.custom_constraint:
                    model, variables = self.custom_constraint(self, model, variables)

                # set objective
                if self.objective_function:
                    obj = self.objective_function(self, model, variables)
                else:
                    obj = 0
                
                # check feasibility
                model, status, var_dict = optimization_utils.optimize(model, obj)

                # store solution information
                solution = {
                    'status': status,
                    'time': model.Runtime,
                    'cuts': 0
                }
                optim_times.append(solution['time'])
                feasible_values.append(var_dict)

                # no semidefinite constraints or non-optimal solution: return NLP status
                if not (self.constraints.moment_matrices and status == "OPTIMAL"):

                    # save final model
                    if self.save_model:
                        model.write(self.save_model)

                    return solution, eigenvalues, optim_times, cut_times, feasible_values

                # while below time and cut limit
                while (solution['cuts'] < self.cut_limit) and (solution['time'] < self.total_time_limit):

                    # check semidefinite feasibility & add cuts if needed
                    s = time.time()
                    model, semidefinite_feas, evals_data = optimization_utils.semidefinite_cut(self, model, variables)
                    t = time.time() - s

                    # store eigenvalue & time data
                    eigenvalues.append(evals_data)
                    cut_times.append(t)

                    # semidefinite feasible: return
                    if semidefinite_feas:

                        # save final model
                        if self.save_model:
                            model.write(self.save_model)

                        return solution, eigenvalues, optim_times, cut_times, feasible_values
                    
                    # record cut
                    solution['cuts'] += 1
                    
                    # semidefinite infeasible: check NLP feasibility with added cut
                    model, status, var_dict = optimization_utils.optimize(model, obj)

                    # update optimization time
                    solution['time'] += model.Runtime

                    # store feasible values & optim time
                    feasible_values.append(var_dict)
                    optim_times.append(model.Runtime)

                    # NLP + cut infeasible: return
                    # (also return for any other status, can only proceed if optimal as need feasible point)
                    if not (status == "OPTIMAL"):

                        # update solution
                        solution['status'] = status

                        # save final model
                        if self.save_model:
                            model.write(self.save_model)

                        return solution, eigenvalues, optim_times, cut_times, feasible_values

                # set custom status
                if solution['cuts'] >= self.cut_limit:

                    # exceeded number of cutting plane iterations
                    solution['status'] = "CUT_LIMIT"
                
                elif solution['time'] >= self.total_time_limit:

                    # exceeded total optimization time
                    solution['status'] = "TOTAL_TIME_LIMIT"

                # print
                if self.printing:
                    print(f"Optimization status: {solution['status']}")
                    print(f"Runtime: {solution['time']}")

                # save final model
                if self.save_model:
                    model.write(self.save_model)

                return solution, eigenvalues, optim_times, cut_times, feasible_values

    def compute_dataset_correlation(self, Sx=0, Sy=1):
        '''Compute dataset correlations from analysis results.'''

        # store
        correlation_list = []
        
        # loop over gene queries of dataset
        for i in tqdm.tqdm(range(self.dataset.total_gene_queries), disable=self.tqdm_disable):

            try:

                # compute final recovered correlation
                correlation = optimization_utils.compute_feasible_correlation(
                    self,
                    self.feasible_values_dict[i][-1],
                    Sx,
                    Sy
                )
                
                # store
                correlation_list.append(correlation)

            except Exception as e:

                # display exception and traceback
                print(f"Computation failed: {e}")
                traceback.print_exception(e)

                # store None as default result
                correlation_list.append(None)

        return correlation_list

    def compute_dataset_fano_factor(self, Sx=1):
        '''Compute dataset fano factor from analysis results.'''

        # store
        fano_list = []
        
        # loop over gene queries of dataset
        for i in tqdm.tqdm(range(self.dataset.total_gene_queries), disable=self.tqdm_disable):

            try:

                # compute final recovered correlation
                fano = optimization_utils.compute_feasible_fano_factor(
                    self,
                    self.feasible_values_dict[i][-1],
                    Sx
                )
                
                # store
                fano_list.append(fano)

            except Exception as e:

                # display exception and traceback
                print(f"Computation failed: {e}")
                traceback.print_exception(e)

                # store None as default result
                fano_list.append(None)

        return fano_list
    
# ------------------------------------------------
# Subclasses for common optimization choices
# ------------------------------------------------

# ------------------------------------------------
# Model-free Optimization
# ------------------------------------------------

class ModelFreeOptimization(Optimization):

    def __init__(self, dataset, **kwargs):

        # preset settings
        constraints = Constraint(
            moment_bounds   = True,
            moment_matrices = True,
            factorization   = True
        )
        reactions = []
        vrs = np.array([])
        db = 0
        R = 0
        S = 2
        U = []

        # initialize superclass
        super().__init__(
            dataset,
            constraints,
            reactions,
            vrs,
            db,
            R,
            S,
            U,
            **kwargs
        )

# ------------------------------------------------
# Birth-Death Optimization subclass
# ------------------------------------------------

class BirthDeathOptimization(Optimization):

    def __init__(self, dataset, **kwargs):

        # preset settings
        constraints = Constraint(
            moment_bounds=True,
            moment_matrices=True,
            moment_equations=True
        )
        reactions = [
            "1",
            "xs[0]",
            "1",
            "xs[1]",
            "xs[0] * xs[1]"
        ]
        vrs = np.array([
            [ 1,  0],
            [-1,  0],
            [ 0,  1],
            [ 0, -1],
            [-1, -1]
        ])
        db = 2
        R = 5
        S = 2
        U = []

        # default fixed rate
        if not ('rate_fixed' in kwargs.keys()):
            kwargs['rate_fixed'] = [(1, 1)]

        # initialize superclass
        super().__init__(
            dataset,
            constraints,
            reactions,
            vrs,
            db,
            R,
            S,
            U,
            **kwargs
        )

# ------------------------------------------------
# Telegraph Optimization subclass
# ------------------------------------------------

class TelegraphOptimization(Optimization):

    def __init__(self, dataset, **kwargs):

        # preset settings
        constraints = Constraint(
            moment_bounds=True,
            moment_matrices=True,
            moment_equations=True,
            unobserved_constraints=True
        )
        reactions = [
            "1 - xs[2]",
            "xs[2]",
            "xs[2]",
            "xs[0]",
            "1 - xs[3]",
            "xs[3]",
            "xs[3]",
            "xs[1]",
            "xs[0] * xs[1]"
        ]
        vrs = np.array([
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [0, 1, 0, 0],
            [0, -1, 0, 0],
            [-1, -1, 0, 0]
        ])
        db = 2
        R = 9
        S = 4
        U = [2, 3]

        # default fixed rate
        if not ('rate_fixed' in kwargs.keys()):
            kwargs['rate_fixed'] = [(3, 1)]

        # initialize superclass
        super().__init__(
            dataset,
            constraints,
            reactions,
            vrs,
            db,
            R,
            S,
            U,
            **kwargs
        )
