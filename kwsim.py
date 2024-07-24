import numpy as np
import cvxpy as cp
from scipy.stats import truncpareto
from typing import List, Tuple

class KWSimulator:
    def __init__(self, a: float = 0.02, b: float = 0.15, alp: float = 0.8, p: float = 0.4, lamb: float = 1):
        """
        Initialize the Order Flow Simulator.

        :param a: Lower bound of the Pareto distribution
        :type a: float
        :param b: Upper bound of the Pareto distribution
        :type b: float
        :param alp: Shape parameter of the Pareto distribution
        :type alp: float
        :param p: Resampling ratio
        :type p: float
        :param lamb: Lambda parameter for risk aversion
        :type lamb: float
        """
        self.a = a
        self.b = b
        self.alp = alp
        self.p = p
        self.lamb = lamb
        self.n = None  # Will be set in simulate method

    def compute_bootstrapped_alphas(self, r: np.ndarray, rho: float, D: np.ndarray) -> np.ndarray:
        """
        Compute bootstrapped alphas for time t.

        :param r: Returns vector
        :type r: np.ndarray
        :param rho: Correlation coefficient
        :type rho: float
        :param D: Covariance matrix
        :type D: np.ndarray
        :return: Bootstrapped alphas
        :rtype: np.ndarray
        """
        z = np.random.multivariate_normal(np.zeros(self.n), D)
        return rho * r + np.sqrt(1 - rho**2) * z

    def solve_mvo_problem(self, alpha_t: np.ndarray, Sigma: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Solve the Mean-Variance Optimization problem.

        :param alpha_t: Alpha vector at time t
        :type alpha_t: np.ndarray
        :param Sigma: Covariance matrix
        :type Sigma: np.ndarray
        :param h_prev: Previous holdings
        :type h_prev: np.ndarray
        :return: Optimal holdings
        :rtype: np.ndarray
        """
        h = cp.Variable(self.n)
        risk = cp.quad_form(h, Sigma)
        ret = alpha_t.T @ h
        objective = cp.Maximize(ret - 0.5 * self.lamb * risk)
        equity = np.sum(h_prev)  # Total equity (net asset value)
        constraints = [
            cp.sum(h) == equity,  # Maintain the same equity
            cp.norm(h, 1) <= 5 * equity,  # 5x leverage constraint
            cp.abs(h) <= 0.05 * cp.sum(cp.abs(h))  # No single position exceeds 5% of total exposure
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return h.value

    def draw_pareto_distributed(self, size: int = 1) -> np.ndarray:
        """
        Draw values from Pareto distribution.

        :param size: Number of samples to draw
        :type size: int
        :return: Drawn samples
        :rtype: np.ndarray
        """
        c = self.b / self.a
        return truncpareto.rvs(b=self.alp, c=c, scale=self.a, size=size)
    
    def step(self, h: np.ndarray, r: np.ndarray, Sigma: np.ndarray, rho: float, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a single step of the simulation.

        :param h: Current holdings
        :type h: np.ndarray
        :param r: Returns
        :type r: np.ndarray
        :param Sigma: Covariance matrix
        :type Sigma: np.ndarray
        :param rho: Correlation coefficient
        :type rho: float
        :param D: Diagonal matrix of conditional variances
        :type D: np.ndarray
        :return: Tuple containing order sizes and PoVs
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # 1. Determine order sizes
        alpha_t = self.compute_bootstrapped_alphas(r, rho, D)
        h_t = self.solve_mvo_problem(alpha_t, Sigma, h)
        dh_t = h_t - h

        # 2. Determine PoVs
        eta = self.draw_pareto_distributed(size=self.n)

        # 3. Resample PoVs
        y_t = np.sort(eta)
        J = np.random.choice(self.n, size=int(self.p * self.n), replace=False)
        y_t[J] = np.random.permutation(y_t[J])
        undo_permutation = np.argsort(np.argsort(eta))

        # 4. Output order flow
        return dh_t, y_t[undo_permutation]

    def simulate(self, w0: float, h0: np.ndarray, T: int, rs: List[np.ndarray], Sigmas: List[np.ndarray], rho: float, Ds: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Run the order flow simulation.

        :param w0: Initial wealth
        :type w0: float
        :param h0: Initial holdings
        :type h0: np.ndarray
        :param T: Number of time steps
        :type T: int
        :param rs: List of returns for each time step
        :type rs: List[np.ndarray]
        :param Sigmas: List of covariance matrices for each time step
        :type Sigmas: List[np.ndarray]
        :param rho: Correlation coefficient
        :type rho: float
        :param Ds: List of covariance matrices for bootstrapped alphas, each matrix should be a diagonal matrix of conditional variances
        :type Ds: List[np.ndarray]
        :return: List of (order sizes, PoVs) for each time step
        :rtype: List[Tuple[np.ndarray, np.ndarray]]
        :raises ValueError: If input lists don't match the specified number of time steps
        """
        if not all(len(x) == T for x in (rs, Sigmas, Ds)):
            raise ValueError("Length of rs, Sigmas, and Ds must match T")

        self.n = len(h0)
        h = h0
        orders = []
        for t in range(T):
            orders.append(self.step(h, rs[t], Sigmas[t], rho, Ds[t]))
            h += orders[-1][0]  # Update holdings
            h *= rs[t] 

        return orders