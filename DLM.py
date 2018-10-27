import numpy as np
from numpy.linalg import solve
from scipy.stats import invwishart, multivariate_normal

class DLM():
    y: np.ndarray
    n: np.ndarray
    df_w: int
    scale_w: np.ndarray
    m_0: np.ndarray
    c_0: np.ndarray

    def __init__(self, y, n, df_w=None, scale_w=None, m_0=None, c_0=None):
        # Extract dimensions for F, G
        self.T = y.shape[0]
        self.J = y.shape[1]
        self.num_random_thetas = 9
        self.P = self.J - 1 + self.num_random_thetas

        # Save y and n as class members, adding empty columns so indexes align with thetas
        extra_row = np.array([np.nan] * self.J).reshape([1, self.J])
        self.y = np.r_[extra_row, y.astype(np.float64)]
        self.n = np.r_[extra_row, n.astype(np.float64)]

        # Save priors
        if m_0 is None:
            m_0 = np.zeros([self.P])
            m_0[0] = 0.5
            self.m_0 = m_0
        else:
            self.m_0 = m_0
        self.c_0 = np.eye(self.P) * (1e-1 ** 2) if c_0 is None else c_0
        self.df_w = 2000 if df_w is None else df_w
        self.scale_w = np.eye(self.num_random_thetas) if scale_w is None else scale_w

        # Create process matrix G
        self.G = np.eye(self.P)
        self.G[0, 1] = 1.

        # Create observation matrix F
        # NOTE: F is modified slightly depending on the day of the week
        self.F = np.zeros([self.J, self.P])
        self.F[:, [0, 2]] = 1
        for i in range(1, self.J):
            self.F[i, self.P - self.J + i] = 1

        # Distributions for sampling
        self.iw = invwishart
        self.mvn = multivariate_normal


    def fit(self, n_samples=100, print_every=20):
        # Initialize variables
        self.thetas = np.zeros([self.P, self.T+1, n_samples+2])
        self.a = np.zeros([self.P, self.T+1])
        self.R = np.zeros([self.P, self.P, self.T+1])
        self.C = np.zeros([self.P, self.P, self.T+1])
        self.m = np.zeros([self.P, self.T+1])
        self.W = np.zeros([self.num_random_thetas, self.num_random_thetas, n_samples+2])
        self.W_i = np.zeros([self.P, self.P])
        self.probs = np.zeros([self.J, self.T+1, n_samples+2])

        # Fill with initial values
        for t in range(self.T+1):
            self.thetas[:, t, 0] = self.m_0
        self.m[:, 0] = self.m_0
        self.C[:, :, 0] = self.c_0
        self.W[:, :, 0] = self.scale_w / (self.df_w + self.P + 1)

        for i in range(1, n_samples+2):
            if i % print_every == 0:
                print("Sample {}".format(i))

            self.W_i[:9, :9] = self.W[:, :, i-1]
            self.filter(i, self.W_i)
            if i >= n_samples + 1:
                break
            self.backward_sample(i)
            self.sample_W(i)

        del self.a, self.R, self.C, self.m, self.W_i
        self.thetas = self.thetas[:, :, 1:(n_samples+1)]
        self.W = self.W[:, :, 1:(n_samples+1)]
        self.probs = self.probs[:, 1:, 1:(n_samples+1)]


    def filter(self, i, W):

        for t in range(1, self.T+1):
            # One step ahead predictive distribution of theta
            a = self.G @ self.m[:, t-1]
            self.a[:, t] = a
            R = self.G @ self.C[:, :, t-1] @ self.G.transpose() + W
            self.R[:, :, t] = R
            #print(a, end="\n\n")
            #print(R, end="\n\n")

            F = np.copy(self.F)
            F[:, (t-1) % 7 + 2] = 1  # for day-of-the-week effect
            p_t = F @ self.thetas[:, t, i-1]
            self.probs[:, t, i-1] = p_t

            y_t = self.y[t, :]
            n_t = self.n[t, :]
            missing = np.isnan(y_t) | np.isnan(n_t)

            if all(missing):
                self.m[:, t] = a
                self.C[:, :, t] = R

            else:
                if any(missing):
                    not_missing = np.bitwise_not(missing)
                    y_t = y_t[not_missing]
                    n_t = n_t[not_missing]
                    p_t = p_t[not_missing]
                    F = F[not_missing, :]
                    #print(y_t)
                    #print(n_t)
                    #print(F)

                # One step ahead predictive distribution of y
                f = F @ a
                FR = F @ R
                V = self.create_V(p_t, n_t)
                Q = FR @ F.transpose() + V
                #print(Q, end="\n\n")

                # Filtering distribution of theta given y
                FR_transpose = FR.transpose()
                self.m[:, t] = a + FR_transpose @ solve(Q, y_t - f)
                self.C[:, :, t] = R - FR_transpose @ solve(Q, FR)
                #print(self.m[:, t])
                #print(self.C[:, :, t])


    def backward_sample(self, i):
        self.thetas[:, self.T, i] = self.mvn.rvs(self.m[:, self.T], self.C[:, :, self.T])
        for t in range(self.T-1, -1, -1):
            C = self.C[:, :, t]
            R = self.R[:, :, t+1]
            e = self.thetas[:, t+1, i] - self.a[:, t+1]
            h = self.m[:, t] + C @ self.G.transpose() @ solve(R, e)
            H = C - C @ self.G.transpose() @ solve(R, self.G @ C)
            self.thetas[:, t, i] = self.mvn.rvs(h, H)


    def create_V(self, p, n):
        assert len(p) == len(n)
        vars = [p[i] * (1 - p[i]) / n[i] for i in range(len(p))]
        return np.diag(vars)


    def sample_W(self, i):
        df_new = self.df_w + self.num_random_thetas * self.T / 2
        scale_new = np.copy(self.scale_w)
        for t in range(1, self.T+1):
            d = self.thetas[:self.num_random_thetas, t, i] - \
                self.G[:self.num_random_thetas, self.num_random_thetas] @ self.thetas[:self.num_random_thetas, t-1, i]
            scale_new += np.outer(d, d)
        self.W[:, :, i] = self.iw.rvs(df_new, scale_new)


