functions {
  real rsm(int y, real theta, real beta, vector kappa) {
    vector[rows(kappa) + 1] unsummed;
    vector[rows(kappa) + 1] probs;
    unsummed = append_row(rep_vector(0, 1), theta - beta - kappa);
    probs = softmax(cumulative_sum(unsummed));
    return categorical_lpmf(y + 1 | probs);
  }
}
data {
  int<lower=1> I; // # items
  int<lower=1> J; // # persons
  int<lower=1> N; // # responses
  array[N] int<lower=1, upper=I> ii; // i for n
  array[N] int<lower=1, upper=J> jj; // j for n
  array[N] int<lower=0> y; // response for n; y in {0 ... m_i}
  int<lower=1> K; // # person covariates
  matrix[J, K] W; // person covariate matrix
}
transformed data {
  int m = max(y); // # steps
}
parameters {
  vector<lower=0>[I] alpha;
  sum_to_zero_vector[I] beta;
  sum_to_zero_vector[m] kappa;
  vector[J] theta;
  vector[K] lambda;
}
model {
  alpha ~ lognormal(.5, 1);
  beta ~ normal(0, 3);
  kappa ~ normal(0, 3);
  theta ~ normal(W * lambda, 1);
  lambda ~ student_t(7, 0, 2.5);
  for (n in 1 : N) {
    target += rsm(y[n], theta[jj[n]] .* alpha[ii[n]], beta[ii[n]], kappa);
  }
}
