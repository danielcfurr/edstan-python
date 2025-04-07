functions {
  real pcm(int y, real theta, vector beta) {
    vector[rows(beta) + 1] unsummed;
    vector[rows(beta) + 1] probs;
    unsummed = append_row(rep_vector(0.0, 1), theta - beta);
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
  array[N] int<lower=0> y; // response for n; y = 0, 1 ... m_i
  int<lower=1> K; // # person covariates
  matrix[J, K] W; // person covariate matrix
}
transformed data {
  array[I] int m; // # parameters per item
  array[I] int pos; // first position in beta vector for item
  m = rep_array(0, I);
  for (n in 1 : N) {
    if (y[n] > m[ii[n]]) {
      m[ii[n]] = y[n];
    }
  }
  pos[1] = 1;
  for (i in 2 : I) {
    pos[i] = m[i - 1] + pos[i - 1];
  }
}
parameters {
  vector<lower=0>[I] alpha;
  sum_to_zero_vector[sum(m)] beta;
  vector[J] theta;
  vector[K] lambda;
}
model {
  alpha ~ lognormal(.5, 1);
  beta ~ normal(0, 3);
  theta ~ normal(W * lambda, 1);
  lambda ~ student_t(7, 0, 2.5);
  for (n in 1 : N) {
    target += pcm(y[n], theta[jj[n]] .* alpha[ii[n]],
                  segment(beta, pos[ii[n]], m[ii[n]]));
  }
}
