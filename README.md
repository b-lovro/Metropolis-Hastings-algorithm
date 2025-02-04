# Metropolis-Hastings Algorithm

This repository contains two implementations of the Metropolis-Hastings algorithm: one in Python and one in MATLAB.

## Available Files

- `Metropolis_Hastings_Python.py`: Python implementation of the Metropolis-Hastings algorithm.
- `Metropolis_Hastings_MATLAB.m`: MATLAB implementation of the Metropolis-Hastings algorithm.

## Metropolis-Hastings Algorithm Overview

The Metropolis-Hastings algorithm is a Markov chain Monte Carlo (MCMC) method for obtaining a sequence of random samples from a probability distribution for which direct sampling is difficult. This algorithm generates a Markov chain such that the states of the chain asymptotically follow the target distribution.

## Python Functions

### Metropolis_Hastings

**Purpose**: This function performs the Metropolis-Hastings algorithm to generate a Markov chain of samples from a specified distribution.

**Parameters**:
- `num_samples` (int): The number of samples to generate. Default is 1000.
- `num_lags` (int): The number of iterations between recorded samples (lags). Default is 10.
- `x` (float): The initial point for the Markov chain. Default is -1.
- `sigma` (float): The standard deviation of the normal distribution used for generating candidate points. Default is 1.
- `burn_in` (int): The number of initial iterations to discard (burn-in period). Default is 0.

**Returns**:
- `X` (numpy.ndarray): Array of generated samples from the Markov chain.
- `acceptance_rate` (numpy.ndarray): Array containing the number of accepted proposals and the total number of proposals.

### Metropolis_Hastings_Decision

**Purpose**: This function performs a single Metropolis-Hastings step to decide whether to accept or reject a candidate point.

**Parameters**:
- `x0` (float): The current point in the Markov chain.
- `sigma` (float): The standard deviation of the normal distribution used for generating candidate points.

**Returns**:
- `x1` (float): The next point in the Markov chain (either the candidate point if accepted, or the current point if rejected).
- `a` (int): Indicator of acceptance (1 if accepted, 0 if rejected).

## Explanation of the Decision Step

Pseudocode of the algorithm:
1. Generate a candidate, denoted as <code>x<sup>&Hat;</sup></code>. The value <code>x<sup>&Hat;</sup></code> is drawn from a proposal distribution of jumps, denoted as <code>q(x<sub>n;</sub>, x<sup>&Hat;</sup>)</code>, which depends on the current state of the Markov chain, <code>x<sub>n;</sub></code>.



2. The next step is called the acceptance-rejection step. The move is accepted with the probability:
    <br>
    <code>r(x<sub>n;</sub>, x<sup>&Hat;</sup>) = min{ π(x<sup>&Hat;</sup>) q(x<sup>&Hat;</sup>, x<sub>n;</sub>) /  π(x<sub>n;</sub>) q(x<sub>n;</sub>, x<sup>&Hat;</sup>) , 1 }</code>
    <br>
    <br>
    Now we need to decide whether to accept the candidate (in which case we take <code>x<sub>n+1;</sub> = x<sup>&Hat;</sup></code>) or reject it (in which case we take <code>x<sub>n+1;</sub> = x<sub>n;</sub></code>). We make this decision by selecting a random number from the interval <code>[0, 1]</code> and denote it by <code>u</code>. Then <code>x<sub>n+1;</sub></code> is chosen based on:
    <br>
    <br>
    <code>
    x<sub>n+1;</sub> =
    <br>
      { x<sup>&Hat;</sup> &nbsp;&nbsp; if &nbsp;&nbsp; u ≤ r(x<sub>n;</sub>, x<sup>&Hat;</sup>)<br>
        x<sub>n;</sub> &nbsp;&nbsp; if &nbsp;&nbsp; u > r(x<sub>n;</sub>, x<sup>&Hat;</sup>)
      }
    </code>

### Why Use a Burn-In Period and Lagged Samples

**Burn-In Period**:
- The burn-in period is used to allow the Markov chain to reach its equilibrium distribution. Initially, the chain starts from a potentially arbitrary state, and it may take some time for it to "forget" this initial state and start sampling from the target distribution. By discarding the initial set of samples (burn-in period), we can reduce the bias introduced by the initial state.

**Lagged Samples**:
- Lagging (or thinning) is used to reduce the autocorrelation between successive samples in the Markov chain. In a Markov chain, consecutive samples are often highly correlated, which can reduce the effective sample size and the efficiency of the sampling process. By taking samples only every few iterations (lagging), we can reduce this autocorrelation, leading to more independent samples and better estimates of the target distribution.


## Usage

Call `Metropolis_Hastings` with desired parameters to generate a Markov chain of samples. The generated samples and the acceptance rate can be analyzed or plotted using the provided plot function.

### Example

```python
## Using pyhton
X, acceptance_rate = Metropolis_Hastings(num_samples=10000, num_lags=100, sigma=1, burn_in=100)
Plot_Results(X, acceptance_rate,num_samples)
```
```python
## Using MATLAB
Metropolis_Hastings(1000,100,-1,1,10);
```


