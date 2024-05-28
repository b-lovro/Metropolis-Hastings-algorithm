import numpy as np
import matplotlib.pyplot as plt

def Metropolis_Hastings(num_samples=1000, num_lags=10, x=-1, sigma=1, burn_in=0):
    X = np.zeros(num_samples)  # samples of the Markov chain
    acceptance_rate = np.array([0, 0])  # vector for recording the acceptance rate

    # MH algorithm
    # Burn-in phase
    for i in range(burn_in):
        x, a = Metropolis_Hastings_Decision(x, sigma)
        acceptance_rate += np.array([a, 1])  # recording the acceptance rate

    # Sampling phase
    for i in range(num_samples):
        for j in range(num_lags):
            x, a = Metropolis_Hastings_Decision(x, sigma)
            acceptance_rate += np.array([a, 1])  # recording the acceptance rate
        X[i] = x  # save the i-th sample

    return X, acceptance_rate

def Metropolis_Hastings_Decision(x0, sigma):
    xp = np.random.normal(x0, sigma)  # generate a candidate from the normal distribution
    acceptance_probability = min(Distribution(xp) / Distribution(x0), 1)  # acceptance probability
    u = np.random.rand()  # random number from the interval [0,1]

    # check the acceptance criterion
    if u <= acceptance_probability:
        x1 = xp  # accept the candidate
        a = 1  # record the acceptance
    else:
        x1 = x0  # reject the candidate and keep the same point
        a = 0  # record the rejection

    return x1, a

def Distribution(x):
    # our distribution
    return np.exp(-x**2) * (2 + np.sin(x * 5) + np.sin(x * 2))

def Plot_Results(X, acceptance_rate, num_samples):
    # Plot the samples of the Markov chain
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(X)
    plt.title('Samples of the Markov Chain')
    plt.xlabel('Iteration')
    plt.ylabel('Sample Value')

    # Plot the histogram of the samples and overlay the original distribution
    plt.subplot(2, 1, 2)
    plt.hist(X, bins=30, density=True, alpha=0.6, color='g')

    # Define a range for plotting the original distribution
    x_values = np.linspace(min(X), max(X), 1000)
    y_values = Distribution(x_values)
    
    # Normalize the original distribution to match the scale of the histogram
    y_values /= np.trapz(y_values, x_values)
    
    plt.plot(x_values, y_values, 'r-', linewidth=2)
    plt.title('Histogram of Samples with Original Distribution')
    plt.xlabel('Sample Value')
    plt.ylabel('Density')
    plt.legend(['Original Distribution', 'Samples'])
    
    plt.tight_layout()
    plt.show()

    # Calculate and display the acceptance rate
    acceptance_rate = acceptance_rate[0] / acceptance_rate[1]
    print(f'Acceptance Rate: {acceptance_rate * 100:.2f}%')

# Run the algorithm
num_samples=1000
num_lags=100
sigma=1
burn_in=100

X, acceptance_rate = Metropolis_Hastings(num_samples, num_lags, sigma, burn_in)
Plot_Results(X, acceptance_rate,num_samples)
