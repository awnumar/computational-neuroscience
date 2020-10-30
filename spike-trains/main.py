import random as rnd
import math

Hz=1.0
sec=1.0
ms=0.001

def get_spike_train(rate,big_t,tau_ref):

    if 1<=rate*tau_ref:
        print("firing rate not possible given refractory period f/p")
        return []


    exp_rate=rate/(1-tau_ref*rate)

    spike_train=[]

    t=rnd.expovariate(exp_rate)

    while t< big_t:
        spike_train.append(t)
        t+=tau_ref+rnd.expovariate(exp_rate)

    return spike_train

def mean(x):
    return sum(x) / len(x)

def var(x):
    mu = mean(x)
    total = 0
    for v in x:
        total += (v - mu) * (v - mu)
    return total / len(x)

# q1 functions

def fano_factor(spike_train, big_t, interval):
    interval_s = 0.0
    interval_e = interval
    counts = []
    while interval_s < big_t:
        count = 0
        for spike in spike_train:
            if spike >= interval_s and spike < interval_e:
                count += 1
            if spike > interval_e:
                break
        counts.append(count)
        interval_s += interval
        interval_e += interval
    return var(counts) / mean(counts)

def coef_of_var(spike_train):
    intervals = []
    for i in range(1, len(spike_train)):
        intervals.append(spike_train[i] - spike_train[i-1])
    return math.sqrt(var(intervals)) / mean(intervals)

def question_1(rate, big_t, tau_ref):
    print(f"Setup: {rate} Hz; {big_t} s; {tau_ref} s")
    spike_train = get_spike_train(rate, big_t, tau_ref)
    print(f"Actual spike frequency: {len(spike_train)/big_t} Hz")
    intervals = [10 * ms, 50 * ms, 100 * ms]
    for interval in intervals:
        print(f"Fano factor ({interval} s): {fano_factor(spike_train, big_t, interval)}")
    print(f"Coefficient of variation: {coef_of_var(spike_train)}\n")

# q2 functions

def fano_factor_hits(samples, width, interval):
    interval_s = 0
    interval_e = interval
    counts = []
    while interval_s < (len(samples) * width):
        count = 0
        for i in range(len(samples)):
            if samples[i] == 1:
                if (i * width) >= interval_s and (i * width) < interval_e:
                    count += 1
                if (i * width) > interval_e:
                    break
        counts.append(count)
        interval_s += interval
        interval_e += interval
    return var(counts) / mean(counts)

def coef_of_var_hits(samples, width):
    intervals = []
    previous = -1
    for i in range(len(samples)):
        if samples[i] == 1:
            if previous == -1:
                previous = i
                continue
            intervals.append(width * (i - previous))
            previous = i
    return math.sqrt(var(intervals)) / mean(intervals)

print("Importing data from rho.dat...")
rho_samples = [int(line.strip()) for line in open("rho.dat", "r")]

def question_2():
    big_t = 1200 * sec
    width = 2 * ms
    print(f"Spike frequency: {len([1 for s in rho_samples if s == 1])/big_t} Hz")
    intervals = [10 * ms, 50 * ms, 100 * ms]
    for interval in intervals:
        print(f"Fano factor ({interval} s): {fano_factor_hits(rho_samples, width, interval)}")
    print(f"Coefficient of variation: {coef_of_var_hits(rho_samples, width)}\n")

# q3 functions

import matplotlib.pyplot as plt
import numpy as np

def question_3(samples, width):
    data = [0.0] * 101
    midpoint = int((len(data)-1)/2)
    N = 0
    for i in range(100, len(samples)-100):
        if samples[i] == 1:
            N += 1
            data[midpoint] += 1
            for j in range(1, 51):
                data[midpoint+j] += samples[i+j]
                data[midpoint-j] += samples[i-j]
    for i in range(len(data)):
        data[i] /= N
    x = [i for i in range(-100, 101, 2)]
    plt.plot(x, data)
    plt.ylabel("Average count")
    plt.xlabel("Relative position")
    plt.show()

# q4 functions

print("Importing data from stim.dat...")
stim_samples = [float(line.strip()) for line in open("stim.dat", "r")]

def question_4(stimulus, response):
    stim_space = [0.0] * 50
    spikes = [i for i in range(len(response)) if i >= 50 and response[i] == 1]
    for spike in spikes:
        for i in range(spike-1, spike-51, -1):
            stim_space[50+i-spike] += stimulus[i]
    for i in range(len(stim_space)):
        stim_space[i] /= len(spikes)
    x = [-i for i in range(100, 0, -2)]
    plt.plot(x, stim_space)
    plt.ylabel("Average stimulus")
    plt.xlabel("Time before spike (ms)")
    plt.show()

# Compute answers
question_1(35 * Hz, 1000 * sec, 5 * ms)
question_1(35 * Hz, 1000 * sec, 0 * ms)
question_2()
question_3(rho_samples, 2 * ms)
question_4(stim_samples, rho_samples)
