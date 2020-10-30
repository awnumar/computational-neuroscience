#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# units

M = 1000000.0
m = 0.001
n = 0.000000001

s = 1.0
V = 1.0
A = 1.0
S = 1.0
Hz = 1.0
Ohm = 1.0

ms = m * s
mV = m * V
MOhm = M * Ohm
nA = n * A
nS = n * S

###

# Integrate and fire model differential equation
def p1q1():
    def dV_by_dt(V):
        v = (-70 * mV) - V + ((10 * MOhm) * (3.1 * nA))
        return v / (10 * ms)

    def euler(f, y, t):
        z = y + (t * f(y))
        if z >= (-40 * mV):
            z = -70 * mV
        return z

    time = [0]
    voltages = [-70 * ms]

    for i in range(int(1/(0.25 * ms))):
        time.append(i * (0.25 * ms))
        voltages.append(euler(dV_by_dt, voltages[-1], 0.25 * ms))

    plt.plot(time, voltages)
    plt.title("Leaky Integrate and Fire Model (1s)")
    plt.xlabel("time (s)")
    plt.ylabel("voltage (V)")
    plt.show()

# execute part 1 question 1
p1q1()

# part 1 question 2

def p1q2(Es):
    def dV_by_dt(N, V, RmIs):
        v = N["El"] - V + N["RmIe"] + RmIs
        return v / N["Tm"]

    def ds_by_dt(S, s):
        return -s / S["Ts"]

    def euler_s(S, y, f, t):
        return y + (t * f(S, y))

    def euler_V(N, RmIs, y, f, t):
        return y + (t * f(N, y, RmIs))

    # Setup neuron parameters
    N = {
        "Tm": 20 * ms,
        "El": -70 * mV,
        "Vrst": -80 * mV,
        "Vth": -54 * mV,
        "RmIe": 18 * mV
    }

    # Initialise voltages
    import random
    N1 = [random.uniform(N["Vrst"], N["Vth"])]
    N2 = [random.uniform(N["Vrst"], N["Vth"])]
    time = [0]

    # Setup synapse parameters
    S = {
        "RmGb": 0.15,
        "Ds": 0.5,
        "Ts": 10 * ms,
        "Es": Es
    }

    # Initialise synapse values
    S1 = [0 * mV]
    S2 = [0 * mV]
    # N1 feeds into S2 which affects N2
    # N2 feeds into S1 which affects N1

    dt = 0.25 * ms
    for i in range(int(1/dt)):
        # Update time step
        time.append(i * dt)

        # Update S2 using N1
        if N1[-1] == N["Vrst"]:
            S2.append(S["Ds"] + euler_s(S, S2[-1], ds_by_dt, dt))
        else:
            S2.append(euler_s(S, S2[-1], ds_by_dt, dt))
        # Update S1 using N2
        if N2[-1] == N["Vrst"]:
            S1.append(S["Ds"] + euler_s(S, S1[-1], ds_by_dt, dt))
        else:
            S1.append(euler_s(S, S1[-1], ds_by_dt, dt))
        
        # Compute change in N1 from S1
        N1_RmIs = S1[-1] * S["RmGb"] * (S["Es"] - N1[-1])
        # Compute change in N2 from S2
        N2_RmIs = S2[-1] * S["RmGb"] * (S["Es"] - N2[-1])

        # Compute new voltage values
        N1.append(euler_V(N, N1_RmIs, N1[-1], dV_by_dt, dt))
        N2.append(euler_V(N, N2_RmIs, N2[-1], dV_by_dt, dt))

        # Check for spike condition
        if N1[-1] >= N["Vth"]: N1[-1] = N["Vrst"]
        if N2[-1] >= N["Vth"]: N2[-1] = N["Vrst"]
    
    plt.plot(time, N1, label="Neuron 1")
    plt.plot(time, N2, label="Neuron 2")
    plt.title(f"Two neurones connected with synapses (1s) (E_s = {Es} mV)")
    plt.xlabel("time (s)")
    plt.ylabel("voltage (V)")
    plt.show()

# Execute part A question 2
p1q2(0)
p1q2(-80 * mV)

# part 2 question 1

def dV_by_dt(N, V, sV):
    return (N["El"] - V + (N["Rm"] * N["Ie"]) + sV) / N["Tm"]

def ds_by_dt(S):
    return -S["s"]/S["Ts"]

def euler_V(N, y, f, t, sV):
    v_ = y + (t * f(N, y, sV))
    if v_ >= N["Vth"]:
        v_ = N["Vrst"]
    return v_

def euler_s(S, f, t):
    return S["s"] + (t * f(S))

def p2q1(firingRate):
    # Setup neuron parameters
    N = {
        "Tm": 10 * ms,
        "El": -65 * mV,
        "Vrst": -65 * mV,
        "Vth": -50 * mV,
        "Rm": 100 * MOhm,
        "Ie": 0 # so RmIe = 0
    }

    # Create all the input synapses
    synapses = []
    for _ in range(40):
        synapses.append({
            "Ts": 2 * ms,
            "Es": 0 * mV,
            "Ds": 0.5,
            "gbar": 4 * nS,
            "s": 0
        })

    # returns array of weights
    def gbars():
        return [s["gbar"] for s in synapses]

    # returns array of s values
    def ss():
        return [s["s"] for s in synapses]

    # Store the state at each timestep here
    voltage = [N["Vrst"]]
    time = [0]

    # Model the system at discrete time steps
    dt = 0.25 * ms
    for i in range(int(1/dt)):
        # Append this current time to state log
        time.append(i * dt)

        # Compute collective synaptic voltage
        sV = N["Rm"] * np.dot(gbars(), ss()) * (synapses[0]["Es"] - voltage[-1])

        # Compute and log the neuron voltage
        voltage.append(euler_V(N, voltage[-1], dV_by_dt, dt, sV))

        # Update synapses
        for s in range(len(synapses)):
            if np.random.uniform() < firingRate * dt:
                # Spike
                synapses[s]["s"] = synapses[s]["Ds"] + euler_s(synapses[s], ds_by_dt, dt)
            else:
                # No spike
                synapses[s]["s"] = euler_s(synapses[s], ds_by_dt, dt)

    plt.plot(time, voltage)
    plt.title(f"Neuron voltage with 40 input synapses firing at {firingRate} Hz")
    plt.xlabel("time (s)")
    plt.ylabel("voltage (V)")
    plt.show()

# execute part 2 question 1
p2q1(15 * Hz)

# part 2 question 2

import math

def mean(s):
    return sum(s)/len(s)

# STDP config
A_plus = 0.2 * nS
A_minus = 0.25 * nS
tau_plus = 20 * ms
tau_minus = 20 * ms

def F(delta_time):
    if delta_time > 0:
        return A_plus * math.exp(-abs(delta_time)/tau_plus)
    else:
        return -A_minus * math.exp(-abs(delta_time)/tau_minus)

def next_synapse_strength(S, time_post):
    delta_time = time_post - S["last_spike"]
    s = S["gbar"] + F(delta_time)
    if s < 0:
        s = 0
    if s > 4 * nS:
        s = 4 * nS
    return s

def p2q2(stpd, initialStrength, firingRate, T):
    # Setup neuron parameters
    N = {
        "Tm": 10 * ms,
        "El": -65 * mV,
        "Vrst": -65 * mV,
        "Vth": -50 * mV,
        "Rm": 100 * MOhm,
        "Ie": 0, # so RmIe = 0
        "last_spike": 0 # time of last spike in this neurone
    }

    # Create all the input synapses
    synapses = []
    for _ in range(40):
        synapses.append({
            "Ts": 2 * ms,
            "Es": 0 * mV,
            "Ds": 0.5,
            "gbar": initialStrength,
            "s": 0,
            "last_spike": 0 # time of last spike in presynaptic neurone
        })

    # returns array of weights
    def gbars():
        return [s["gbar"] for s in synapses]

    # returns array of s values
    def ss():
        return [s["s"] for s in synapses]

    # Store the state at each timestep here
    voltage = [N["Vrst"]]
    time = [0]

    # Model the system at discrete time steps
    dt = 0.25 * ms
    for i in range(int(T/dt)):
        # Append this current time to state log
        time.append(i * dt)

        # Compute collective synaptic voltage
        sV = N["Rm"] * np.dot(gbars(), ss()) * (synapses[0]["Es"] - voltage[-1])

        # Compute and log the neuron voltage
        voltage.append(euler_V(N, voltage[-1], dV_by_dt, dt, sV))

        # handle weight changes
        if stpd and voltage[-1] == N["Vrst"]:
            N["last_spike"] = time[-1]
            for k in range(len(synapses)):
                synapses[k]["gbar"] = next_synapse_strength(synapses[k], N["last_spike"])

        # Update synapses
        for k in range(len(synapses)):
            if np.random.uniform() < firingRate * dt:
                # Spike
                synapses[k]["last_spike"] = time[-1]
                synapses[k]["s"] = synapses[k]["Ds"] + euler_s(synapses[k], ds_by_dt, dt)
                if stpd:
                    synapses[k]["gbar"] = next_synapse_strength(synapses[k], N["last_spike"])
            else:
                # No spike
                synapses[k]["s"] = euler_s(synapses[k], ds_by_dt, dt)

    # plot histogram
    print(f"mean end strength: {mean(gbars())}")
    plt.hist(gbars())
    plt.title(f"Histogram of synaptic weights at the end of 300s simulation (stpd={stpd})")
    plt.xlabel("strength (S)")
    plt.ylabel("frequency density")
    plt.show()

    # get average firing rates
    _t = []
    _f = []
    for _i in range(int(T/(10*s))):
        # get current time
        _ctime = _i * 10 * s
        _t.append(_ctime)

        # compute mean frequency
        _f.append(len([1 for k in voltage[int(_ctime/dt):int((_ctime+(10*s))/dt)] if k == N["Vrst"]])/(10*s))

    # plot average firing rate of N as a function of time
    plt.plot(_t, _f)
    plt.title(f"Average firing rate of post-synaptic neuron (stpd={stpd})")
    plt.xlabel("time (s)")
    plt.ylabel("average frequency (Hz)")
    plt.show()

    # Average firing rate in last 30s of simulation
    afq = len([1 for k in voltage[int((T-30.0)/dt):] if k == N["Vrst"]])/(30*s)
    print(f"average firing rate in last 30s is {afq} Hz")

# execute part b question 2
p2q2(True, 4 * nS, 15 * Hz, 300 * s)

def p2q3_simulate(stpd, firingRate, T):
    # Setup neuron parameters
    N = {
        "Tm": 10 * ms,
        "El": -65 * mV,
        "Vrst": -65 * mV,
        "Vth": -50 * mV,
        "Rm": 100 * MOhm,
        "Ie": 0, # so RmIe = 0
        "last_spike": 0 # time of last spike in this neurone
    }

    # Create all the input synapses
    synapses = []
    for _ in range(40):
        synapses.append({
            "Ts": 2 * ms,
            "Es": 0 * mV,
            "Ds": 0.5,
            "gbar": 4 * nS,
            "s": 0,
            "last_spike": 0 # time of last spike in presynaptic neurone
        })

    # returns array of weights
    def gbars():
        return [s["gbar"] for s in synapses]

    # returns array of s values
    def ss():
        return [s["s"] for s in synapses]

    # Store the state at each timestep here
    voltage = [N["Vrst"]]
    time = [0]

    # Model the system at discrete time steps
    dt = 0.25 * ms
    for i in range(int(T/dt)):
        # Append this current time to state log
        time.append(i * dt)

        # Compute collective synaptic voltage
        sV = N["Rm"] * np.dot(gbars(), ss()) * (synapses[0]["Es"] - voltage[-1])

        # Compute and log the neuron voltage
        voltage.append(euler_V(N, voltage[-1], dV_by_dt, dt, sV))

        # handle weight changes
        if stpd and voltage[-1] == N["Vrst"]:
            N["last_spike"] = time[-1]
            for k in range(len(synapses)):
                synapses[k]["gbar"] = next_synapse_strength(synapses[k], N["last_spike"])

        # Update synapses
        for k in range(len(synapses)):
            if np.random.uniform() < firingRate * dt:
                # Spike
                synapses[k]["last_spike"] = time[-1]
                synapses[k]["s"] = synapses[k]["Ds"] + euler_s(synapses[k], ds_by_dt, dt)
                if stpd:
                    synapses[k]["gbar"] = next_synapse_strength(synapses[k], N["last_spike"])
            else:
                # No spike
                synapses[k]["s"] = euler_s(synapses[k], ds_by_dt, dt)


    # Average firing rate in last 30s of simulation
    afr = len([1 for k in voltage[int((T-30.0)/dt):] if k == N["Vrst"]])/(30*s)

    # final strengths
    final_g_ = gbars()

    return afr, final_g_

def p2q3_output_hz_vs_input_hz():
    # initialise variables to store results
    inputFrequencies = []
    outputFrequencies_adaptive = []

    # for each 1 Hz increment in the input firing rate
    for i in range(10, 21):
        print(f"Simulating {i * Hz} Hz input firing frequency (stpd=on)")
        inputFrequencies.append(i * Hz)
        afr, fg = p2q3_simulate(True, inputFrequencies[-1], 300)
        outputFrequencies_adaptive.append(afr)
    
    # do the same for stpd off
    outputFrequencies = []
    for f in inputFrequencies:
        print(f"Simulating {f} Hz input firing frequency (stpd=off)")
        afr, fg = p2q3_simulate(False, f, 300)
        outputFrequencies.append(afr)
    
    # plot both
    plt.plot(inputFrequencies, outputFrequencies_adaptive, label="STPD=on")
    plt.plot(inputFrequencies, outputFrequencies, label="STPD=off")
    plt.title(f"Steady state firing frequency as a function of input frequency")
    plt.xlabel("input frequency (Hz)")
    plt.ylabel("output frequency (Hz)")
    plt.legend()
    plt.show()

def p2q3_steady_strength():
    print("simulate stpd=on, 10 Hz for 300s")
    afr, fg10 = p2q3_simulate(True, 10 * Hz, 300)
    print(f"mean end strength for 10 Hz input: {mean(fg10)}")

    print("simulate stpd=on, 20 Hz for 300s")
    afr, fg20 = p2q3_simulate(True, 20 * Hz, 300)
    print(f"mean end strength for 20 Hz input: {mean(fg20)}")

    # plot 10 Hz case
    plt.hist(fg10)
    plt.title("Histogram of synaptic weights at the end of 300s simulation (stpd=on, input=10Hz)")
    plt.xlabel("strength (S)")
    plt.ylabel("frequency density")
    plt.show()

    # plot 20 Hz case
    plt.hist(fg20)
    plt.title("Histogram of synaptic weights at the end of 300s simulation (stpd=on, input=20Hz)")
    plt.xlabel("strength (S)")
    plt.ylabel("frequency density")
    plt.show()

# execute p2 q3
p2q3_output_hz_vs_input_hz()
p2q3_steady_strength()

def standard_deviation(v):
    m = mean(v)
    var = 0
    for e in v:
        var += (e - m)*(e - m)
    return math.sqrt(var / len(v))

def firingRate(t, average, B, f):
    return average + (B * math.sin(2 * math.pi * f * t))

def p2q4_simulate(B, T):
    # Setup neuron parameters
    N = {
        "Tm": 10 * ms,
        "El": -65 * mV,
        "Vrst": -65 * mV,
        "Vth": -50 * mV,
        "Rm": 100 * MOhm,
        "Ie": 0, # so RmIe = 0
        "last_spike": 0 # time of last spike in this neurone
    }

    # Create all the input synapses
    synapses = []
    for _ in range(40):
        synapses.append({
            "Ts": 2 * ms,
            "Es": 0 * mV,
            "Ds": 0.5,
            "gbar": 4 * nS,
            "s": 0,
            "last_spike": 0 # time of last spike in presynaptic neurone
        })

    # returns array of weights
    def gbars():
        return [s["gbar"] for s in synapses]

    # returns array of s values
    def ss():
        return [s["s"] for s in synapses]

    # Store the state at each timestep here
    voltage = [N["Vrst"]]
    time = [0]

    # Model the system at discrete time steps
    dt = 0.25 * ms
    for i in range(int(T/dt)):
        # Append this current time to state log
        time.append(i * dt)

        # Compute collective synaptic voltage
        sV = N["Rm"] * np.dot(gbars(), ss()) * (synapses[0]["Es"] - voltage[-1])

        # Compute and log the neuron voltage
        voltage.append(euler_V(N, voltage[-1], dV_by_dt, dt, sV))

        # handle weight changes
        if voltage[-1] == N["Vrst"]:
            N["last_spike"] = time[-1]
            for k in range(len(synapses)):
                synapses[k]["gbar"] = next_synapse_strength(synapses[k], N["last_spike"])

        # Update synapses
        for k in range(len(synapses)):
            if np.random.uniform() < firingRate(time[-1], 20 * Hz, B, 10 * Hz) * dt:
                # Spike
                synapses[k]["last_spike"] = time[-1]
                synapses[k]["s"] = synapses[k]["Ds"] + euler_s(synapses[k], ds_by_dt, dt)
                synapses[k]["gbar"] = next_synapse_strength(synapses[k], N["last_spike"])
            else:
                # No spike
                synapses[k]["s"] = euler_s(synapses[k], ds_by_dt, dt)

    # return the final synaptic strengths
    return gbars()

def p2q4():
    # input conditions
    Bs = [0 * Hz, 5 * Hz, 10 * Hz, 15 * Hz, 20 * Hz]

    # output states
    strengths           = []
    means               = []
    standard_deviations = []

    # run simulation
    for B in Bs:
        print(f"Running correlated simulation with B = {B} Hz")
        strengths.append(p2q4_simulate(B, 300 * s))

    # compute means and standard deviations
    for sd in strengths:
        means.append(mean(sd))
        standard_deviations.append(standard_deviation(sd))
    
    # plot means and standard deviations as function of B
    plt.plot(Bs, means, label="mean")
    plt.plot(Bs, standard_deviations, label="standard deviation")
    plt.title(f"Final synaptic strengths as function of correlation constant B")
    plt.xlabel("correlation constant B (Hz)")
    plt.ylabel("synaptic strength (S)")
    plt.legend()
    plt.show()

    # plot histograms of marginals
    plt.hist(strengths[0])
    plt.title(f"Histogram of synaptic weights at the end of 300s simulation (B = {Bs[0]})")
    plt.xlabel("strength (S)")
    plt.ylabel("frequency density")
    plt.show()

    plt.hist(strengths[4])
    plt.title(f"Histogram of synaptic weights at the end of 300s simulation (B = {Bs[4]})")
    plt.xlabel("strength (S)")
    plt.ylabel("frequency density")
    plt.show()

# execute part b q 4
p2q4()
