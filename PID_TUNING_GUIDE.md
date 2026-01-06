# PID Tuning Guide for gym-pybullet-drones

This guide explains where and how to tune PID values when running `pid.py`.

---

## PID Coefficients Location

The PID coefficients are defined in **`DSLPIDControl.py`** at lines **37-42**:

```python
self.P_COEFF_FOR = np.array([.4, .4, 1.25])        # Position control P gains
self.I_COEFF_FOR = np.array([.05, .05, .05])      # Position control I gains
self.D_COEFF_FOR = np.array([.2, .2, .5])         # Position control D gains
self.P_COEFF_TOR = np.array([70000., 70000., 60000.])  # Attitude control P gains
self.I_COEFF_TOR = np.array([.0, .0, 500.])       # Attitude control I gains
self.D_COEFF_TOR = np.array([20000., 20000., 12000.]) # Attitude control D gains
```

---

## Two Ways to Tune PID Values

### Method 1: Modify in `pid.py` (Recommended)

**Best for**: Quick tuning, experimenting, per-script customization

I've added commented code in `pid.py` that you can uncomment and modify:

```python
#### Initialize the controllers ############################
if drone in [DroneModel.CF2X, DroneModel.CF2P]:
    ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
    #### Tune PID coefficients (optional) ####################
    for controller in ctrl:
        controller.setPIDCoefficients(
            p_coeff_pos=np.array([.4, .4, 1.25]),      # Position P gains [x, y, z]
            i_coeff_pos=np.array([.05, .05, .05]),   # Position I gains [x, y, z]
            d_coeff_pos=np.array([.2, .2, .5]),      # Position D gains [x, y, z]
            p_coeff_att=np.array([70000., 70000., 60000.]),  # Attitude P gains [roll, pitch, yaw]
            i_coeff_att=np.array([.0, .0, 500.]),    # Attitude I gains [roll, pitch, yaw]
            d_coeff_att=np.array([20000., 20000., 12000.])   # Attitude D gains [roll, pitch, yaw]
        )
```

**Advantages**:
- ✅ Doesn't modify source code
- ✅ Easy to experiment with different values
- ✅ Can have different values per script
- ✅ Easy to revert changes

**Example**: Tune only Z-axis position control
```python
for controller in ctrl:
    controller.setPIDCoefficients(
        p_coeff_pos=np.array([.4, .4, 2.0]),  # Increased Z P gain for faster altitude response
    )
```

---

### Method 2: Modify `DSLPIDControl.py` Directly

**Best for**: Permanent changes, default values for all scripts

Edit **`gym_pybullet_drones/control/DSLPIDControl.py`** at lines **37-42**:

```python
# Original values
self.P_COEFF_FOR = np.array([.4, .4, 1.25])
self.I_COEFF_FOR = np.array([.05, .05, .05])
self.D_COEFF_FOR = np.array([.2, .2, .5])
self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
self.I_COEFF_TOR = np.array([.0, .0, 500.])
self.D_COEFF_TOR = np.array([20000., 20000., 12000.])

# Your tuned values
self.P_COEFF_FOR = np.array([.5, .5, 1.5])  # Increased gains
# ... etc
```

**Advantages**:
- ✅ Changes apply to all scripts using DSLPIDControl
- ✅ No need to modify each script

**Disadvantages**:
- ❌ Modifies source code
- ❌ Harder to revert
- ❌ Affects all scripts

---

## Understanding the PID Coefficients

### Position Control Gains (FOR = Force)

**`P_COEFF_FOR`** - Proportional gains `[x, y, z]`
- **What it does**: Responds to position error
- **Effect**: Higher = faster response, but can cause overshoot
- **Typical values**: `[0.4, 0.4, 1.25]`
- **Note**: Z-axis (altitude) typically needs higher gain

**`I_COEFF_FOR`** - Integral gains `[x, y, z]`
- **What it does**: Eliminates steady-state error
- **Effect**: Higher = faster error correction, but can cause oscillation
- **Typical values**: `[0.05, 0.05, 0.05]`
- **Warning**: Too high can cause instability

**`D_COEFF_FOR`** - Derivative gains `[x, y, z]`
- **What it does**: Dampens oscillations, responds to velocity
- **Effect**: Higher = more damping, smoother response
- **Typical values**: `[0.2, 0.2, 0.5]`
- **Note**: Z-axis typically needs higher damping

### Attitude Control Gains (TOR = Torque)

**`P_COEFF_TOR`** - Proportional gains `[roll, pitch, yaw]`
- **What it does**: Responds to attitude error
- **Effect**: Higher = faster attitude correction
- **Typical values**: `[70000, 70000, 60000]`
- **Note**: Much larger than position gains (different units)

**`I_COEFF_TOR`** - Integral gains `[roll, pitch, yaw]`
- **What it does**: Eliminates attitude steady-state error
- **Effect**: Higher = faster error correction
- **Typical values**: `[0, 0, 500]` (roll/pitch often set to 0)
- **Note**: Yaw typically needs integral term

**`D_COEFF_TOR`** - Derivative gains `[roll, pitch, yaw]`
- **What it does**: Dampens attitude oscillations
- **Effect**: Higher = more damping, smoother attitude
- **Typical values**: `[20000, 20000, 12000]`

---

## Tuning Strategy

### Step 1: Start with Default Values
The default values in `DSLPIDControl.py` are well-tuned for Crazyflie drones. Start here.

### Step 2: Identify the Problem

**Overshoot/oscillation**:
- Reduce P gain
- Increase D gain

**Slow response**:
- Increase P gain
- Check if D gain is too high (causing damping)

**Steady-state error**:
- Increase I gain (carefully!)
- Too high I gain causes oscillation

**Unstable/diverging**:
- Reduce all gains
- Start with lower P, then gradually increase

### Step 3: Tune One Axis at a Time

Focus on one axis (e.g., Z-axis altitude) first:

```python
# Example: Tune only Z-axis position control
controller.setPIDCoefficients(
    p_coeff_pos=np.array([.4, .4, 1.5]),  # Increased Z P gain
    d_coeff_pos=np.array([.2, .2, .6]),  # Increased Z D gain for damping
)
```

### Step 4: Tune Position First, Then Attitude

1. **Position control** (outer loop) - affects trajectory following
2. **Attitude control** (inner loop) - affects stability

If position is good but attitude is shaky, tune attitude gains.

---

## Common Tuning Scenarios

### Scenario 1: Drone Oscillates Around Target

**Symptoms**: Drone overshoots and oscillates around waypoint

**Solution**:
```python
controller.setPIDCoefficients(
    p_coeff_pos=np.array([.3, .3, 1.0]),  # Reduce P gains
    d_coeff_pos=np.array([.3, .3, .6]),   # Increase D gains (more damping)
)
```

### Scenario 2: Slow Response to Commands

**Symptoms**: Drone takes too long to reach target position

**Solution**:
```python
controller.setPIDCoefficients(
    p_coeff_pos=np.array([.5, .5, 1.5]),  # Increase P gains
    d_coeff_pos=np.array([.15, .15, .4]), # Reduce D gains (less damping)
)
```

### Scenario 3: Altitude Drift

**Symptoms**: Drone doesn't maintain altitude accurately

**Solution**:
```python
controller.setPIDCoefficients(
    p_coeff_pos=np.array([.4, .4, 1.5]),  # Increase Z P gain
    i_coeff_pos=np.array([.05, .05, .08]), # Increase Z I gain
)
```

### Scenario 4: Unstable Attitude

**Symptoms**: Drone wobbles or tilts excessively

**Solution**:
```python
controller.setPIDCoefficients(
    p_coeff_att=np.array([60000., 60000., 50000.]),  # Reduce P gains
    d_coeff_att=np.array([25000., 25000., 15000.]),  # Increase D gains
)
```

### Scenario 5: Yaw Drift

**Symptoms**: Drone rotates slowly when it shouldn't

**Solution**:
```python
controller.setPIDCoefficients(
    i_coeff_att=np.array([.0, .0, 800.]),  # Increase yaw I gain
)
```

---

## Quick Reference: Default Values

```python
# Position Control (FOR)
P_COEFF_FOR = [0.4,   0.4,   1.25]  # [x, y, z]
I_COEFF_FOR = [0.05,  0.05,  0.05]  # [x, y, z]
D_COEFF_FOR = [0.2,   0.2,   0.5]   # [x, y, z]

# Attitude Control (TOR)
P_COEFF_TOR = [70000, 70000, 60000]  # [roll, pitch, yaw]
I_COEFF_TOR = [0,     0,     500]    # [roll, pitch, yaw]
D_COEFF_TOR = [20000, 20000, 12000]  # [roll, pitch, yaw]
```

---

## Tips for Effective Tuning

1. **Make small changes**: Change gains by 10-20% at a time
2. **Test thoroughly**: Run multiple simulations to verify stability
3. **Log data**: Use the Logger to visualize how changes affect performance
4. **Tune in simulation first**: Don't tune on real hardware until simulation is stable
5. **Document changes**: Keep notes on what values work best
6. **Consider physics model**: Different physics models (PYB_DW, etc.) may need different gains

---

## Example: Complete Tuning Session

```python
# In pid.py, after controller initialization:

#### Tune PID coefficients ####################
for controller in ctrl:
    # Start with slightly more aggressive position control
    controller.setPIDCoefficients(
        p_coeff_pos=np.array([.45, .45, 1.4]),      # Slightly higher
        i_coeff_pos=np.array([.05, .05, .06]),      # Slightly higher Z
        d_coeff_pos=np.array([.25, .25, .6]),       # More damping
        # Keep attitude gains at default (they're usually fine)
    )
```

---

## Troubleshooting

### Problem: Changes don't seem to take effect

**Solution**: Make sure you're calling `setPIDCoefficients()` **after** creating the controller but **before** the control loop.

### Problem: Drone becomes unstable after tuning

**Solution**: Revert to default values and make smaller changes.

### Problem: Different behavior with different physics models

**Solution**: Tune gains separately for each physics model (PYB, PYB_DW, etc.).

---

## Advanced: Per-Drone Tuning

You can tune each drone differently:

```python
ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

# Tune first drone more aggressively
ctrl[0].setPIDCoefficients(
    p_coeff_pos=np.array([.5, .5, 1.5]),
)

# Tune second drone more conservatively
ctrl[1].setPIDCoefficients(
    p_coeff_pos=np.array([.3, .3, 1.0]),
)
```

---

## Summary

- **Location**: PID coefficients in `DSLPIDControl.py` lines 37-42
- **Best method**: Use `setPIDCoefficients()` in `pid.py` (Method 1)
- **Start**: With default values
- **Tune**: One axis at a time, small changes
- **Test**: Thoroughly before making permanent changes

Happy tuning! 🚁

