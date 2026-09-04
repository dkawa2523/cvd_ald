# Process modes and reduced surface models

Read this reference when selecting, deriving, or modifying the response equation.

## Shared role vocabulary

Use roles as hypotheses about effects, not chemical labels:

- `A`: species whose arrival supplies a growth-relevant adsorbed state;
- `B`: species that promotes conversion of the adsorbed state into film;
- `I`: species that reduces available sites or otherwise suppresses the productive state;
- `*`: free surface site.

Raw CFD fields remain anonymous until independent chemical evidence supports a mapping.

## Steady CVD: minimal quasi-steady site balance

For a single productive adsorbed state and one competitive occupant, a useful minimum is

```text
theta_* + theta_A + theta_I = 1
d theta_A/dt = a theta_* - (d + q) theta_A
theta_I = i theta_*
v = H q theta_A
```

with

```text
a = k_ads C_A,s
d = k_loss
q = k_conv,A                     for A or AI
q = k_conv,AB C_B,s              for AB or AIB
i = K_I C_I,s
```

`d` is an observable finite loss group. Call it desorption only when data distinguish thermal desorption from nonproductive reaction or unmodelled transport loss.

Under the quasi-steady approximation,

```text
theta_A = a / {a + (d + q)(1 + i)}
v = H a q / {a + (d + q)(1 + i)}
```

This response combines low-concentration sensitivity, finite-site saturation, co-reactant conversion, and blocking without free empirical orders.

For AB/AIB, choose identification-data references and define

```text
u_A = C_A/C_A,ref
u_B = C_B/C_B,ref
u_I = C_I/C_I,ref
R = H a_ref
delta = k_loss/a_ref
b = k_conv,AB C_B,ref/a_ref
kappa = K_I C_I,ref
```

Then

```text
v = R u_A b u_B / {u_A + (delta + b u_B)(1 + kappa u_I)}
```

Store the references with the fit. Never recompute them from validation or test data.

## Useful exact candidates and reductions

For each raw-species assignment, consider only candidates that have distinct physical meaning:

- constant baseline;
- A;
- AI;
- unordered AB pair;
- AIB when three species are available;
- exact boundaries such as `delta = 0` when the boundary tests a real kinetic hypothesis.

For an AB steady response without inhibition, A/B exchange can be exactly symmetric after exchanging the corresponding kinetic coefficients. Enumerate the pair once and report its direction as unresolved. In a full AIB expression, retain orientations only if the inhibitor coupling makes their predictions distinct; still report evidence conservatively when the data do not separate them.

Compare a candidate with its exact reductions to assess effect necessity. A small numerical score advantage over a reduction is not sufficient evidence for an added effect.

## Structural identifiability limits

State these limits before interpreting coefficients:

1. Multiplying all surface rates by `gamma` and dividing the thickness-conversion factor `H` by `gamma` leaves the steady response unchanged. Steady maps do not identify absolute relaxation time or elementary rate constants.
2. The no-inhibitor AB steady response may be invariant to A/B exchange. More precise data of the same steady type will not identify direction.
3. Bulk concentration cannot be interpreted as surface concentration when transport depletion is appreciable.

Resolve the first two with qualitatively new observations such as step or pulse response, selective consumption/product flux, known chemistry, or surface-state measurement.

## Transport closure

When the supplied concentration is not a validated wall state, the surface model is an effective response. A simple transport closure is

```text
N_j = k_m,j (C_b,j - C_s,j)
N_j = nu_j r_s(C_s, theta)
```

or the CFD wall-gradient equivalent. Flow, pressure, geometry, diffusivity, wall flux, and sampling-plane data are needed to distinguish reaction control from mass-transfer control. Do not add a transport parameter to the fit unless these data constrain it.

## ALD: retain the process clock

Use the same raw-role hypotheses but preserve dose, purge, and cycle state when the observations contain them. A minimal ALD model may evolve coverage through time-dependent inlet functions and compute growth per cycle. It should expose which role is active during each step and whether purge removes the state.

Do not replace transient ALD evidence with the steady CVD equation. Conversely, do not build a transient ALD framework for a steady CVD map that contains no time information.

## When to expand the model

Add another site, intermediate, reversible step, nucleation state, or film-property state only when at least one of these is present:

- reproducible residual structure that contradicts the current limiting behavior;
- time data showing additional relaxation scales or memory;
- surface or product measurements showing the missing state;
- a designed perturbation that distinguishes the expanded candidate from the current reduction.

Otherwise keep the reduced model and record the missing evidence.
