# Reaction-role equations and physical interpretation

## Purpose and claim boundary

The model set converts Fluent species fields into a small number of testable surface
roles. Role (A) is a species whose arrival or storage is associated with growth;
(B) is a gas-phase or adsorbed conversion/regeneration partner; and (I) is a
competitive blocker. These symbols are hypotheses attached to raw species columns. They
are not chemical identities.

The steady equations are observable reductions. Their fitted parameters describe the
shape and scale of the measured deposition-rate response under the supplied conditions.
They become elementary kinetic constants only if surface concentration, site density,
stoichiometry, temperature dependence, and the assumed elementary steps are independently
established.

## Notation and units

| Symbol | Meaning | Unit in the implementation |
| --- | --- | --- |
| (C_{j,\mathrm{ref}}) | Fluent concentration at the stated reference plane | kmol m\(^{-3}\) |
| (C_{j,s}) | concentration adjacent to the reactive wall | kmol m\(^{-3}\) |
| (k_{m,j}) | film mass-transfer coefficient | m s\(^{-1}\) |
| (J_j) | wall-normal molar flux | kmol m\(^{-2}\) s\(^{-1}\) |
| \(X_j\) | explicitly selected steady reaction driver: \(C_{j,\mathrm{ref}}\), \(C_{j,s}\), or independent \(J_{j,\mathrm{cap}}\) | concentration or flux unit declared by the input mode |
| \(X_{j,0}\) | positive reference driver estimated from identification data | same as \(X_j\) |
| \(u_j=X_j/X_{j,0}\) | normalized steady reaction driver | 1 |
| \(\theta_j\) | occupied fraction of the modeled site or capacity pool | 1 |
| \(\theta_*\) | free-site fraction | 1 |
| \(\chi\) | oxidized fraction of a Mars-van Krevelen capacity pool | 1 |
| \(\Gamma_s\) | surface site or redox-capacity density | kmol m\(^{-2}\) |
| \(r\) | event rate per modeled site/capacity | s\(^{-1}\) |
| \(v\) | deposition rate | nm s\(^{-1}\) |
| \(h\) | film thickness | nm |
| \(R\) | profiled deposition-rate scale of a steady reduction | nm s\(^{-1}\) |

Concentrations in the current CSV census are bulk/reference-plane concentrations. The
same steady equations can use supplied wall concentrations through `direct_surface` or
an independently calculated wafer supply flux through `direct_flux`. One input mode is
fixed before chemical-model enumeration. Flux-response groups and concentration-response
groups are separate interpretations even when their normalized algebra is identical.

## Model inventory

### Surface reaction and state models

| Implemented model | Core response | Physical question and assumed reaction | Suitable observations and use | Advantages | Limitations | Principal references |
| --- | --- | --- | --- | --- | --- | --- |
| Constant baseline | (v=R) | Can one condition-independent rate explain all maps? No species role is assumed. | Required reference for every steady census. | Prevents an elaborate model being credited for a trend that a constant already explains. | Has no chemistry and no spatial response. | Statistical baseline; no kinetic attribution. |
| Total-concentration nuisance baseline | \(v=R(C_{\mathrm{tot}}/C_{\mathrm{tot},0})^n\) | Does a common concentration scale explain transfer without choosing any species role? | Steady concentration/rate datasets with changing total concentration; evaluated beside the constant baseline. | Detects improvement caused only by overall dilution or pressure scaling and prevents attributing it to a role equation. | Has no species or mechanism meaning; a fitted exponent can absorb several operating changes. | Empirical dimensional-analysis baseline; no kinetic attribution. |
| Single-(A) / (AI) saturation | (v=R u_A/[u_A+\lambda(1+\kappa u_I)]) | Does one adsorbing or storing species give a saturating response, optionally suppressed by a blocker? | Steady CVD with independent (A) variation through low-response and saturation regimes; (I) must also be varied for the (AI) form. | Smallest interpretable saturation model; exact no-(I) reduction. | Cannot represent a required co-reactant or distinguish adsorption from another saturating bottleneck. | Langmuir [1]. |
| Sequential `aib_qss` | (v=R u_A b u_B/[u_A+(\delta+b u_B)(1+\kappa u_I)]) | (A) occupies the surface, gas-phase (B) converts adsorbed (A), and (I) blocks the free-site pool: a Langmuir-Rideal-type hypothesis. | Default steady CVD comparison when (A/B) conditions include a low-(B) regime and the response is quasi-steady. | Represents saturation, sequential dependence, optional inhibition, and an exact finite-loss reduction with few parameters. | A no-(I) steady response is invariant to exchange of the two raw species after reparameterization; fitted groups are not elementary constants. | Langmuir [1]; Eley and Rideal [2]. |
| Parallel `parallel_a_ab_qss` | (v=R u_A(c+b u_B)/[u_A+(\delta+c+b u_B)(1+\kappa u_I)]) | Adsorbed (A) can convert through an (A)-only channel and an additional gas-(B) channel. | CVD when nonzero growth is plausible as (B\rightarrow0), with conditions spanning that limit and a (B)-responsive regime. | Separates the fractions assigned to (A)-only and (A+B) pathways. | (c) and (b u_B) are confounded when (B) changes little; the steady family has no independent dynamic state implementation. | Langmuir [1]; Eley and Rideal [2]. |
| `langmuir_hinshelwood_qss` | (v=R(a u_A)(b u_B)/(1+a u_A+b u_B+\kappa u_I)^2) | Both (A) and (B) adsorb competitively on one site pool and react as adsorbates. | Exploratory steady CVD comparison when both reactants independently traverse low coverage and saturation, or when adsorption/retention evidence for (B) exists. | Tests a physically different denominator and explicitly reports θA, θB, θI, and free sites. | Symmetric under (A/B) exchange with adsorption-parameter exchange; one film-rate map cannot establish coadsorption. | Langmuir [1]; standard Langmuir-Hinshelwood kinetics [3]. |
| Dynamic `role_cvd_aib` | (d\theta_A/dt=r_{\mathrm{ads}}-r_{\mathrm{des}}-\nu_A r_{\mathrm{event}}) | Does a continuously renewed adsorbed-(A) state and optional (B)-assisted conversion reproduce transient CVD behavior? | Time-resolved Fluent concentration histories and thickness or rate observations. | Couples surface coverage to independent (A/B) transport closures and exposes surface/transport flux consistency. | One stored state cannot represent multiple site types, reconstruction, nucleation, or detailed product networks. | Langmuir [1]; Eley and Rideal [2]. |
| Dynamic `role_cvd_mvk` | (d\chi/dt=r_{\mathrm{reg}}-r_{\mathrm{red}}) | Does (A) consume an oxidized surface/lattice reservoir while (B) regenerates it? | A/B switching, pulse or step response, preferably with an oxidation-state observable. | Encodes reservoir memory and separately reports reduction, regeneration, and relaxation time. | At steady state its two-reactant response collapses to the sequential no-loss form; steady film rate alone cannot identify the reservoir. | Mars and van Krevelen [4]. |
| Dynamic `role_ald_state` | (d\theta_A/dt=r_{\mathrm{store}}-r_{\mathrm{release}}-r_{\mathrm{convert}}), (d\theta_I/dt=r_{I,\mathrm{store}}-r_{I,\mathrm{release}}) | Do dose, purge, and co-reactant exposure act through stored precursor and inhibitor states? | Transient ALD dose/purge/cycle data with final thickness or GPC; state-sensitive observations improve identifiability. | Represents self-limiting storage, purge memory, conversion, and inhibition without introducing a species-first mechanism. | Explicit bounded substeps can require a small time step; final thickness alone poorly separates storage, release, and conversion rates. | Puurunen [5]; George [6]. |

The term “production” in output tables means that a family participates in the routine
steady comparison. It is not evidence that the corresponding chemical mechanism is true.
The Langmuir-Hinshelwood family remains exploratory but is included by `--models all`.

### Transport and net-film layers

| Layer | Implemented form | Appropriate use | Benefit | Limitation |
| --- | --- | --- | --- | --- |
| `direct_surface` | (C_s) supplied directly, equivalent to (k_m\rightarrow\infty) within the local film closure | Fluent wall or near-wall concentration already represents the kinetic boundary | Avoids fitting a transport coefficient | Does not calculate an absolute flux unless a compatible flux field is also supplied |
| Steady `direct_flux` input | \(u_j=J_{j,\mathrm{cap}}/J_{j,0}\) | Fluent supplies a nonnegative arrival or transport-capacity flux calculated independently of the fitted wall reaction | Preserves condition-specific wafer delivery directly and avoids inventing a concentration-to-flux conversion | Fitted groups are conditional flux-response parameters; a realized reactive flux would be circular input |
| `fit_scalar` | (J=k_m(C_{\mathrm{ref}}-C_s)), with scalar or supplied (k_m) field | Reference-plane concentrations with an independently chosen or fitted film coefficient | Simple coupling and useful transport-sensitivity study | Reaction and transport can be confounded; a fitted (k_m) is conditional on the film approximation |
| `from_cfd_flux_sink` | (k_{m,\mathrm{CFD}}=J_{\mathrm{cap}}/(C_{\mathrm{ref}}-C_b)), (k_m=\gamma k_{m,\mathrm{CFD}}) | CFD provides a transport-capacity flux under a documented boundary condition | Retains spatial CFD transport structure while allowing calibration by \(\gamma\) | A realized reactive flux must not be reused as transport capacity; units and sign must be known |
| Stagnant-film utility | (k_m=D_{\mathrm{eff}}/\delta_{\mathrm{eff}}) | Known diffusion coefficient and effective film thickness | Transparent limiting estimate | δeff is a model quantity, and scalar Fick diffusion omits multicomponent coupling |
| Rotating-disk utility | (k_m=C_kD^{2/3}\omega^{1/2}\nu^{-1/6}) | Laminar rotating-disk scaling with known diffusivity and viscosity | Links rotation to a physically interpretable transport scale | Geometry and flow assumptions may not match a reactor; ω=0 requires an explicit fallback |
| Bosanquet diffusivity option | (D_{\mathrm{eff}}^{-1}=D_m^{-1}+D_K^{-1}) | Molecular and Knudsen resistances act in series | Useful reduced pore-transport estimate | Does not replace a Maxwell-Stefan multicomponent wall model |
| `deposition_only` | (v_{\mathrm{net}}=v_{\mathrm{dep}}) | No observed etch or loss channel | Clear sign convention | Cannot explain net removal |
| `dep_etch_loss` | (v_{\mathrm{net}}=v_{\mathrm{dep}}-v_{\mathrm{etch}}-v_{\mathrm{loss}}) | Independent etch/loss rates or justified fractions are available | Keeps film accounting separate from surface mechanism selection | Fractions without independent observations are bookkeeping parameters, not identified pathways |

The transport utilities calculate candidate (k_m) fields. The active role pipeline
accepts `direct_surface`, `fit_scalar`, and `from_cfd_flux_sink`. Full Maxwell-Stefan
diffusion and Stefan-flow coupling are not implemented; multicomponent diffusion is a
known extension when dilute independent-film transport is inadequate [7,8].

For a (B)-consuming AIB event, the implemented transport-demand ratio is

\[
\phi_B=
\frac{\Gamma_s\nu_B k_{\mathrm{rxn}}
\theta_A^{p_A}\theta_*^{p_*}}
{C_{B,\mathrm{scale}}k_{m,B}},
\qquad
\frac{C_{B,s}}{C_{B,\mathrm{ref}}}=\frac{1}{1+\phi_B}.
\]

Thus (\phi_B\ll1) indicates weak depletion across the scalar film and
(\phi_B\gg1) indicates strong transport demand within this closure. It is a local
dimensionless balance, not the fraction of the inlet feed that reacts. The inhibitor
availability

\[
f_I=\frac{1}{1+K_I C_{I,\mathrm{ref}}}
\]

reports the free-site suppression assumed by the AIB model. Surface and transport
fluxes are emitted separately:

\[
J_{j,\mathrm{transport}}=k_{m,j}(C_{j,\mathrm{ref}}-C_{j,s}),
\qquad
J_{j,\mathrm{surface}}=\Gamma_s\nu_j r_j.
\]

Their agreement checks the local closure. Neither quantity can be inferred from the
present steady CSV data because that path has no calibrated (k_m), site density, or
wall concentration.

## Steady observable reductions

### Normalization and amplitude profiling

For every species (j), the identification set defines a reference for the explicitly
selected local driver

\[
X_{j,0}=\operatorname{median}_{n\in\mathcal T} X_{j,n},
\qquad u_{j,n}=\frac{X_{j,n}}{X_{j,0}}.
\]

This normalization removes arbitrary input scale from the nonlinear shape parameters.
For concentration input, \(X=C\); for independent wafer supply flux, \(X=J_{\mathrm{cap}}\).
A steady candidate has

\[
\hat v_n=R f(\mathbf u_n;\boldsymbol\phi),
\]

where (R\ge0) has units nm s\(^{-1}\), and φ contains positive dimensionless
shape parameters. For fixed φ, the weighted least-squares optimum is calculated
analytically:

\[
R^*(\boldsymbol\phi)=\max\left[
0,\frac{\sum_n w_n f_n v_n}{\sum_n w_n f_n^2}
\right].
\]

Profiling (R) reduces the nonlinear search dimension and gives an exact conditional
optimum. It also means that (R) absorbs site density, film conversion, and any rate
constant scale not independently measured.

Two nuisance responses bound the value of role assignment. The constant baseline uses
\(f=1\). The total-concentration baseline uses

\[
f_{\mathrm{tot}}=
\left(\frac{C_{\mathrm{tot}}}{C_{\mathrm{tot},0}}\right)^n,
\qquad 0.01\le n\le10.
\]

It is invariant to composition changes at fixed total concentration. Improvement by a
role equation over this baseline therefore cannot be explained only by a shared total-
concentration trend. The exponent is an empirical nuisance parameter and is never
reported as a reaction order of an elementary step.

### Single-species saturation and competitive inhibition

The smallest adsorbed-(A) balance is

\[
\frac{d\theta_A}{dt}=k_{\mathrm{ads}}C_{A,s}\theta_*
-k_{\mathrm{loss}}\theta_A,
\qquad
\theta_*+\theta_A+\theta_I=1.
\]

Assuming a rapidly equilibrated blocker,

\[
\theta_I=K_I C_{I,s}\theta_*,
\qquad
\theta_*=\frac{1-\theta_A}{1+K_I C_{I,s}}.
\]

With (d\theta_A/dt=0), a growth rate proportional to θA reduces to

\[
v=R\frac{u_A}{u_A+\lambda(1+\kappa u_I)},
\]

where λ is an effective half-saturation/loss ratio and
κ scales inhibitor coverage. Setting κ=0 gives the exact single-(A) reduction.
An apparent inhibitor effect is accepted only when the parent model improves its
no-(I) reduction across condition refits.

### Sequential AIB quasi-steady model

For first-order adsorption and an adsorbed-(A)/gas-(B) event,

\[
\begin{aligned}
r_{\mathrm{ads}} &= k_{\mathrm{ads}}C_{A,s}\theta_*,\\
r_{\mathrm{des}} &= k_{\mathrm{des}}\theta_A,\\
r_{AB} &= k_{\mathrm{rxn}}\theta_A\frac{C_{B,s}}{C_{B,\mathrm{scale}}},\\
\frac{d\theta_A}{dt} &= r_{\mathrm{ads}}-r_{\mathrm{des}}-\nu_A r_{AB}.
\end{aligned}
\]

Using the blocker relation above and imposing quasi-steady coverage yields

\[
\theta_A=
\frac{k_{\mathrm{ads}}C_{A,s}}
{k_{\mathrm{ads}}C_{A,s}+
\left(k_{\mathrm{des}}+\nu_A k_{\mathrm{rxn}}C_{B,s}/C_{B,\mathrm{scale}}\right)
(1+K_I C_{I,s})}
\]

where the compact executable form is more transparently written as

\[
v=R\frac{u_A b u_B}
{u_A+(\delta+b u_B)(1+\kappa u_I)}.
\]

The line above is the implemented definition. Its dimensionless groups correspond to

\[
\delta\sim\frac{k_{\mathrm{des}}}{k_{\mathrm{ads}}C_{A,0}},\qquad
b\sim\frac{\nu_A k_{\mathrm{rxn}}C_{B,0}}
{k_{\mathrm{ads}}C_{A,0}C_{B,\mathrm{scale}}},\qquad
\kappa\sim K_I C_{I,0},
\]

while (R) absorbs the remaining scale. The correspondence is conditional because the
steady fitter estimates (R,\delta,b,\kappa) directly and does not know
(\Gamma_s,\alpha_h,\nu_A), or the surface concentrations.

The `no_desorption` reduction sets δ=0. In the report this is described as removal
of a finite nonproductive-loss group. A performance difference does not identify the
loss physically as desorption; irreversible loss, deactivation, or an omitted pathway
can produce the same steady effect.

For the no-inhibitor AB form, exchange of (u_A) and (u_B), followed by a scale and
parameter transformation, leaves the response family unchanged. The code therefore
reports the pair as undirected unless an inhibitor, transient state, or external chemical
information breaks the symmetry.

### Parallel A and A+B model

Let adsorbed (A) convert by two channels,

\[
r_A=k_A\theta_A,\qquad
r_{AB}=k_{AB}\theta_A C_{B,s}/C_{B,\mathrm{scale}}.
\]

The normalized quasi-steady response is

\[
v=R\frac{u_A(c+b u_B)}
{u_A+(\delta+c+b u_B)(1+\kappa u_I)}.
\]

The pathway fractions reported by the code are

\[
f_A=\frac{c}{c+b u_B},\qquad
f_{AB}=\frac{b u_B}{c+b u_B},\qquad f_A+f_{AB}=1.
\]

The exact reductions test δ=0, (c=0), removal of (B), and removal of (I) where
applicable. Evidence for an (A)-only path requires data near (u_B=0); otherwise
(c) and (b u_B) have nearly the same effect and cannot be separated reliably.

### Two-adsorbate Langmuir-Hinshelwood model

Assume competitive adsorption on one uniform site pool:

\[
\theta_A=K_A C_{A,s}\theta_*,\quad
\theta_B=K_B C_{B,s}\theta_*,\quad
\theta_I=K_I C_{I,s}\theta_*.
\]

The site balance gives

\[
\theta_*=\frac{1}{1+K_A C_{A,s}+K_B C_{B,s}+K_I C_{I,s}}.
\]

For a bimolecular surface event (r_{AB}=k\theta_A\theta_B), normalization produces

\[
v=R\frac{(a u_A)(b u_B)}
{(1+a u_A+b u_B+\kappa u_I)^2}.
\]

The squared denominator is a direct consequence of multiplying two coverages that share
the same free-site denominator. The model assumes adsorption equilibrium, one class of
sites, no lateral interaction, and a rate-controlling surface event. Different adsorption
stoichiometry or multiple site pools would change the powers and denominator. The
implemented model should therefore be called evidence for this response form, not proof
of a microscopic Langmuir-Hinshelwood mechanism.

## Dynamic CVD models

### AIB coverage model with transport closure

The dynamic CVD model retains θA and integrates

\[
\frac{d\theta_A}{dt}=
k_{\mathrm{ads}}C_{A,s}\theta_*^{m}
-k_{\mathrm{des}}\theta_A
-\nu_A k_{\mathrm{rxn}}\theta_A^{p_A}\theta_*^{p_*}
\left(\frac{C_{B,s}}{C_{B,\mathrm{scale}}}\right)^{\mathbb 1_B},
\]

with

\[
\theta_*=\frac{1-\theta_A}{1+K_I C_{I,\mathrm{ref}}},\qquad
\frac{dh}{dt}=\alpha_h\Gamma_s r_{\mathrm{event}}.
\]

The local film balances are solved algebraically:

\[
C_{A,s}=
\frac{C_{A,\mathrm{ref}}+(\Gamma_s k_{\mathrm{des}}\theta_A)/k_{m,A}}
{1+(\Gamma_s k_{\mathrm{ads}}\theta_*^m)/k_{m,A}},
\]

\[
C_{B,s}=
\frac{C_{B,\mathrm{ref}}}
{1+\Gamma_s\nu_B k_{\mathrm{rxn}}\theta_A^{p_A}\theta_*^{p_*}/
(C_{B,\mathrm{scale}}k_{m,B})}.
\]

These expressions enforce (k_m(C_{\mathrm{ref}}-C_s)) against the modeled net wall
demand. They do not implement a multicomponent Stefan-Maxwell boundary layer.

### Mars-van Krevelen redox reservoir

The state χ is the fraction of modeled redox capacity available for reduction by (A):

\[
r_{\mathrm{red}}=k_{\mathrm{red}}C_{A,s}\chi,\qquad
r_{\mathrm{reg}}=k_{\mathrm{reg}}C_{B,s}(1-\chi),
\]

\[
\frac{d\chi}{dt}=r_{\mathrm{reg}}-r_{\mathrm{red}},\qquad
\frac{dh}{dt}=\alpha_h\Gamma_s r_{\mathrm{red}}.
\]

The independent film closures are

\[
C_{A,s}=\frac{C_{A,\mathrm{ref}}}
{1+\Gamma_s k_{\mathrm{red}}\chi/k_{m,A}},
\qquad
C_{B,s}=\frac{C_{B,\mathrm{ref}}}
{1+\Gamma_s\nu_B k_{\mathrm{reg}}(1-\chi)/k_{m,B}}.
\]

At kinetic steady state and with fixed surface concentrations,

\[
r_{\mathrm{red}}=r_{\mathrm{reg}}
=\frac{k_{\mathrm{red}}C_{A,s}\,k_{\mathrm{reg}}C_{B,s}}
{k_{\mathrm{red}}C_{A,s}+k_{\mathrm{reg}}C_{B,s}}.
\]

After normalization this is the same functional family as sequential AB with δ=0.
The steady census therefore gives it one representative rather than an additional model
selection vote. MvK discrimination requires a transient where the reservoir memory

\[
\tau_{\mathrm{redox}}=
\left(k_{\mathrm{red}}C_{A,s}+k_{\mathrm{reg}}C_{B,s}\right)^{-1}
\]

changes the response, or an independent measurement of χ.

## Dynamic ALD storage model

The ALD model uses

\[
\theta_*=1-\theta_A-\theta_I,
\]

\[
\frac{d\theta_A}{dt}=
k_{\mathrm{store},A}C_{A,s}\theta_*
-k_{\mathrm{release},A}\theta_A-r_{\mathrm{conv}},
\]

\[
\frac{d\theta_I}{dt}=
k_{\mathrm{store},I}C_{I,s}\theta_*
-k_{\mathrm{release},I}\theta_I.
\]

The conversion channel is selected by the presence of (B):

\[
r_{\mathrm{conv}}=
\begin{cases}
k_{\mathrm{convert},A}\theta_A, & B\text{ absent},\\
k_{\mathrm{convert},AB}C_{B,s}\theta_A, & B\text{ present}.
\end{cases}
\qquad
\frac{dh}{dt}=\alpha_h r_{\mathrm{conv}}.
\]

Here \(r_{\mathrm{conv}}\) has units of coverage per second and \(\alpha_h\) has
units of nanometres per unit coverage converted. With site density
\(\Gamma_s\,[\mathrm{kmol\,m^{-2}}]\), the absolute role fluxes are

\[
J_{A,s}=\Gamma_s\left(
k_{\mathrm{store},A}C_{A,s}\theta_*
-k_{\mathrm{release},A}\theta_A\right),
\qquad
J_{B,s}=\Gamma_s\nu_B r_{\mathrm{conv}}.
\]

Storage and release enter the (A) film closure without double-counting conversion:

\[
C_{A,s}=
\frac{k_{m,A}C_{A,\mathrm{ref}}+
\Gamma_s k_{\mathrm{release},A}\theta_A}
{k_{m,A}+\Gamma_s k_{\mathrm{store},A}\theta_*},
\]

and the (B) sink gives

\[
C_{B,s}=
\frac{k_{m,B}C_{B,\mathrm{ref}}}
{k_{m,B}+\Gamma_s\nu_Bk_{\mathrm{convert},AB}\theta_A}.
\]

These expressions enforce \(k_m(C_{\mathrm{ref}}-C_s)=J_s\) in
\(\mathrm{kmol\,m^{-2}\,s^{-1}}\). The explicit \(\Gamma_s\) factor prevents a
coverage rate from being compared directly with a molar transport flux. Setting
\(\Gamma_s=1\) is permitted for normalized studies, but the resulting flux magnitude
is then normalized rather than absolute.

This is a minimal latent-state model for role assimilation. It describes storage,
release, and conversion but makes no claim about a named ligand-exchange sequence or
specific surface termination.

## Estimation, discrimination, and visualization quantities

### Whole-wafer Loss functions

Let condition \(k\) contain observations \(y_{ki}\), predictions
\(\hat y_{ki}\), and nonnegative point weights \(q_{ki}\). The implementation first
normalizes weights within every wafer,

\[
w_{ki}=\frac{q_{ki}}{\sum_iq_{ki}},
\qquad \sum_iw_{ki}=1,
\]

and then averages the condition Loss values. Thus every identification wafer receives
one vote even when maps contain different point counts. The fitted coefficients are
shared by all identification wafers; the code does not fit an independent kinetic
equation to each wafer.

| CLI name | Condition Loss \(L_k\) | Use | Main limitation |
| --- | --- | --- | --- |
| `mse` | \(\sum_iw_{ki}(\hat y_{ki}-y_{ki})^2\) | Dimensional linear-rate fit; preserves the physical cost of an absolute deposition-rate error | High-rate conditions can dominate the numerical scale |
| `wafer_normalized_mse` | \(\sum_iw_{ki}(\hat y_{ki}-y_{ki})^2/s_k^2\), \(s_k^2=\sum_iw_{ki}y_{ki}^2\) | Gives low- and high-rate wafers comparable relative influence | Treats equal fractional errors as equally costly and loses nm\(^2\) s\(^{-2}\) units |
| `wafer_normalized_mae` | \(\sum_iw_{ki}|\hat y_{ki}-y_{ki}|/s_k\) | Relative, less sensitive to isolated residuals | Nondifferentiable at zero and can underweight systematic small residual structure |
| `symmetric_normalized_mse` | \(2\sum_iw_{ki}(\hat y_{ki}-y_{ki})^2/\sum_iw_{ki}(y_{ki}^2+\hat y_{ki}^2)\) | Symmetric scaling when neither magnitude should define the denominator alone | Its scale depends on the prediction and is less direct physically |

The total objective is \(L=K^{-1}\sum_k L_k\). An optional radial uncertainty model
multiplies the point weights through the declared center-to-edge standard-uncertainty
ratio. It is evidence-based weighting only when uncertainty or replicate variance
supports it; otherwise it is a sensitivity analysis. Objective values from different
Loss definitions are not compared directly. Candidate and optimizer comparisons return
to dimensional condition-CV and heldout RMSE.

### Prediction and wafer-shape metrics

For one heldout wafer with \(N\) points, define residual
\(e_i=\hat y_i-y_i\), observed mean \(\bar y\), and predicted mean
\(\bar{\hat y}\). The ordinary rate metrics are

\[
\operatorname{RMSE}=\sqrt{\frac1N\sum_i e_i^2},\qquad
\operatorname{bias}=\frac1N\sum_i e_i,
\qquad
\operatorname{relative\ RMSE}=\frac{\operatorname{RMSE}}{|\bar y|}.
\]

Condition-mean transfer and wafer shape are separated by centering:

\[
e_i^{\circ}=(\hat y_i-\bar{\hat y})-(y_i-\bar y),
\]

\[
\operatorname{RMSE}_{\mathrm{centered}}=
\sqrt{\frac1N\sum_i(e_i^{\circ})^2},
\qquad
R^2_{\mathrm{centered}}=
1-\frac{\sum_i(e_i^{\circ})^2}{\sum_i(y_i-\bar y)^2}.
\]

A negative centered \(R^2\) means that the predicted pattern is worse than assigning
the correct measured mean to every point. It is compatible with a small ordinary RMSE
when the condition mean is predicted well but the in-plane amplitude or pattern is not.
Spatial correlation measures phase agreement after centering, while the ratio of
predicted to observed range measures amplitude capture; neither replaces centered
error.

### Role importance and assignment stability

For assigned role \(j\), let \(\hat y_i\) be the selected-model prediction and
\(\hat y_i^{(-j)}\) the prediction after replacing that role's local input by its
identification reference \(X_{j,0}\). The code condition-balances the squared difference:

\[
S_j=\left[
\frac1K\sum_{k=1}^K\frac1{N_k}
\sum_{i\in k}(\hat y_i-\hat y_i^{(-j)})^2
\right]^{1/2}.
\]

This is a one-at-a-time prediction sensitivity. Nonlinear interaction means that the
\(S_j\) values do not sum to the deposition rate or to one. Let \(f_j\) be the fraction
of outer condition refits selecting the same raw species in role \(j\), and let \(E\)
be the selected model's fixed-heldout RMSE. The dimensionless scale comparison

\[
Q_j=\frac{S_j}{E}
\]

separates two practically different forms of non-identification. Low \(f_j\) with
\(Q_j\ll1\) is unstable but predictively negligible in the tested range. Low \(f_j\)
with \(Q_j\gtrsim1\) is an influential unresolved assignment. \(Q_j=1\) is a visual
reference, not a universal statistical rejection threshold.

### Alternative-family prediction difference

Let \(\hat y_{m,i}\) be the heldout prediction from the best candidate in family \(m\)
and \(\hat y_{\star,i}\) the selected-family prediction at the same coordinates. The
model-conditional prediction separation is

\[
D_m=\sqrt{\frac1N\sum_i
(\hat y_{m,i}-\hat y_{\star,i})^2},
\qquad H_m=\frac{D_m}{E}.
\]

Small \(H_m\) means that the fitted reaction interpretation changes with little
consequence for the tested prediction. Large \(H_m\) means that family ambiguity is
also a prediction risk. These quantities do not assign mechanism probabilities; they
show the consequence of choosing among the fitted observable equations.

### Local kinetic-ratio sensitivity and partial Loss slices

For positive fitted shape parameter \(p_j\), the local logarithmic sensitivity at data
point \(i\) is

\[
g_{ij}=\frac{\partial\ln \hat y_i}{\partial\ln p_j}.
\]

The implementation reports

\[
G_j=\sqrt{\frac1N\sum_i g_{ij}^2}
\]

and the Pearson correlation between centered columns \(g_{·j}\) and
\(g_{·\ell}\). A small \(G_j\) identifies a locally inactive direction. A
correlation magnitude near one indicates that two directions create almost the same
spatial response after sign and scaling, so their separate values are practically weak.
This derivative design diagnoses local information; it does not by itself provide a
global uncertainty interval [12,14].

For a plotted Loss slice, one parameter \(p_j\) is fixed on a logarithmic grid, the
other shape parameters remain at their fitted values, and only the separable nonnegative
rate scale \(R\) is reprofiled:

\[
\widetilde L_j(p)=\min_{R\ge0}
L\{R f(\mathbf u;p,\hat{\boldsymbol\phi}_{-j})\}.
\]

A broad flat slice is direct evidence that the current observations scarcely constrain
that direction under the selected equation. It is a partial slice rather than a full
profile likelihood, because \(\boldsymbol\phi_{-j}\) is not reoptimized. Formal
likelihood intervals require a noise model and joint reprofiling [14].

### Post-selection spatial residual response

The optional spatial stage begins only after the chemical model and its coefficients
are frozen. Let \(\rho_i\) be normalized wafer radius and let the basis contain
\(\rho^2\), or \((\rho^2,\rho^4)\). Every basis column is centered within each
identification condition, producing \(\Phi_{ki}\). The fitted target is the centered
log residual:

\[
d_{ki}=\{\ln y_{ki}-\overline{\ln y}_k\}
-\{\ln\hat y^{\mathrm{chem}}_{ki}
-\overline{\ln\hat y^{\mathrm{chem}}}_k\},
\]

\[
\hat{\boldsymbol\beta}
=\arg\min_{\boldsymbol\beta}
\frac1K\sum_k\frac1{N_k}
\sum_{i\in k}(d_{ki}-\Phi_{ki}\boldsymbol\beta)^2.
\]

The raw factor is \(g_i=\exp(\Phi_i\hat{\boldsymbol\beta})\). On each application
wafer it is rescaled so that

\[
\hat y_i^{\mathrm{corr}}=
\hat y_i^{\mathrm{chem}}g_i
\frac{\overline{\hat y^{\mathrm{chem}}}}
{\overline{\hat y^{\mathrm{chem}}g}},
\qquad
\overline{\hat y^{\mathrm{corr}}}
=\overline{\hat y^{\mathrm{chem}}}.
\]

The spatial response therefore cannot repair the chemical condition mean and cannot
change role or family selection. Its coefficients describe a transferable radial
residual basis. They do not identify temperature, transport, chamber geometry, or
another physical cause without a corresponding measured field. Treating model
discrepancy separately from calibration parameters is also necessary to avoid assigning
omitted spatial physics to kinetics [15].

### Figures as evidence views

| Figure class | Primary numerical source | Valid conclusion |
| --- | --- | --- |
| Optimization convergence | `optimization_history.csv` | Numerical progress for each best family candidate |
| Equation comparison and reaction path | `role_ranking.csv`, family registry | Transfer error, selection stability, and assumed topology |
| Alternative-model agreement | `reaction_model_predictions.csv` | Prediction consequence of family ambiguity |
| Role stability and importance | `role_stability.csv`, `role_importance_and_stability.csv` | Whether assignment uncertainty is harmless or influential |
| State and pathway fractions | `reaction_model_states.csv`, `reaction_state_summary.csv` | Model-conditional occupation and rate allocation |
| Parameter sensitivity and Loss slices | `parameter_sensitivity_correlations.csv`, `parameter_loss_slices.csv` | Local weak and coupled parameter directions |
| Heldout maps and radial profiles | `test_predictions.csv` | Mean transfer, spatial phase, amplitude, and residual structure |
| Spatial-response figures | `spatial_response_summary.csv`, `spatial_response_coefficients.csv` | Predictive benefit of the separate residual basis |

The reaction-path arrows, fitted fractions, and spatial basis are explanatory views of
declared equations. They are not independent surface measurements. Figures therefore
remain subordinate to the split, source artifact, units, and observation semantics.

## Approximation hierarchy and interpretation

The implemented models occupy different levels and should not be ranked in one flat
list without compatible observations:

1. The steady equation census compares observable film-rate response shapes.
2. Dynamic CVD and ALD models compare state memory against time-resolved observations.
3. Transport closures determine how the supplied Fluent location is connected to the
   surface.
4. Net-film models combine independently supported deposition, etch, and loss rates.

Several different physical mechanisms can share one steady algebraic form. Conversely,
the same chemistry can appear to follow different apparent equations when transport,
temperature, or unobserved surface state changes. A low CV error therefore establishes
predictive compatibility within the tested domain. Mechanism adoption additionally
requires role stability, exact-reduction evidence, adequate input contrast, and a
mechanism-specific observable.

## References

1. I. Langmuir, “The Adsorption of Gases on Plane Surfaces of Glass, Mica and Platinum,” *Journal of the American Chemical Society* **40** (1918) 1361–1403. [doi:10.1021/ja02242a004](https://doi.org/10.1021/ja02242a004).
2. D. D. Eley and E. K. Rideal, “The Catalysis of the Parahydrogen Conversion by Tungsten,” *Proceedings of the Royal Society A* **178** (1941) 429–451. [doi:10.1098/rspa.1941.0066](https://doi.org/10.1098/rspa.1941.0066).
3. C. N. Hinshelwood, *The Kinetics of Chemical Change in Gaseous Systems*, Oxford University Press, 1926.
4. P. Mars and D. W. van Krevelen, “Oxidations Carried Out by Means of Vanadium Oxide Catalysts,” *Chemical Engineering Science*, Special Supplement **3** (1954) 41–59. [doi:10.1016/S0009-2509(54)80005-4](https://doi.org/10.1016/S0009-2509(54)80005-4).
5. R. L. Puurunen, “Surface Chemistry of Atomic Layer Deposition: A Case Study for the Trimethylaluminum/Water Process,” *Journal of Applied Physics* **97** (2005) 121301. [doi:10.1063/1.1940727](https://doi.org/10.1063/1.1940727).
6. S. M. George, “Atomic Layer Deposition: An Overview,” *Chemical Reviews* **110** (2010) 111–131. [doi:10.1021/cr900056b](https://doi.org/10.1021/cr900056b).
7. R. Krishna and J. A. Wesselingh, “The Maxwell-Stefan Approach to Mass Transfer,” *Chemical Engineering Science* **52** (1997) 861–911. [doi:10.1016/S0009-2509(96)00458-7](https://doi.org/10.1016/S0009-2509(96)00458-7).
8. J. A. Wesselingh and R. Krishna, “Stefan-Maxwell Mass Transport,” *Chemical Engineering Science* **64** (2009) 4796–4803. [doi:10.1016/j.ces.2009.07.002](https://doi.org/10.1016/j.ces.2009.07.002).
9. V. G. Levich, *Physicochemical Hydrodynamics*, Prentice-Hall, Englewood Cliffs, 1962.
10. E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed., Springer, 1996. [doi:10.1007/978-3-642-05221-7](https://doi.org/10.1007/978-3-642-05221-7).
11. S. Varma and R. Simon, “Bias in Error Estimation When Using Cross-Validation for Model Selection,” *BMC Bioinformatics* **7** (2006) 91. [doi:10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91).
12. G. Franceschini and S. Macchietto, “Model-Based Design of Experiments for Parameter Precision: State of the Art,” *Chemical Engineering Science* **63** (2008) 4846–4872. [doi:10.1016/j.ces.2007.11.034](https://doi.org/10.1016/j.ces.2007.11.034).
13. B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall/CRC, 1993. [doi:10.1007/978-1-4899-4541-9](https://doi.org/10.1007/978-1-4899-4541-9).
14. A. Raue, C. Kreutz, T. Maiwald, J. Bachmann, M. Schilling, U. Klingmüller, and J. Timmer, “Structural and Practical Identifiability Analysis of Partially Observed Dynamical Models by Exploiting the Profile Likelihood,” *Bioinformatics* **25** (2009) 1923–1929. [doi:10.1093/bioinformatics/btp358](https://doi.org/10.1093/bioinformatics/btp358).
15. M. C. Kennedy and A. O'Hagan, “Bayesian Calibration of Computer Models,” *Journal of the Royal Statistical Society: Series B* **63** (2001) 425–464. [doi:10.1111/1467-9868.00294](https://doi.org/10.1111/1467-9868.00294).
