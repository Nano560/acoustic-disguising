# SHT-based wave-field separation on the outer shell

This note walks step-by-step through the separation that replaces the
high-frequency / normal-incidence approximation
`p_in ≈ (p − z₀·v_n)/2` in `python/mdd/wavefield_separation.py`.

## What the separation does

You have, on the outer Fibonacci shell of radius $r_2 = 0.3$ m, two
real-valued time-domain tensors of shape $(n_t, n_{\text{outer}}, n_{\text{ill}})$:
the pressure $p(\Omega_j, t)$ and the impedance-scaled normal velocity
$z_0 v_n(\Omega_j, t)$ (`outer_vnz`). MDD needs the **incoming** part of
the pressure on the same shell — the half of the field that, in the
spherical-Hankel sense, is propagating radially inward. Everything below
operates per illumination source independently; ignore that axis for the
explanation.

The exterior pressure field at any frequency $\omega$ can be written as

$$
\hat p(r,\Omega,\omega) \;=\; \sum_{n=0}^{\infty}\sum_{m=-n}^{n}\bigl[A_{nm}(\omega)\,h_n^{(1)}(kr) \;+\; B_{nm}(\omega)\,h_n^{(2)}(kr)\bigr]\,Y_n^m(\Omega),
$$

with $k = \omega/c_0$. Under the $e^{-i\omega t}$ convention used by NumPy
and pylops, $h_n^{(1)}$ is the outgoing radial mode and $h_n^{(2)}$ is the
incoming one. The radial particle velocity is
$\hat v_n = \partial_r \hat p / (i\omega\rho_0)$, so multiplying by
$z_0 = \rho_0 c_0$ gives
$\widehat{z_0 v_n} = -i\,(A_{nm} h_n^{(1)\prime}(kr) + B_{nm} h_n^{(2)\prime}(kr))\,Y_n^m$.
The job is to recover $B_{nm}(\omega)$ for every mode and every frequency.

The pipeline does this in nine steps.

## Step 1. Cartesian → spherical angles

`_spherical_to_angles(outer_positions, radius)` converts each
Fibonacci-sphere point $\mathbf x_j = (x_j, y_j, z_j)$ to $(\theta_j, \phi_j)$
with $\theta = \arccos(z/r)\in[0,\pi]$, $\phi = \mathrm{atan2}(y,x)\in[-\pi,\pi]$,
after checking that $\|\mathbf x_j\| \approx r_2$ for every sample. This
is geometry only, frequency-independent. Output: two arrays of length
$n_{\text{outer}}=300$.

## Step 2. Build the spherical-harmonic design matrix Y

`_build_sh_design(theta, phi, n_max)` constructs $Y \in \mathbb C^{M\times K}$
with $M = n_{\text{outer}}$, $K = (N+1)^2$, and entries
$Y_{j,(n,m)} = Y_n^m(\theta_j, \phi_j)$. The columns are stored in
$(n=0..N,\ m=-n..n)$ row-major order so a parallel array `mode_n[k]` records
the degree of column $k$. With $N = 12$ and $M = 300$ this matrix is
$300\times 169$. It depends only on the geometry, so build it once.

## Step 3. Pre-compute the projector Y⁺

The Fibonacci sampling is not a quadrature rule, so the SH projection is a
damped least-squares fit:

$$
\mathbf p_{nm}(\omega) \;\approx\; \arg\min_{\mathbf c}\,\bigl\|\,\mathbf Y\,\mathbf c - \hat{\mathbf p}(\omega)\bigr\|^2 + \lambda\,\|\mathbf c\|^2 \;=\; (\mathbf Y^H \mathbf Y + \lambda I)^{-1} \mathbf Y^H\,\hat{\mathbf p}(\omega).
$$

`_sh_pinv(Y, reg)` returns the matrix on the right and stores it once. The
regularisation parameter is **`sht_reg`** ($\lambda$, default $10^{-6}$).
On 300 quasi-uniform points with $K=169$, $\mathbf Y^H \mathbf Y$ is
well-conditioned and $\lambda$ only stabilises the highest-degree tail.

## Step 4. Forward FFT in time

`np.fft.rfft(outer_p, axis=0)` and the same on `outer_vnz`. Real-valued
input gives a Hermitian-symmetric spectrum, so we only need the
positive-frequency half, $n_f = \lfloor n_t/2 \rfloor + 1$ bins. The
frequency axis is `np.fft.rfftfreq(nt, d=dt)` with the wavenumber array
$k_f = 2\pi f/c_0$. Output shape: $(n_f, M, n_{\text{ill}})$ complex.

## Step 5. Project each frequency onto the SH basis

A single Einstein-summation contraction:

```python
P_nm = einsum("kp,fpi->fki", Y_pinv, P_fft)
V_nm = einsum("kp,fpi->fki", Y_pinv, V_fft)
```

This produces $\hat p_{nm}(\omega)$ and $\widehat{z_0 v_n}_{nm}(\omega)$ of
shape $(n_f, K, n_{\text{ill}})$.

## Step 6. Per-(n, ω) radial coefficients α and β

For each $n=0..N$ at fixed $kr_2$, `_radial_coefficients(n_max, kr)`
evaluates with `scipy.special.spherical_jn` / `spherical_yn` (and their
derivatives):

$$
\alpha_n(x) \;=\; 1 - i\,x^2\bigl(j_n(x)\,j_n'(x) + y_n(x)\,y_n'(x)\bigr) \quad\text{(complex)},
$$

$$
\beta_n(x) \;=\; x^2\bigl(j_n(x)^2 + y_n(x)^2\bigr) \quad\text{(real, non-negative)}.
$$

These come from solving the radial 2×2 system

$$
\begin{bmatrix} h_n^{(1)}(kr_2) & h_n^{(2)}(kr_2) \\[2pt] -i\,h_n^{(1)\prime}(kr_2) & -i\,h_n^{(2)\prime}(kr_2)\end{bmatrix}\!\begin{bmatrix}A_{nm} \\ B_{nm}\end{bmatrix} = \begin{bmatrix}\hat p_{nm} \\ \widehat{z_0 v_n}_{nm}\end{bmatrix}
$$

in closed form via the spherical-Bessel Wronskian $j_n y_n' - y_n j_n' = 1/x^2$,
then evaluating $B_{nm}\,h_n^{(2)}(kr_2)$ — which simplifies to
$\tfrac12(\alpha_n \hat p_{nm} - \beta_n \widehat{z_0 v_n}_{nm})$. In the
$kr \to \infty$ limit $j^2 + y^2 \to 1/x^2$ and $j j' + y y' \to 0$, so
$\alpha_n \to 1$, $\beta_n \to 1$, and the formula collapses to the legacy
$(p - z_0 v_n)/2$.

## Step 7. Frequency-dependent mode truncation

For each frequency bin `f_idx` we determine the highest reliable degree

$$
n_{\text{eff}}(\omega) \;=\; \min\!\bigl(N,\ \lceil k r_2 \rceil + n_{\text{pad}}\bigr),
$$

and zero out modes with $n > n_{\text{eff}}$. This is the most important
and most subtle parameter (**`sht_n_pad`**, default 1). It exists because
$y_n(x) \sim -(2n-1)!!\,x^{-(n+1)}$ diverges for $n > x$, which makes
$\beta_n \sim x^{-2n}$ blow up. The closed-form formula then relies on
catastrophic cancellation between two huge numbers, and floating-point
noise on `outer_vnz` at evanescent modes gets amplified by orders of
magnitude. With `n_pad = 1` you keep one extra mode beyond the radiating
cutoff for safety; with `n_pad ≥ 3` the separation breaks (verified — for
an offset-source field, `n_pad=4` produced a "incoming" estimate three
times larger than the input). This matches the standard rule from
spherical-microphone-array literature.

The radial split itself is then a vectorised broadcast over $m$ and the
illumination axis at each frequency:

```python
Pin_nm[f, keep, :] = 0.5 * (alpha_n[mode_n[keep]] * P_nm[f, keep, :]
                          - beta_n [mode_n[keep]] * V_nm[f, keep, :])
```

## Step 8. Re-synthesise on the spatial grid

Another einsum contraction reverses the SH projection,

```python
Pin_fft = einsum("pk,fki->fpi", Y, Pin_nm)
```

evaluating $\sum_{n,m} B_{nm}(\omega)\,h_n^{(2)}(kr_2)\,Y_n^m(\Omega_j)$ at
each Fibonacci sample point. Output shape: $(n_f, M, n_{\text{ill}})$ complex.

## Step 9. Inverse FFT to time

`np.fft.irfft(Pin_fft, n=nt, axis=0)` returns the real-valued time-domain
incoming pressure of shape $(n_t, n_{\text{outer}}, n_{\text{ill}})$. The
result is real because the radial-split operator is Hermitian-symmetric in
$\omega$: $\alpha_n(-\omega) = \overline{\alpha_n(\omega)}$ and $\beta_n$
is real and even, which is a consequence of $j_n$ being even and $y_n$
odd in $x$ for the parities of $n$ involved. This `outer_pin` tensor goes
straight into `pylops.waveeqprocessing.MDD` exactly where the legacy
`(outer_p − outer_vnz)/2` used to.

---

## The parameters, summarised

**`separation_method`** (str, default `"sht"`). Selects between `"sht"`
(this nine-step pipeline) and `"legacy"` (the asymptotic high-frequency
formula). Keep `"legacy"` available for comparison plots in the paper.

**`sht_n_max`** (int, default 12). The spherical-harmonic truncation
degree $N$. Capped above by the angular Nyquist of the Fibonacci grid:
$(N+1)^2 \lesssim n_{\text{outer}}$, so for 300 points $N \le 16$. Capped
from physics by $N \gtrsim k_{\max} r_2$ — modes with $n > k r_2$ at any
given frequency are evanescent and contribute nothing (Step 7 truncates
them anyway). For `fmax_hz = 12 kHz` and `r_outer = 0.3 m`, the relevant
$N$ is around $k r_2 \approx 15$, which is right at the angular-Nyquist
limit.

**`sht_n_pad`** (int, default 1). The safety margin on the per-frequency
truncation $n_{\text{eff}}(\omega) = \lceil k r_2 \rceil + n_{\text{pad}}$.
**Do not raise above 2.** Raising it does not improve accuracy — it lets
evanescent-mode noise leak into the result. Verified empirically.

**`sht_reg`** (float, default $10^{-6}$). Tikhonov damping $\lambda$ on
the SH least-squares projection. With 300 well-spread sample points and
$N \le 15$, the projector is well-conditioned and $\lambda$ is essentially
cosmetic; raise to $10^{-3}$ if you want extra robustness against noisy
data, lower to $10^{-9}$ if you want the cleanest reconstruction on
synthetic data.

**`sound_speed`** (float, default 1500 m/s). Background acoustic sound
speed, used to form $k = \omega/c_0$. Read from the reverb HDF5 attrs
(`dom_c0`) when present, otherwise from `[mdd].sound_speed` in the TOML,
otherwise from this default. Mismatch with the FDTD value would smear the
Hankel arguments and degrade the cancellation.

**`fmax_hz`** (float, inherited from `[mdd].fmax_hz`). Upper frequency
cap. Bins above this are zeroed in `Pin_fft`. Set this to your actual
signal bandwidth — for an 18 kHz Ricker that's around 30–40 kHz of
energy, with practically nothing past 50 kHz.

**The outer-shell radius `r_2`** is not a knob — it's read from the
reverb HDF5 attrs (`radius_outer`, currently 0.3 m) and verified against
the geometry of `outer_positions` to within 0.5% relative tolerance.

**One implicit parameter: the angular Nyquist.** The Fibonacci sampling on
`nPoints_outer = 300` resolves $N \le 15$ and therefore $k r_2 \le 15$,
i.e. $f \le c_0 N/(2\pi r_2) \approx 12$ kHz. Above 12 kHz the SHT
separation becomes angularly aliased on the outer shell and degrades to
no better than legacy. This is a property of the data, not the
algorithm; the only way to push the band higher is to increase
`nPoints_outer` and rerun the reverb stage.

---

## Validation

Synthetic tests on a 300-point Fibonacci sphere of radius 0.3 m, fc =
8–18 kHz, sampling rate 100 kHz:

| Test                                                 | SHT         | Legacy      |
|------------------------------------------------------|-------------|-------------|
| Outgoing monopole at origin → expect $p_{\rm in}\approx 0$ | $5\times10^{-7}$ | $5\times10^{-2}$ |
| Incoming spherical wave → expect $p_{\rm in} = p$    | $5\times10^{-6}$ rel | — |
| Outgoing+incoming superposition → recover incoming   | $2\times10^{-5}$ rel | $0.19$ rel |
| Offset monopole (multipoles required) → expect $p_{\rm in}\approx 0$ | $1.5\times10^{-7}$ | $5.8\times10^{-2}$ |
| Plane wave from $+\hat x$ → spherical-Hankel modal split | $0.47$ ($\approx \tfrac12$) | — |

The first four numbers are the relative magnitude of the spurious
incoming component; the last is the modal-energy split predicted by
$j_n = \tfrac12(h_n^{(1)} + h_n^{(2)})$.

---

## A third option: local plane-wave / two-shell decomposition

The 2D acoustic-disguising pipeline (`mdd/functions/curvedArrayDecomTwoLayers.m`)
uses a fundamentally different approach that's worth comparing.
For each receiver point on the outer shell, it takes a tangent-plane
patch of K nearest outer points plus K nearest inner points, builds a
local frame with $+z$ along the outward radial direction, and solves a
damped least-squares fit to a small dictionary of plane waves
$\exp(i(k_x x + k_y y \pm k_z z))$. The two signs of $k_z$ split incoming
from outgoing; pressure on both shells is the only data needed.

A 3D port is implemented in `python/mdd/wavefield_separation_local.py`,
exposed as `separation_method = "local_pw"`. The headline advantage is
that the relevant Nyquist is the LOCAL patch sampling, not the full-shell
sampling — so it's not bandwidth-limited by `nPoints_outer` the way SHT
is. The headline disadvantage is that the method has a fundamental
spectral failure mode that 2D escapes and 3D doesn't, plus a magnitude-
matching bias.

**Two-shell singularities at $f = n \cdot c_0 / (2 \Delta r)$.** With
shells at $r_2 = 0.3$ m and $r_1 = 0.2$ m and $c_0 = 1500$ m/s, these
sit at 7.5, 15, 22.5, … kHz. At those frequencies the inward and outward
plane-wave bases at normal incidence become indistinguishable on the
two-shell sample (the 2×2 determinant goes to zero), so any noise in the
data partitions arbitrarily between the in and out modes. An 18 kHz Ricker
spectrum sits squarely on top of two of these singularities. Stronger
regularisation hides the spikes but biases the result toward zero.

**Magnitude-matching bias from spherical decay.** A plane-wave basis has
identical amplitude at the inner and outer shells, but the real point-source
field has $|p_{\text{outer}}|/|p_{\text{inner}}| \approx d_{\text{inner}} / d_{\text{outer}}$
(roughly $1.33$ at the front of the shell for an external source at
$r_{\text{src}} = 0.6$ m). To fit a $33\%$ amplitude mismatch with two
basis functions of equal magnitude, the LS introduces non-physical in/out
combinations — directly leaking energy across the split. In 2D the analog
field has $1/\sqrt{r}$ decay (Hankel-function asymptotics for cylindrical
waves), so the mismatch is gentler and the method works.

**Empirical result on real reverb_hom data.** At the peak of the direct
arrival on the source-facing shell point (where $p_{\rm in}/p$ should be
$\approx 1$): legacy gives $0.977$, SHT gives a similar value, local_pw
gives $-0.058$ — wrong sign and tiny magnitude. At the back point (where
the ratio should be $\approx 0$): legacy $0.009$, local_pw $0.65$.
Frequency-domain comparison shows local_pw producing a roughly
half-magnitude "incoming" with no clear advantage over legacy.

**When local_pw might still help.** The method works correctly on a
clean synthetic plane wave (verified — ratio $\approx 0.80$ at the
inward pole, $\approx 0$ at the outward pole, with the regularisation
explaining the $0.80$ instead of $1.00$). So it can be useful when the
incident field is locally near-plane-wave, which is true for sources at
many wavelengths away from the shell. The current setup has
$r_{\rm src}/r_{\rm outer} = 2$, which is inside the near-field. Pushing
illumination sources to $r_{\rm src} \gtrsim 5 r_{\rm outer}$ would
flatten the wavefronts enough that the plane-wave basis would fit cleanly,
and the spectral comb would be the only remaining issue — controllable by
choosing $\Delta r$ to keep all comb teeth outside the source bandwidth.

**Bottom line for the current paper geometry.** SHT is a clean
single-shell method but bandwidth-limited to ~12 kHz by the 300-point
angular Nyquist; local_pw avoids that limit but suffers from the
two-shell comb plus the near-field magnitude mismatch and underperforms
legacy on the actual reverb data. Legacy remains the most defensible
default for the 18 kHz Ricker on this geometry; SHT is worth running as
a low-band cross-check; local_pw is most useful as a future option for
geometries with more distant illumination or a smaller $\Delta r$.
