# Methods

## Equation

On the unit domain in one, two, or three dimensions, temperature $T$ follows

$$\frac{\partial T}{\partial t} = \nabla \cdot \big(\alpha(\mathbf{x}) \nabla T\big).$$

## Spatial discretization

Cells are centered at $x_i = (i + \tfrac12)h$ with $h = 1/n$. This is a finite-volume form: the change in a cell equals the net flux through its faces, so whatever leaves one cell enters its neighbor. Energy conservation therefore holds at the discrete level, up to round-off.

Flux through a face uses the harmonic mean of the two cell diffusivities:

$$\alpha_{i+1/2} = \frac{2\alpha_i\alpha_{i+1}}{\alpha_i + \alpha_{i+1}}.$$

The harmonic mean is what you get by requiring the flux to be continuous across an interface between two materials in series. An arithmetic mean would let heat through a thin insulator far too easily. For a layered slab with the interface on a cell face, this form reproduces the exact piecewise-linear steady state, which is why the composite slab test reaches round-off error rather than $O(h^2)$.

Boundaries are imposed with one layer of ghost cells:

| Boundary | Ghost value | Meaning |
|---|---|---|
| Insulated | $T_{\text{ghost}} = T_1$ | Zero flux through the wall |
| Fixed temperature $g$ | $T_{\text{ghost}} = 2g - T_1$ | The face value, averaged, equals $g$ |
| Periodic | Filled by MPI from the opposite side | `MPI_Cart_create` with periodic axes |

## Time integration

The solver uses the $\theta$ family:

$$(I - \theta\,\Delta t\,L)\,T^{k+1} = (I + (1-\theta)\,\Delta t\,L)\,T^k + \Delta t\,b.$$

| Scheme | $\theta$ | Order | Stability |
|---|---|---|---|
| Explicit (FTCS) | 0 | 1 in time | $\Delta t \le 1 / \big(2\alpha_{\max}\sum_d h_d^{-2}\big)$ |
| Crank-Nicolson | 1/2 | 2 | Unconditional, but stiff modes decay slowly |
| Backward Euler | 1 | 1 | Unconditional and strongly damping |

With $\Delta t$ tied to $h^2$ by the stability limit, the explicit scheme's first-order time error shrinks like $h^2$, so the combined error measured on grid refinement is second order.

**The explicit limit depends on the data's modes.** The limit comes from the highest-frequency checkerboard mode, whose amplification factor is $|1 - 2\,\text{CFL}|$ when every axis is excited. A field that varies along only one axis of a 2D grid has no such mode, and at CFL 1.5 it stays stable. The stability experiment therefore starts from a Gaussian, which has content on all axes. This appeared as a failing test during development.

## Implicit solve

$A = I - \theta\,\Delta t\,L$ is symmetric positive definite, so the system is solved with conjugate gradients and a Jacobi preconditioner. Each iteration needs one halo exchange and three `MPI_Allreduce` calls. Those global reductions are the classic scaling bottleneck of Krylov methods, and they are timed separately.

## Parallel decomposition

`MPI_Dims_create` and `MPI_Cart_create` build a Cartesian process grid. Each axis uses a block partition:

$$n_c = \lfloor n/P \rfloor + (c < n \bmod P), $$

which gives every cell to exactly one rank and differs by at most one cell between ranks. The original program used $\lfloor n/P \rfloor$ alone, so at $P = 3$ with $n = 1024$ it silently dropped a cell.

Faces are described by `MPI_Type_create_subarray`, so the same code handles every axis without packing buffers by hand.

### Halo exchange strategies

| Mode | Pattern | Cost model |
|---|---|---|
| `chain` | Each rank exchanges with its left neighbor, then its right | A dependency wave crosses the ranks, so latency grows with $P$ |
| `sendrecv` | All ranks shift one way, then the other | Two rounds, independent of $P$ |
| `overlap` | Post non-blocking transfers, update interior cells, wait, update the shell | Allows overlap when the MPI library makes progress in the background |

`chain` is the pattern from the original program, kept as a measurable baseline. It deadlocks on a periodic ring, because no rank is free to go first, so the solver refuses that combination.

### Parallel I/O

Snapshots are written with collective MPI-IO: `MPI_File_set_view` with a subarray file type, then `MPI_File_write_all`. Output is identical to a serial file in x-fastest order, independent of the rank count.

## Benchmark method

Following Hoefler and Belli (2015):

- Runs are executed in randomized order, so thermal drift does not line up with the rank count.
- Reported values are medians over repeats, with 10th and 90th percentiles kept.
- The timer takes the slowest rank's wall clock after a barrier.
- Hardware and its limits are stated with the results.

Speedups are fitted with the Universal Scalability Law (Gunther, 1993):

$$S(P) = \frac{P}{1 + \sigma(P-1) + \kappa P(P-1)},$$

where $\sigma$ captures contention and $\kappa$ captures coherence costs. The fit is descriptive only. The law assumes identical workers, which a hybrid laptop CPU does not provide.

## Neural surrogate

**Task.** Map the initial temperature and diffusivity to the temperature at $\tau = 0.02$ on the insulated unit square.

**Data.** Materials are smooth or near-binary fields from random cosine series, with $\alpha \in [0.05, 1]$. Initial conditions are 1 to 5 Gaussian blobs. Every target comes from the MPI solver at CFL 0.25, chosen so that the solver's own time error stays well below the error being measured. Samples are drawn from a resolution-independent specification, so one seed can be rendered at any grid size.

**Splits.** The test, validation, training, and shifted sets use disjoint seed ranges. The shifted set has a contrast of 100, finer material features, and narrower blobs, none of which appear in training.

**Models.**

- *FNO* (Li et al., 2020): a lifting layer, four Fourier layers with 12 modes and width 32, and a projection. Inputs are padded because the boundaries are not periodic. Implemented from the paper.
- *U-Net* (Ronneberger et al., 2015): three levels with a similar parameter count, as a local convolutional comparison.

Both predict the temperature *change*. On an insulated domain total energy is constant, so subtracting the mean of the predicted change enforces conservation exactly. This is the "conservative" variant.

**Baselines.** A surrogate is only interesting if it beats simple alternatives:

- *Persistence*: predict no change, which defines an error of 100% on the change metric.
- *Exact solve, averaged material*: an exact discrete cosine solution with the geometric-mean diffusivity.
- *POD-ridge*: linear regression between principal components.
- *Coarse solver*: the actual MPI solver on a coarser grid, followed by cubic upsampling. This is the serious competitor.

**Metrics.** The main metric is relative L2 error of the temperature *change*, because a model can score well on the full field by barely changing the input. Bootstrap intervals cover test-sample variation; standard deviations across training seeds are reported separately. Energy drift and maximum-principle violations measure physical correctness.

## Runtime model

Two models predict the time per step from 720 timed runs:

- *Analytic*: non-negative least squares on cells per rank, halo cells per rank, and neighbor count, weighted to minimize relative error. Each coefficient has a physical meaning.
- *Gradient boosting*: on log features, more flexible but less interpretable.

Both are validated by holding out an entire grid size or rank count, which tests extrapolation rather than interpolation.

## References

1. Crank, J., and Nicolson, P. (1947). A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type. *Mathematical Proceedings of the Cambridge Philosophical Society*. https://doi.org/10.1017/S0305004100023197
2. Gunther, N. J. (1993). A simple capacity model of massively parallel transaction systems. *International CMG Conference*.
3. Hoefler, T., and Belli, R. (2015). Scientific benchmarking of parallel computing systems. *SC15*. https://doi.org/10.1145/2807591.2807644
4. Li, Z., et al. (2020). Fourier Neural Operator for parametric partial differential equations. *arXiv*. https://doi.org/10.48550/arXiv.2010.08895
5. Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*. https://doi.org/10.1007/978-3-319-24574-4_28
