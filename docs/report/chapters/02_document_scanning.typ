#import "../template/lib.typ": *

= Document Scanning

== Literature Review

Document dewarping has evolved from geometric heuristics to learning-based methods that infer
pixel-wise warps. Early learning pipelines such as DocUNet frame the task as image-to-image
translation, predicting a flattened page appearance without explicit geometric constraints
@Ma_2018_CVPR. DewarpNet introduced synthetic supervision via the Doc3D dataset and explicitly
predicted document shapes to improve geometric consistency @Das_2019_ICCV. More recently, UVDoc
predicts a coarse 2D sampling grid alongside a 3D surface grid, enabling surface-aware reasoning
and higher-quality unwarping @Verhoeven:UVDoc:2023.

These approaches suggest two consistent themes: (i) dense sampling grids are robust for
photometric fidelity, and (ii) surface geometry is an informative prior when the page is
strongly curved. The project thus explores how to exploit UVDoc's 3D grid to stabilize and
regularize the sampling process while remaining compatible with grid-based unwarping.

== 3D-Aware Deskewing Concept

Let the UVDoc model predict a coarse grid of 2D sampling coordinates
$bold(u)_(i j) = (u_(i j), v_(i j))$ and 3D surface points
$bold(p)_(i j) = (x_(i j), y_(i j), z_(i j))$ on a regular mesh. Standard UVDoc inference upsamples
$bold(u)_(i j)$ to a dense grid and uses `grid_sample` to obtain the deskewed image. This
procedure is photometrically faithful but may allocate pixels unevenly over the surface when
the page is strongly curved.

To improve sampling regularity, a surface-aware parameterization is computed from the 3D mesh.
Arc-lengths along the mesh rows and columns provide monotonic coordinates $(s_j, t_i)$ that
approximate geodesic distances. The 2D grid $bold(u)_(i j)$ is then resampled onto a uniform
$(s, t)$ lattice, yielding a dense grid that is closer to uniform surface sampling. The final
deskewed image is produced by applying this reparameterized grid to the original input.

== Implementation Notes (Traenslenzor)

The implementation adds an optional `uv3d` deskew mode that:

- computes 1D arc-length profiles from UVDoc's 3D grid,
- enforces monotonicity and applies mild smoothing,
- inverts the profiles to build a dense index grid,
- resamples the predicted 2D grid via `grid_sample`,
- falls back to the standard `uv2d` path if the 3D parameterization is ill-conditioned.

This preserves the original UVDoc sampling mechanism while exploiting 3D cues to regularize
the sampling density, yielding improved stability on curved pages.

== Discussion and Limitations

The arc-length parameterization is separable and therefore only approximates true geodesic
flattening. It assumes row/column monotonicity and may underperform on extreme folds or
self-occlusions. The 3D grid is also scale-ambiguous, so only relative distances are used.
Future work could replace the separable parameterization with a full mesh-flattening method or
add confidence-guided blending between the 2D and 3D-derived grids.
