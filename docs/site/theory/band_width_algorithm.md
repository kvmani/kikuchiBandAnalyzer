# Band-Width Algorithm

The detector estimates band width by reducing a two-dimensional Kikuchi pattern
to one-dimensional intensity profiles across candidate bands.

## Line Model

A candidate central line can be represented in image coordinates as

$$
\ell(t) = p_0 + t \hat{u},
$$

where $p_0$ is a point on the line, $\hat{u}$ is the unit direction along the
band, and $t$ is the along-band coordinate. The perpendicular sampling direction
is

$$
\hat{n} = (-u_y, u_x).
$$

## Profile Sampling

For each offset $s$, the profile intensity is estimated by averaging samples
parallel to the central line:

$$
I(s) = \frac{1}{N}\sum_{i=1}^{N} P(p_i + s\hat{n}),
$$

where $P(x, y)$ is the pattern intensity. Interpolation is used when sample
locations are not integer pixels.

## Edge Selection

The measured width is the distance between two selected shoulders:

$$
w = |s_2 - s_1|.
$$

The implementation should record the selected edge indices, the profile values,
and quality warnings. Ambiguous shoulders, flat profiles, and out-of-bounds
sampling are not silent failures; they are diagnostic states.

## Quality Metadata

Useful quality fields include:

- peak or shoulder contrast;
- number of valid samples;
- profile smoothness;
- edge confidence;
- whether fallback logic was used.
