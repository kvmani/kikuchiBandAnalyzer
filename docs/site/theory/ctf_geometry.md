# CTF Scan Geometry

HKL `.ctf` files describe EBSD scan points in text columns. A rectangular scan
grid is inferred from coordinate values and acquisition order.

For a scan with `N_x` columns and `N_y` rows, each pixel has an index

$$
i = y N_x + x,
$$

where `x` and `y` are zero-based raster coordinates after normalization. The
pattern folder must provide one image for each `i`.

Euler angles are interpreted using the convention declared by the source data
and phase adapter. For the FCC Ni target workflow, the configured phase and HKL
families define which bands are simulated or matched before width measurement.

The reader must preserve the distinction between physical coordinates and raster
indices. Physical coordinates determine scan spacing; raster indices determine
array placement and pattern lookup.
