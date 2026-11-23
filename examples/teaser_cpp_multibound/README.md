# Rigid registration with several noise bounds

This example passes one or more noise bounds to `RobustRegistrationSolver::solve` with the two input clouds and their correspondences. The output gives the selected source-to-target 4 × 4 transform and its noise bound.

With several bounds, the scorer uses the matched points directly. To verify on
separate clouds, pass `--verify` and two PLY paths. These clouds can be denser
than the matched inputs and need no correspondences. Each verification cloud
is voxel-downsampled with a cell width of twice its own estimated median
nearest-neighbor spacing. The scorer compares candidates by averaging both
directions of nearest-neighbor agreement, normalized by each receiving cloud's
spacing.

Build and install the TEASER++ C++ library from this checkout first.

From the repository root:

```sh
cmake -S examples/teaser_cpp_multibound -B build-multibound -DCMAKE_BUILD_TYPE=Release
cmake --build build-multibound
./build-multibound/teaser_cpp_multibound source.ply target.ply matches.txt 0.01 0.02
./build-multibound/teaser_cpp_multibound source.ply target.ply matches.txt auto
./build-multibound/teaser_cpp_multibound source.ply target.ply matches.txt 0.01 0.02 \
  --verify dense_source.ply dense_target.ply
```

Set `CMAKE_PREFIX_PATH` to the TEASER++ install prefix when using a custom installation directory. Each row of `matches.txt` contains a zero-based source point index and its matching target point index, separated by whitespace. After `matches.txt`, list one or more solver noise bounds in the units of your input clouds. The example above tries `0.01` and `0.02`.

With one numeric bound, the example solves once and skips verification. With
`auto`, it tries `s`, `2s`, and `4s`, where `s` is the median
nearest-neighbor spacing of the matched target points. Separate verification
clouds only change how the candidates are ranked.

To verify the example runs with the data in this checkout:

```sh
cloud=examples/example_data/bun_zipper_res3.ply
matches=$(mktemp)
awk 'BEGIN { for (i = 0; i < 64; ++i) print i, i }' > "$matches"
./build-multibound/teaser_cpp_multibound "$cloud" "$cloud" "$matches" 0.001 0.002
rm "$matches"
```
