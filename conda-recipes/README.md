# Conda recipes for TEASER++

This directory contains recipes for building conda packages
for TEASER++.

Ideally, someone will eventually [set up a conda-forge feedstock](https://conda-forge.org/docs/maintainer/adding_pkgs.html)
so that users will be able to install from [conda-forge](https://conda-forge.org/#about).

In the meantime, you can use these recipes to build the package for inclusion in privately maintained
conda repositories.

Currently, there is only a recipe for `teaserpp-python`, which provides a minimal python binding
(e.g. no header files, only the shared libraries needed at runtime).

To build this, run this command from this directory on each platform you want to support:

```
conda build teaserpp-python
```

This will build versions of the package for multiple python versions (currently 3.6 through 3.9).
You can build a restricted set of python versions by specifying the `--variants` flag:

```
conda build --variants '{"python" : ["3.7", "3.8"]}' teaserpp-python'
```

This recipe has been run on linux and macos. To run on Windows, someone will need to
add a `build.bat` file corresponding to `build.sh`, and it is possible that some other
tweaks may be required if there are dependency differences on Windows.


