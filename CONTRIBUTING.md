# Contributing to Lite Classify

Thank you for helping improve Lite Classify. Bug reports, questions, documentation fixes and
code are all welcome.

## Reporting a problem

Open an issue and include:

- the edition (CPU or GPU) and version, from **About** in the app;
- what you did, what you expected and what happened;
- the console text at the bottom of the window (it logs every step, including errors);
- the image size and pixel size, if the problem involves loading or segmentation.

## Contributing code or documentation

1. Read the [Contributor License Agreement](CLA.md). Every pull request needs the signed
   statement from the CLA in its description, or it cannot be merged.
2. Set up the development environment as described in the [README](README.md#running-from-source).
3. Keep changes focused: one fix or feature per pull request.
4. Match the existing style. `obia_engine.py` stays free of Qt; the user interface lives in
   `obia_app.py` and `obia_ui.py`.
5. Before opening the pull request, run the app through the workflow you changed. If the
   change affects segmentation, features or classification, also run the headless self-test:

       python liteclassify.py --selftest path\to\ortho.tif [path\to\dsm.tif]

   It writes its results to `%TEMP%\liteclassify_selftest.json`.

Substantial contributors are listed in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Modified versions

If you distribute a modified version, the GPL and the additional terms in [NOTICE.md](NOTICE.md)
apply: keep the credit line, mark it as modified, and give it a different name.
