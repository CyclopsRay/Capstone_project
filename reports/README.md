# reports

- `undetect-demo.qmd`: the report of the demo run. Every number, table and
  figure in it is computed from `results/demo/` when it renders.
- `references.bib`: the works it cites.

`pixi run reproduce` renders it as the last step; `pixi run report` renders it
again from existing results. The rendered `undetect-demo.html` is not committed.
