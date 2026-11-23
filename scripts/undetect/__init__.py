"""UnDetect: the transcriptional response to a compound that was never profiled, predicted in two parts.

1. Context (backbone.py): a network predicts, from a sample's paired control profile alone, the response that any
   compound would show in that sample.
2. Compound effect (kernels.py, model.py): kernel ridge regression over a multi-view compound kernel (structure,
   targets, mechanism of action, Cell Painting) carries the residual responses of profiled compounds to unprofiled
   ones, shared across cell lines and per line; a gate on the control's principal components rescales it per gene.

model.fit_fold fits and evaluates one fold; metrics.pcc_deg is the evaluation metric.
"""
