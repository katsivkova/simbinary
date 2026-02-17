## Introduction

A code to simulate Gaia epoch astrometry for binary objects with a particular interest in Variable Induced Mover (VIM) sources. `simbinary` consists in two modules `SimBinary`, the one simulating data, and `fitGaia`, a short cut to fit data with [kepmodel](https://gitlab.unige.ch/delisle/kepmodel/-/tree/main?ref_type=heads). 

## Install

```
pip3 install git+https://github.com/katsivkova/simbinary 
```

To get editable clone repository:

```
git clone https://github.com/katsivkova/simbinary 
cd simbinary
pip install -e .
```