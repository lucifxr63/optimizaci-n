# LKH 3.0 Configuration

This folder contains an example of how to run [LKH 3.0](http://akira.ruc.dk/~keld/research/LKH/) for the BMTSP.

1. `example.tsp` is the TSPLIB file with node coordinates. Node `1` is
   considered the depot.
2. `example.par` is a parameter file for LKH. Important parameters:
   - `SALESMEN` sets the number of salesmen.
   - `PROBLEM_FILE` and `OUTPUT_TOUR_FILE` specify input and output.

To run LKH (assuming it is installed and in your `PATH`):

```bash
LKH example.par
```

The solver will output a file `example.tour` containing the best found tour.
The first number is the cost. Each subsequent line lists the sequence of
nodes to visit for all salesmen, separated by `-1` delimiters (one tour for
each salesman).
