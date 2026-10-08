# Data

The data comes from the Sorbonne Data Challenge (Université Paris 1 Panthéon-Sorbonne × ComCyber, French Ministry of the Interior) and is not redistributed here. Participants received it from the organisers.

Expected layout:

```
data/
├── digraphs/                     # one control-flow graph per binary: <sha256>.json
└── training_set_metadata.csv     # one row per binary, one 0/1 column per behaviour
```

## Control-flow graphs

About 20,000 Windows binaries (PE executables and DLLs). For each one, the organisers extracted the control-flow graph in Graphviz *digraph* syntax (despite the `.json` extension):

```
Digraph G {
"180001010" [label = "180001010 : CALL : push rsi"]
"180001010" -> "180001043"
"180001043" [label = "180001043 : CALL : movzx eax, al"]
...
}
```

Each node is a block of x86 / x86-64 instructions identified by its address, labelled with its type (`INST`, `CALL`, `JCC`, `RET`, …) and the assembly instruction; edges are jumps, calls and fall-throughs.

## Labels

`training_set_metadata.csv` (`;`-separated) has 23,102 rows and 454 columns: `name` (SHA-256 of the binary) and 453 binary columns, one per behaviour observed in a sandbox (e.g. *allocate RWX memory*, *write file on Windows*, *act as TCP client*, packer names such as *aspack*). A binary can exhibit several behaviours: this is a multi-label problem.

## Generated files

The notebook writes intermediate files that are ignored by git:

- `your_data_updated.csv`: metadata restricted to the graphs that are present;
- `npz_matrices/`: TF matrices per batch, the full TF-IDF matrix and its SVD-reduced version.
