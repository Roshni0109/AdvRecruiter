# data/ — Place your data files here

This folder is intentionally excluded from Git (see `.gitignore`) because the files are too large to store on GitHub.

## Files you need to place here manually:

| File | Size | Where to get it |
|---|---|---|
| `candidates.jsonl` | ~465 MB | From the Redrob hackathon bundle |

## How to set up:

Copy the `candidates.jsonl` file from your hackathon bundle into this folder:

```
AdvRecruiter/
└── data/
    └── candidates.jsonl   ← place it here
```

The scripts will look for the file at `./data/candidates.jsonl` by default.
