# benchmark — per-operation latency at realistic scale

A latency probe over every public `Context` operation, on a uniform
random graph (average degree ~10, one giant component — the worst case
for neighborhood walks), at two scales: 20k concepts / 100k
associations and 200k / 1M. 30% of writes carry a source, 200 labels,
1000 sources — roughly the shape a real corpus takes.

## Run

Release mode or the numbers mean nothing:

```sh
cargo run --release --example benchmark
```

## Reading the numbers

- Treat timings as **relative** — between operations, and between
  commits on the same machine — never as absolutes.
- `ingest` prints throughput and per-association cost; the read
  operations print per-call cost at their natural call rates.
- `to_bytes` / `from_bytes` time the whole-image persistence round
  trip — `from_bytes` includes full validation and index rebuild,
  which is what every server boot and cache reload pays.
- For memory and disk sizing at a target corpus shape, pair this with
  `taguru estimate --associations 1_000_000`, which builds a context
  of that shape and measures it.
