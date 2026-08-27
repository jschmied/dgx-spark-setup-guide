**Correction, and I owe you one for the controls you ran: the fault was my checkpoint.**

Two of my 206 files are corrupt — correct size, wrong sha256: `model-bf16-00011.safetensors`
(dense BF16 body) and `model-plefp8-00000.safetensors` (PLE shards 0-12). They are the two files
still being written when my download stalled; I verified **sizes** against the HF API, deleted
aria2's control files, and called it complete. `lfs.sha256` was in the same API response and I
did not check it.

That means the `*.self_attn.*` exclusion gap I asked you to test is **not** implicated by
anything I reported, and you should not spend a run on it on my account. My "it's the body"
reasoning, my BF16-buffer experiment and the twenty eliminations were all downstream of corrupt
weights. The one content check I did — PLE row 0 against the official BF16 table, cosine
0.999635 — happened to land inside the intact head of the corrupt file.

Your work stands on its own regardless: the `group_size` elimination is real, and your parameter
accounting for the 3.3 GiB gap (RadixArk leaving ~2.52 B more parameters in BF16, i.e. not
quantizing the QSA projections that Inferact does) is a genuine and precisely measured difference
between the two exports. It just is not evidenced by my failure.

Re-fetching the two files now; I will report whether RadixArk then serves coherently with the
one-gate patch, since that is the part of my report that still matters — and if it does, your
checkpoint table line becomes "loads and works with a one-line change", not "loads and emits
garbage".

Thank you for the reference logits and for offering the controls. The lesson for the thread:
verify `lfs.sha256`, not file size — a size-correct corrupt shard loads cleanly, reports sane
shapes, produces correct-magnitude activations, and yields garbage that survives every
configuration change.
