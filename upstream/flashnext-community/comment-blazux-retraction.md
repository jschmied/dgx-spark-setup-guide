**Retracting this: the fault was my checkpoint, not your recipe or my driver.**

Two of my 206 safetensors files are corrupt — correct size, wrong content:

```
model-bf16-00011.safetensors    want 
                                got  (sha256 mismatch)
model-plefp8-00000.safetensors  want dc2e845b7edd35bda92834fba3626bf7d199e28d6aceac99fee654aade390cfd
                                got  011c57ca547ad8b1b41732cce84054432146d3205f43a36353561294e7f06db6
```

They are exactly the two files still being written when my download stalled. I compared **sizes**
against the HF API, saw 418/418 match, deleted aria2's `.aria2` control files, and called it
verified. HuggingFace publishes `lfs.sha256` in the same API response I was already parsing, and
I did not use it.

One of them is dense BF16 body weights and the other is PLE shards 0-12, which is why the garbage
was invariant to every configuration I tried, including yours. My one content check happened to
sample row 0 of shard 0 — inside the intact head of the corrupt file — so it matched the official
table at cosine 0.999635 and I concluded the checkpoint was sound.

So: your recipe is fine, your hook is fine, and the driver/firmware question I asked you was
noise. Sorry for the detour. Re-fetching the two files and I will confirm here once it serves.

For anyone who lands on this later, the useful part: **verify `lfs.sha256`, not file size.** A
size-correct, byte-corrupt shard loads cleanly, reports sane tensor shapes, produces
correct-magnitude activations, and yields fluent token salad that survives every config change
you can think of.
