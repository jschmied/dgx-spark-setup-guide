Confirmed on the updated diff: two files, no trailing whitespace, nothing over 88 chars, and the
fix itself is unchanged — `decoder_layer_cls` restored on the class and at the call site. Thanks
for splitting it out.

Worth a maintainer's attention over the usual queue: DFlash2 checkpoints are unloadable on `main`
until this lands, so anyone building from source rather than a release is blocked, and it also
blocks re-measuring DFlash2 performance against current main.
