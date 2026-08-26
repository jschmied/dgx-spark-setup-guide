Closing in favour of **#52527**, which now carries this commit (`df6372608`, authorship intact)
alongside the runtime metric — agreed with @Sahil170595 there.

The log line and the metric answer the same question from two sides: "will I lose reuse" at
startup, "did I" at runtime. One PR is the right place for both. If reviewers would rather land
the startup line separately, the cherry-pick can be split back out and this reopened.

Root cause and the GB10 measurement stay on record in #52527 and #53595.
