# Examples

The example targets are built when `NVCR_BUILD_EXAMPLES=ON`.

`nvcr_packet_example` demonstrates the packet envelope only:

```bash
./build/examples/nvcr_packet_example
```

Raw-video encoding and decoding are provided by the first-class `nvcr` executable,
not an example round trip. See [CLI usage](../docs/cli.md).
