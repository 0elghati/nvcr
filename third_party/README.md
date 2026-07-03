# Third-party integrations

NVCR does not vendor dependencies by default.

The original DCVC-RT native C++ entropy coder must be integrated here (or supplied
as an external CMake target) behind `nvcr::dcvcrt::EntropyCoderAdapter`. Do not
copy, translate, or reimplement its algorithms. Preserve its upstream license and
history, ideally with a pinned Git submodule or FetchContent declaration once the
authoritative upstream revision is selected.

