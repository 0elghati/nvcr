# Changelog

## [0.11.0](https://github.com/0elghati/nvcr/compare/v0.10.0...v0.11.0) (2026-08-07)


### Features

* **experiments:** Run on RTX 5060 Laptop ([#78](https://github.com/0elghati/nvcr/issues/78)) ([5aaeeba](https://github.com/0elghati/nvcr/commit/5aaeeba0c6722894f225dc02f186f9c2826deaf4))
* **experiments:** run on RTX3050 laptop ([#80](https://github.com/0elghati/nvcr/issues/80)) ([0398e3f](https://github.com/0elghati/nvcr/commit/0398e3f901c3a0a9768220dd8042c19aaef53bb7))

## [0.10.0](https://github.com/0elghati/nvcr/compare/v0.9.0...v0.10.0) (2026-08-07)


### Features

* measurement pipeline & rtx4070 measurement ([#76](https://github.com/0elghati/nvcr/issues/76)) ([aa16f20](https://github.com/0elghati/nvcr/commit/aa16f20f67cb48636aa5f004925696d69dd7bc9c))

## [0.9.0](https://github.com/0elghati/nvcr/compare/v0.8.2...v0.9.0) (2026-08-07)


### Features

* added RTX3050 profile ([#74](https://github.com/0elghati/nvcr/issues/74)) ([975a150](https://github.com/0elghati/nvcr/commit/975a150ba2fa55e8d41ffbe77eaefbafb1d562e1))
* artifact resolver provenance ([#68](https://github.com/0elghati/nvcr/issues/68)) ([4c0cf69](https://github.com/0elghati/nvcr/commit/4c0cf69ee6ccce621e11d36dcffeeb8110b013c9))
* softwarex evidence automation ([#73](https://github.com/0elghati/nvcr/issues/73)) ([fb80dca](https://github.com/0elghati/nvcr/commit/fb80dca933491de794c23bb93970908a062bcf61))


### Bug Fixes

* **artifacts:** separate engine profiles from components ([#70](https://github.com/0elghati/nvcr/issues/70)) ([cb582fe](https://github.com/0elghati/nvcr/commit/cb582fee21becc44d097a22dd7e0c741b60f87f1))

## [0.8.2](https://github.com/0elghati/nvcr/compare/v0.8.1...v0.8.2) (2026-08-05)


### Bug Fixes

* **docker:** Build amd64 CUDA 12.8 image for compiler GPU set ([#64](https://github.com/0elghati/nvcr/issues/64)) ([09ed9ac](https://github.com/0elghati/nvcr/commit/09ed9ac603a7daa72439d6ffadf6deacab230e18))

## [0.8.1](https://github.com/0elghati/nvcr/compare/v0.8.0...v0.8.1) (2026-08-05)


### Bug Fixes

* **installer:** relax desktop cuda engine catalog ([#62](https://github.com/0elghati/nvcr/issues/62)) ([d504cc1](https://github.com/0elghati/nvcr/commit/d504cc12204c9e23f001e01f45ea14d7a9e8254a))
* Relax desktop CUDA engine runtime matching ([#60](https://github.com/0elghati/nvcr/issues/60)) ([19586b3](https://github.com/0elghati/nvcr/commit/19586b3d5f00d6ceae3ccf9e0b575731bccc1c55))

## [0.8.0](https://github.com/0elghati/nvcr/compare/v0.7.0...v0.8.0) (2026-08-05)


### Features

* **artifacts:** add bounded TensorRT engine portability ([#55](https://github.com/0elghati/nvcr/issues/55)) ([038adc7](https://github.com/0elghati/nvcr/commit/038adc79a752c2f08286f134c5fefdeb61e5d6bb))
* **targets:** validate RTX 5060 Blackwell path ([#56](https://github.com/0elghati/nvcr/issues/56)) ([0ac426f](https://github.com/0elghati/nvcr/commit/0ac426fe5ecb213d33b7005abb31e3784bb74224))


### Performance Improvements

* record Orin sequence-GOP diagnostic ([#59](https://github.com/0elghati/nvcr/issues/59)) ([9239719](https://github.com/0elghati/nvcr/commit/9239719208d2e6679fde29a72f271b99a8411cbb))

## [0.7.0](https://github.com/0elghati/nvcr/compare/v0.6.0...v0.7.0) (2026-08-04)


### Features

* **artifacts:** add rolling engine catalog installer ([#49](https://github.com/0elghati/nvcr/issues/49)) ([e17f78e](https://github.com/0elghati/nvcr/commit/e17f78e67b02ffd587846c8b31c7964ddba2be67))
* **x86:** Rolling engine assets ([#51](https://github.com/0elghati/nvcr/issues/51)) ([446a6d5](https://github.com/0elghati/nvcr/commit/446a6d511488ddf33f6ef1393479531e91a796f9))

## [0.6.0](https://github.com/0elghati/nvcr/compare/v0.5.1...v0.6.0) (2026-08-03)


### Features

* add architecture-aware container runtime ([#47](https://github.com/0elghati/nvcr/issues/47)) ([8ccb6e3](https://github.com/0elghati/nvcr/commit/8ccb6e3c1709a33e77fcdd08beb41a1d200b451d))

## [0.5.1](https://github.com/0elghati/nvcr/compare/v0.5.0...v0.5.1) (2026-08-02)


### Performance Improvements

* **dcvcrt:** optimize Orin execution and validate fixed edge profiles ([#45](https://github.com/0elghati/nvcr/issues/45)) ([06397b3](https://github.com/0elghati/nvcr/commit/06397b337cdb4a88dc875ce6544408943610691f))

## [0.5.0](https://github.com/0elghati/nvcr/compare/v0.4.1...v0.5.0) (2026-07-30)


### Features

* **bench:** add warmed repeated resolution runs ([#35](https://github.com/0elghati/nvcr/issues/35)) ([e9f8a1e](https://github.com/0elghati/nvcr/commit/e9f8a1e55a11f1551b13ee02f9f26f3e5e82b8b5))
* **dcvcrt:** add four-resolution TensorRT profiles ([#42](https://github.com/0elghati/nvcr/issues/42)) ([200f31c](https://github.com/0elghati/nvcr/commit/200f31cc78a686cf2b0df0d37d8a1efd3cc9c95a))
* enhance dcvcrt device i decode performance ([#32](https://github.com/0elghati/nvcr/issues/32)) ([8b9f613](https://github.com/0elghati/nvcr/commit/8b9f613e6892e16b8e14b1e7c1866e39bb017e8c))


### Bug Fixes

* **dcvcrt:** preserve direct YUV reconstruction ([#37](https://github.com/0elghati/nvcr/issues/37)) ([d9975f4](https://github.com/0elghati/nvcr/commit/d9975f4e360160114b0046b2c564c1c941e32433))


### Performance Improvements

* **dcvcrt:** optimize I-frame decode staging ([#34](https://github.com/0elghati/nvcr/issues/34)) ([023f951](https://github.com/0elghati/nvcr/commit/023f951d39f6b8002e48ddaba3b9a4e0c675e542))

## [0.4.1](https://github.com/0elghati/nvcr/compare/v0.4.0...v0.4.1) (2026-07-29)


### Bug Fixes

* public install repo ([#29](https://github.com/0elghati/nvcr/issues/29)) ([83bb4f5](https://github.com/0elghati/nvcr/commit/83bb4f5f055e16718d7f4d59fd3d2fa104aedd7b))

## [0.4.0](https://github.com/0elghati/nvcr/compare/v0.3.0...v0.4.0) (2026-07-29)


### Features

* restructure dcvcrt backend & added support for 720p build ([#26](https://github.com/0elghati/nvcr/issues/26)) ([1361742](https://github.com/0elghati/nvcr/commit/1361742dc9789de0f0cb871493261f4bd7ba44fc))

## [0.3.0](https://github.com/0elghati/nvcr/compare/v0.2.3...v0.3.0) (2026-07-23)


### Features

* Project restructure ([#22](https://github.com/0elghati/nvcr/issues/22)) ([2417a79](https://github.com/0elghati/nvcr/commit/2417a79bf76ff718853d2919e0f295b828b74754))
* **runtime:** add scoped artifact validation and access-unit workflow ([#18](https://github.com/0elghati/nvcr/issues/18)) ([e6b8666](https://github.com/0elghati/nvcr/commit/e6b8666e3b02ea629fc5274b7a2b828d2d2d261e))

## [0.2.3](https://github.com/0elghati/nvrc/compare/v0.2.2...v0.2.3) (2026-07-22)


### Bug Fixes

* **ci:** pin transitive TensorRT apt dependencies ([#16](https://github.com/0elghati/nvrc/issues/16)) ([16bda86](https://github.com/0elghati/nvrc/commit/16bda867bed1e6696c428adb333fad588a6d1f84))

## [0.2.2](https://github.com/0elghati/nvrc/compare/v0.2.1...v0.2.2) (2026-07-22)


### Bug Fixes

* **ci:** release binary runtime deps ([#14](https://github.com/0elghati/nvrc/issues/14)) ([c3eec5c](https://github.com/0elghati/nvrc/commit/c3eec5c159f00fb697782e957d9379f7d77790c5))

## [0.2.1](https://github.com/0elghati/nvrc/compare/v0.2.0...v0.2.1) (2026-07-22)


### Bug Fixes

* **ci:** Pass repository path to fix release please issue ([#12](https://github.com/0elghati/nvrc/issues/12)) ([9c6d135](https://github.com/0elghati/nvrc/commit/9c6d13522d7949b86d158c02f68899d77e944910))

## [0.2.0](https://github.com/0elghati/nvrc/compare/v0.1.0...v0.2.0) (2026-07-22)


### Features

* document portable binary platform in quick start ([#9](https://github.com/0elghati/nvrc/issues/9)) ([ab1297a](https://github.com/0elghati/nvrc/commit/ab1297a410596797db28ac13ce5bda0826583a30))


### Bug Fixes

* add GitHub About section description ([#7](https://github.com/0elghati/nvrc/issues/7)) ([0e7759a](https://github.com/0elghati/nvrc/commit/0e7759abe60568b7050ecc12548b2193449373b3))
* **ci:** release please fix ([#11](https://github.com/0elghati/nvrc/issues/11)) ([86b0125](https://github.com/0elghati/nvrc/commit/86b01253ca6a28e38165cbcfd54a273496322711))

## [0.1.0](https://github.com/0elghati/nvrc/compare/v0.1.0...v0.1.0) (2026-07-22)


### Bug Fixes

* add GitHub About section description ([#7](https://github.com/0elghati/nvrc/issues/7)) ([0e7759a](https://github.com/0elghati/nvrc/commit/0e7759abe60568b7050ecc12548b2193449373b3))

## [0.1.0](https://github.com/0elghati/nvrc/compare/v0.1.0...v0.1.0) (2026-07-22)


### Features

* **build:** auto-detect CUDA/TensorRT toolchain and add portable multi-arch option ([f3d2984](https://github.com/0elghati/nvrc/commit/f3d2984799beeecd1cfc78fd619355fde38985ea))
* **dcvcrt:** implement TensorRT P-frame backend with device-stamped engine manifests ([02efea6](https://github.com/0elghati/nvrc/commit/02efea67e0a47fcae310b92f94da2cbf19ddf45a))
* **scripts:** add one-command install and platform auto-detection helpers ([d0f8b20](https://github.com/0elghati/nvrc/commit/d0f8b20fbc54e2f1bf03b530553311330f63d7e3))
* **scripts:** auto-tune DCVC-RT artifact pipeline and remove hardcoded paths ([1ed3ce2](https://github.com/0elghati/nvrc/commit/1ed3ce2338a3f7c1c114484de4422f7211a729f5))


### Bug Fixes

* bootstrap initial release-please PR ([#3](https://github.com/0elghati/nvrc/issues/3)) ([f99491e](https://github.com/0elghati/nvrc/commit/f99491e06760f24be765e53c6635bba5223b4e5e))

## Changelog

All notable changes to this project will be documented in this file.
