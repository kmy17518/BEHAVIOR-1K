# Third-party notices

## MANUS Integrated SDK

The optional source-built `native/manus_bridge` sidecar links against a
user-installed official SDK and calls `CoreSdk_InitializeIntegrated`. It does
not require or redistribute the MANUS Core desktop application. Proprietary
MANUS headers, `libManusSDK_Integrated.so`, glove calibration files, and license
data are external inputs: they are not distributed or copied into this
repository. Use of the integration is subject to the MANUS Software License
Agreement supplied with the installed SDK.

Powered by MANUS.

This software contains source code provided by Manus Technology Group B.V.

## AnyDexRetarget

Parts of `dex_teleop/retargeting`, `dex_teleop/tracking/hts.py`, the hand
configuration YAML files, and the retargeting URDFs are adapted from
AnyDexRetarget.

MIT License

Copyright (c) 2025 Shiquan Qiu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## dex-retargeting

The optional DexPilot backend uses `dex-retargeting==0.4.6` at runtime. The
package is not vendored in this repository.

The MIT License (MIT)

Copyright (c) 2023 Yuzhe Qin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## NVIDIA IsaacTeleop and Isaac Sim references

The packaged Sharpa DexPilot configuration files are adapted from NVIDIA
IsaacTeleop. The VIVE/libsurvive coordinate conversion and provider boundary
were informed by NVIDIA's `isaacsim.xr.input_devices` extension. Adapted
configuration files retain their upstream copyright and SPDX headers.

Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.

SPDX-License-Identifier: Apache-2.0

## libsurvive / pysurvive

VIVE lighthouse support optionally imports the external `pysurvive` bindings
for libsurvive. Neither dependency is distributed by `dex_teleop`; users must
install and use them under their upstream licenses.
