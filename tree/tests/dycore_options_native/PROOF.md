# Direct NVRTC standalone proof

35918 comparisons, 8795586 words, 0 differences, no exclusions. 3584 scientific branch evaluations. One serial native process and one Hex process, no MPI. Direct NVRTC --fmad=false --ftz=false --prec-div=true --prec-sqrt=true. Whole-model forecast proof and G0-G2 remain mandatory and unmeasured.

| Receipt | Comparisons | Words | SHA256 |
|---|---:|---:|---|
| direct-synthetic/module-fixed-replay.json | 2566 | 132336 | ab3ac323c51bd5fdd61ddbd71d5bfa242f1122623ce824a2483387d68b17f405 |
| direct-synthetic/module-zero-replay.json | 2566 | 132336 | 08f4da43ffe048cbc9fc7780b55b6ca91fb8da970a8541ea1520754831071604 |
| direct-synthetic/module-levels55-replay.json | 2566 | 903794 | 829251b5a3929a56409d01b729eb554296990e7a50d21193be3d347885dbc8dd |
| direct-synthetic/module-zero-seed-replay.json | 5126 | 264432 | 6d20d7ae71a64ba59ab543ea5522ff2ef3514cd92a7d34729181cdacee494cea |
| module-subnormal-replay.json | 2566 | 132336 | ac7a513187381c515d2210be08c87c00365b25a203d1d99db8a05dbe27653671 |
| real-small-2023-03-31/module-real-dry-replay.json | 2566 | 903794 | 0e014ce5afe08708eda849ff1a0133503867d2d11acdf9ebd59de93a9977a4f7 |
| real-small-2023-03-31/module-real-moist-replay.json | 2566 | 903794 | 28720043c2fa857e17812ac2baba36307a5e4101ec5eb7bc664967d76a96e726 |
| real-small-2022-12-22/module-real-dry-replay.json | 2566 | 903794 | 33654d3ebd7ec297737c02471edcc7eb697214f4991c1cd711db95a67e75c41d |
| real-small-2022-12-22/module-real-moist-replay.json | 2566 | 903794 | 87d090f2bdc40995b613473f08f79810aa7441b8a7fb24c54010fc6fa8e04d04 |
| real-medium-2023-03-31/module-real-dry-replay.json | 2566 | 903794 | 263cfeaacf72c3e31d786023ce45355ac74c6f09c30a389d41019db831b8d5da |
| real-medium-2023-03-31/module-real-moist-replay.json | 2566 | 903794 | f71d0bbd613e7fbfde428aba78fc1840c863372290b8982fd2d296d63cb8010b |
| real-medium-2022-12-22/module-real-dry-replay.json | 2566 | 903794 | 93cc72d9532bba7c9f0355f7d776c684a481d36b132258b5661cf7ec50caaeb7 |
| real-medium-2022-12-22/module-real-moist-replay.json | 2566 | 903794 | e047f34c81d8dcd4f9b326155912a4ef5284b7f8594d1df57d692ef4399786a7 |

Subnormal native multiply/upper sentinel: 16 words, 0 differences; forced-FTZ counterfactual differs in 8 multiply words. Receipt SHA256 f453d2a7582933102d4e3f550e6b9db09d3f480facd54f1d9f9b3133d8ca3ccf.

Earlier CuPy receipts retain historical diagnostic scope. Accepted input publication hashes are rechecked when sealing. Real columns use synthetic tetrahedral topology; this does not certify a forecast.
