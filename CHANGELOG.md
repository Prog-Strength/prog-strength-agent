# [0.8.0](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.7.0...v0.8.0) (2026-05-19)


### Features

* **build:** publish agent image to ECR and pull from it on deploy ([ee6b450](https://github.com/Prog-Strength/prog-strength-agent/commit/ee6b4501e95c80d639df3b6ece5a7b77300979e6))

# [0.7.0](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.6.0...v0.7.0) (2026-05-19)


### Features

* **telemetry:** publish token + routing counters to Prometheus ([15a9440](https://github.com/Prog-Strength/prog-strength-agent/commit/15a944042dda0829d7f84baf8f6b083f0e6a78f6))
* **telemetry:** publish tool-call counters and latency histogram ([e66099a](https://github.com/Prog-Strength/prog-strength-agent/commit/e66099aa3def681d204c1f2cebb2c42b5ada84db))

# [0.6.0](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.5.0...v0.6.0) (2026-05-18)


### Features

* **telemetry:** instrument the agent and stream events to the API ([2c54f1b](https://github.com/Prog-Strength/prog-strength-agent/commit/2c54f1b699ba8ee1c9fad5603e2f1273b6dd9cc8))

# [0.5.0](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.4.0...v0.5.0) (2026-05-17)


### Features

* tiered model routing (Haiku default, Sonnet on complex) ([c009faf](https://github.com/Prog-Strength/prog-strength-agent/commit/c009fafb53dceb7052be95e3ef3b10ba760f9582))

# [0.4.0](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.3.0...v0.4.0) (2026-05-17)


### Features

* **model:** default to Sonnet 4.6 instead of Opus 4.7 ([4c3e23d](https://github.com/Prog-Strength/prog-strength-agent/commit/4c3e23df9ec8a1ae234b091b5618850d034ac357))

# [0.3.0](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.2.1...v0.3.0) (2026-05-17)


### Features

* **auth:** Per-request MCP session with user JWT forwarding ([f914e30](https://github.com/Prog-Strength/prog-strength-agent/commit/f914e30cd5ec664aeaf997ffcef89ebdc766f679))

## [0.2.1](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.2.0...v0.2.1) (2026-05-17)


### Bug Fixes

* **chat:** strip output-only fields from assistant content for replay ([6f3089f](https://github.com/Prog-Strength/prog-strength-agent/commit/6f3089fa65582a96b05951a30f491291bd0b99eb))

# [0.2.0](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.1.0...v0.2.0) (2026-05-17)


### Features

* **chat:** SSE streaming with CORS for browser clients ([700aab7](https://github.com/Prog-Strength/prog-strength-agent/commit/700aab7535f084d60a1c411489db1bc12e200609))

# [0.1.0](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.0.0...v0.1.0) (2026-05-16)


### Features

* **cicd:** Add release and deploy workflow ([a1d42aa](https://github.com/Prog-Strength/prog-strength-agent/commit/a1d42aa021e3efbf056866fc346a6bf1a3f992a6))
