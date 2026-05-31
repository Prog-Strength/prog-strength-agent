## [0.11.1](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.11.0...v0.11.1) (2026-05-31)


### Bug Fixes

* **router:** drop user input text from INFO log before CloudWatch flip ([a29380c](https://github.com/Prog-Strength/prog-strength-agent/commit/a29380c7c7e7400fa1cd3652924a1611c8663fb6))

# [0.11.0](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.10.0...v0.11.0) (2026-05-31)


### Features

* **chat:** hyped fitness-coach voice + bro-energy system prompt ([479ce4e](https://github.com/Prog-Strength/prog-strength-agent/commit/479ce4eb8b7e629393ec82b0658bcf65e365b2df))

# [0.10.0](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.9.1...v0.10.0) (2026-05-31)


### Features

* **chat:** /speak endpoint for OpenAI TTS-driven voice replies ([2e055f3](https://github.com/Prog-Strength/prog-strength-agent/commit/2e055f320c06e2510d97f1dcea8ba2844a95bc35))

## [0.9.1](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.9.0...v0.9.1) (2026-05-30)


### Bug Fixes

* **chat:** stop Haiku titling every conversation "New Chat" ([82e9b7d](https://github.com/Prog-Strength/prog-strength-agent/commit/82e9b7d880eea67dfe93571e74292f677f3bef28))

# [0.9.0](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.8.1...v0.9.0) (2026-05-30)


### Features

* **chat:** /title endpoint generates 3-6 word session titles via Haiku ([ff57a8b](https://github.com/Prog-Strength/prog-strength-agent/commit/ff57a8b7b41877296e71f0331c907124fea43a17))

## [0.8.1](https://github.com/Prog-Strength/prog-strength-agent/compare/v0.8.0...v0.8.1) (2026-05-19)


### Bug Fixes

* **build:** build image on a native ARM runner ([740e9a7](https://github.com/Prog-Strength/prog-strength-agent/commit/740e9a7f1a59d3dca637be8497dd79b8c1ec32d5))

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
