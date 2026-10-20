# Morpheus

[![Release v0.2.0](https://img.shields.io/github/v/release/Aporia-Nous/morpheus?color=blue&label=release)](https://github.com/Aporia-Nous/morpheus/releases/tag/v0.2.0)
[![Tests Passing](https://img.shields.io/badge/conformance-28%2F28%20passed-brightgreen)](https://github.com/Aporia-Nous/morpheus)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A sovereign programming language — open core.**

Morpheus is a language for code that *morphs*: functions that rewrite themselves at runtime, values that carry a **confidence**, and a predictive prophesy primitive. It is **open-core** — the language and its tooling are open; a separate production runtime is private.

---

### Verified Architecture & Implementation Proof
- **Reference Interpreter (interp.js)**: Runs in the browser and Node.js with zero runtime dependencies.
- **Native C++ Runtime Header (
untime/morph_rt.h)**: C FFI boundary, arena memory allocator, and high-performance struct definitions.
- **Compiler Pipeline (morphc.js)**: Lexer, recursive-descent parser, AST generation, and target bytecode generation.
- **Rigorous Verification (conformance.js & 	est.js)**: Complete 28/28 verified test suite covering function dispatch, pattern matching, error-healing semantics, and module imports.

---

## Learn it — Morphoria, an Aporia Nous game
Play a course that teaches the language one small step at a time — earn XP, build combos, keep your lives, climb the ranks, and finish in a **build workspace that grades your code with real tests before it compiles**:

`ash
node morph.js play learn/morphoria.morph
`

See docs/LEARN.md, then build real apps in examples/apps/.

## Open (this repo)
- The language: syntax, keywords, the spec.
- A **reference interpreter** (interp.js) — runs in the browser and in Node.
- The standard library (stdlib/).
- An **extension API** — register builtins, add language features, ship packages.
- Docs and an interactive tutorial.

## Try it
- **Interactive:** open index.html (or the GitHub Pages site).
- **Node:**
  `ash
  node morph.js run examples/hello.morph
  `
