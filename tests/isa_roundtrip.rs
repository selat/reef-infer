//! Hardware-free check of the instruction decoder/encoder: every bitstream in
//! the compiled models splits cleanly into known instructions and re-encodes
//! byte for byte.
//!
//! Models come from `tests/models/` plus any files or directories listed in
//! `REEF_ISA_MODELS` (colon-separated, searched recursively for
//! `*edgetpu*.tflite`). With no models available the test is skipped.

use std::path::{Path, PathBuf};

use reef_infer::isa;
use reef_infer::model::load_edgetpu_ops;

fn collect(p: &Path, out: &mut Vec<PathBuf>) {
    if p.is_dir() {
        for e in std::fs::read_dir(p).into_iter().flatten().flatten() {
            collect(&e.path(), out);
        }
    } else if p
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.contains("edgetpu") && n.ends_with(".tflite"))
    {
        out.push(p.to_path_buf());
    }
}

#[test]
fn compiled_bitstreams_round_trip() {
    let mut paths = Vec::new();
    collect(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models"),
        &mut paths,
    );
    for p in std::env::var("REEF_ISA_MODELS")
        .unwrap_or_default()
        .split(':')
        .filter(|s| !s.is_empty())
    {
        collect(Path::new(p), &mut paths);
    }
    if paths.is_empty() {
        eprintln!("no compiled models found; skipping (set REEF_ISA_MODELS)");
        return;
    }
    let mut checked = 0;
    for path in &paths {
        let Ok(ops) = load_edgetpu_ops(path.to_str().unwrap()) else {
            continue;
        };
        for ex in ops.iter().flat_map(|m| &m.executables) {
            for ch in &ex.instruction_chunks {
                let instrs = isa::split(&ch.bitstream).unwrap_or_else(|e| panic!("{path:?}: {e}"));
                assert_eq!(
                    instrs.first().map(|i| i.opcode()),
                    Some(0x3e),
                    "{path:?}: no start"
                );
                assert_eq!(
                    instrs.last().map(|i| i.opcode()),
                    Some(0x3f),
                    "{path:?}: no end"
                );
                let mut words = Vec::new();
                let decoded: Vec<_> = instrs
                    .iter()
                    .map(|ins| {
                        ins.decode()
                            .unwrap_or_else(|e| panic!("{path:?}@{}: {e}", ins.word_index))
                    })
                    .collect();
                check_program(path, &decoded);
                for (ins, d) in instrs.iter().zip(&decoded) {
                    if d.layout.name == "wideToNarrow" {
                        // tail = innermost narrow row length - head length
                        let g = |n: &str| d.get(n).unwrap();
                        let expected = if g("head_en") == 1 {
                            g("n_lim0") - g("head_words_m1")
                        } else {
                            0
                        };
                        assert_eq!(
                            g("tail_words"),
                            expected,
                            "{path:?}@{}: tail_words",
                            ins.word_index
                        );
                    }
                    words.extend(d.encode().unwrap());
                }
                assert!(
                    isa::assemble(words.chunks(1)) == ch.bitstream,
                    "{path:?}: re-encode differs"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "models found but none loaded");
    eprintln!("checked {checked} bitstreams from {} models", paths.len());
}

/// Cross-instruction rules found in the corpus (docs/CODEGEN_RESEARCH.md).
fn check_program(path: &Path, prog: &[isa::Decoded<'_>]) {
    let named = |d: &isa::Decoded<'_>, n: &str| d.get(n).unwrap();
    // Loopback: every outfeed with loop_en pairs, in order, with an earlier
    // infeed with loop_en and the same transfer limits.
    let dims = |d: &isa::Decoded<'_>| ["d0_limit", "d1_limit", "d2_limit"].map(|n| named(d, n));
    let loop_in: Vec<_> = prog
        .iter()
        .enumerate()
        .filter(|(_, d)| d.layout.name == "infeed" && named(d, "loop_en") == 1)
        .collect();
    let loop_out: Vec<_> = prog
        .iter()
        .enumerate()
        .filter(|(_, d)| d.layout.name == "outfeed" && named(d, "loop_en") == 1)
        .collect();
    assert_eq!(
        loop_in.len(),
        loop_out.len(),
        "{path:?}: loopback infeed/outfeed count"
    );
    for ((i, a), (o, b)) in loop_in.iter().zip(&loop_out) {
        assert!(i < o, "{path:?}: loopback infeed after its outfeed");
        assert_eq!(dims(a), dims(b), "{path:?}: loopback limits differ");
    }
    // Streamed parameters: a pop's buf_off is where the next parameter infeed reads.
    for (k, p) in prog.iter().enumerate() {
        if p.layout.name != "pop" || named(p, "stream") != 0 || named(p, "buf_off") == 0 {
            continue;
        }
        if let Some(inf) = prog[k + 1..]
            .iter()
            .find(|d| d.layout.name == "infeed" && named(d, "stream") == 0)
        {
            assert_eq!(
                named(inf, "buf_off"),
                named(p, "buf_off"),
                "{path:?}: pop/infeed buf_off"
            );
        }
    }
}
