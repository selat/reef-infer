//! Field-level diff of two compiled models' bitstreams, executable by
//! executable (instructions compared in order when the counts match).
//!
//!     cargo run --example isa_diff -- a_edgetpu.tflite b_edgetpu.tflite

use reef_infer::isa;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (ma, mb) = (
        reef_infer::model::load_tflite_model(&a[1]).unwrap(),
        reef_infer::model::load_tflite_model(&a[2]).unwrap(),
    );
    for (ei, (ea, eb)) in ma.executables.iter().zip(&mb.executables).enumerate() {
        println!(
            "exec{ei} params {} vs {}",
            ea.parameters.len(),
            eb.parameters.len()
        );
        let (ia, ib) = (
            isa::split(&ea.instruction_chunks[0].bitstream).unwrap(),
            isa::split(&eb.instruction_chunks[0].bitstream).unwrap(),
        );
        if ia.len() != ib.len() {
            println!("  instruction count {} vs {}", ia.len(), ib.len());
            continue;
        }
        for (n, (x, y)) in ia.iter().zip(&ib).enumerate() {
            let (dx, dy) = (x.decode().unwrap(), y.decode().unwrap());
            if dx.layout.name != dy.layout.name {
                println!("  #{n} {} vs {}", dx.layout.name, dy.layout.name);
                continue;
            }
            for (f, (va, vb)) in dx
                .layout
                .fields
                .iter()
                .zip(dx.values.iter().zip(&dy.values))
            {
                if va != vb {
                    println!("  #{n} {}.{}: {} vs {}", dx.layout.name, f.name, va, vb);
                }
            }
        }
    }
}
