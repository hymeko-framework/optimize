//! The browser surface compiles from memory what the desktop compiles from disk.
//!
//! `web::describe` hands Core's in-memory `MemProvider` the sources a page would send, keyed by
//! the path they are imported as. The success path builds no `JsValue`, so it runs natively.

#![cfg(feature = "web")]

mod common;

use hymeko_optimize::read::project;
use hymeko_optimize::web::describe;

#[test]
fn describe_reads_the_same_topology_from_memory() {
    let (ir, it) = divider!();
    let problem = project(&ir, &it).expect("the model projects");

    let read = |file: &str| std::fs::read_to_string(common::model_path(file)).expect("readable");
    let described = describe(
        "optimization/resistor_divider.hymeko".into(),
        vec![
            "optimization/resistor_divider.hymeko".into(),
            "optimization/meta_optimization.hymeko".into(),
        ],
        vec![
            read("resistor_divider.hymeko"),
            read("meta_optimization.hymeko"),
        ],
    )
    .unwrap_or_else(|_| panic!("the in-memory sources compile and project"));

    assert!(!problem.net_names.is_empty());
    for name in problem.net_names.iter().chain(&problem.branch_names) {
        assert!(described.contains(&format!("\"{name}")), "{name} missing:\n{described}");
    }
}
