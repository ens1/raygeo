use pyo3_stub_gen::Result;
use std::{fs, path::Path};

fn normalize_process_stub(path: &Path) -> Result<()> {
    let content = fs::read_to_string(path)?;
    let normalized = content
        .replace(
            "Marker.ProcessStart: ...\n    \n",
            "Marker.ProcessStart: ...\n\n",
        )
        .replace(
            "Marker.ProcessEnd: ...\n    \n",
            "Marker.ProcessEnd: ...\n\n",
        )
        .replace(
            "Mark the start of a host-defined process.\n        \n",
            "Mark the start of a host-defined process.\n\n",
        );
    if normalized != content {
        fs::write(path, normalized)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let stub = raygeo::stub_info()?;
    let python_root = stub.python_root.clone();
    stub.generate()?;
    normalize_process_stub(
        &python_root.join("raygeo/cnc/execution/specs/__init__.pyi"),
    )?;
    normalize_process_stub(&python_root.join("raygeo/ops/__init__.pyi"))?;
    Ok(())
}
