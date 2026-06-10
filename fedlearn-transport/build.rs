fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure `protoc` is available even on hosts that do not ship one.
    // If the operator has provided their own via `PROTOC`, respect it.
    if std::env::var_os("PROTOC").is_none() {
        if let Ok(path) = protoc_bin_vendored::protoc_bin_path() {
            std::env::set_var("PROTOC", path);
        }
    }

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(&["proto/fedlearn.proto"], &["proto"])?;

    println!("cargo:rerun-if-changed=proto/fedlearn.proto");
    Ok(())
}
