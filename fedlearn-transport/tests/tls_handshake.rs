//! mTLS transport integration tests.

mod common;

use common::TestPki;

#[test]
fn test_pki_generates_parseable_material() {
    let pki = TestPki::generate();
    let (client_cert, client_key) = pki.client_cert("client-1");

    for (label, pem) in [
        ("ca", pki.ca_pem.as_str()),
        ("server_cert", pki.server_cert_pem.as_str()),
        ("client_cert", client_cert.as_str()),
    ] {
        let (_, parsed) =
            x509_parser::pem::parse_x509_pem(pem.as_bytes()).unwrap_or_else(|e| {
                panic!("{label} PEM does not parse: {e}");
            });
        assert!(
            parsed.parse_x509().is_ok(),
            "{label} is not a valid certificate"
        );
    }

    assert!(pki.server_key_pem.contains("PRIVATE KEY"));
    assert!(client_key.contains("PRIVATE KEY"));
}
