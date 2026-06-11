//! Shared test support: ephemeral PKI generated at runtime with rcgen.
//! No certificate material is ever committed to the repository.

#![allow(dead_code)]

use rcgen::{
    BasicConstraints, Certificate, CertificateParams, DnType, IsCa, KeyPair, KeyUsagePurpose,
};

/// An ephemeral certificate authority plus a server identity for `localhost`.
pub struct TestPki {
    pub ca_pem: String,
    pub server_cert_pem: String,
    pub server_key_pem: String,
    ca_cert: Certificate,
    ca_key: KeyPair,
}

impl TestPki {
    pub fn generate() -> Self {
        let (ca_cert, ca_key) = make_ca("quantumacy test ca");

        let server_key = KeyPair::generate().expect("server keypair");
        let mut params =
            CertificateParams::new(vec!["localhost".to_string()]).expect("server params");
        params
            .distinguished_name
            .push(DnType::CommonName, "localhost");
        let server_cert = params
            .signed_by(&server_key, &ca_cert, &ca_key)
            .expect("sign server cert");

        Self {
            ca_pem: ca_cert.pem(),
            server_cert_pem: server_cert.pem(),
            server_key_pem: server_key.serialize_pem(),
            ca_cert,
            ca_key,
        }
    }

    /// Mint a CA-signed client certificate whose CN is `cn`.
    /// Returns (cert_pem, key_pem).
    pub fn client_cert(&self, cn: &str) -> (String, String) {
        let key = KeyPair::generate().expect("client keypair");
        let mut params = CertificateParams::new(Vec::new()).expect("client params");
        params.distinguished_name.push(DnType::CommonName, cn);
        let cert = params
            .signed_by(&key, &self.ca_cert, &self.ca_key)
            .expect("sign client cert");
        (cert.pem(), key.serialize_pem())
    }

    /// A client certificate from a *different* CA — must be rejected by the
    /// server. Returns (cert_pem, key_pem, rogue_ca_pem).
    pub fn wrong_ca_client(cn: &str) -> (String, String, String) {
        let (rogue_ca, rogue_key) = make_ca("rogue ca");
        let key = KeyPair::generate().expect("rogue client keypair");
        let mut params = CertificateParams::new(Vec::new()).expect("rogue params");
        params.distinguished_name.push(DnType::CommonName, cn);
        let cert = params
            .signed_by(&key, &rogue_ca, &rogue_key)
            .expect("sign rogue client cert");
        (cert.pem(), key.serialize_pem(), rogue_ca.pem())
    }
}

fn make_ca(cn: &str) -> (Certificate, KeyPair) {
    let key = KeyPair::generate().expect("ca keypair");
    let mut params = CertificateParams::new(Vec::new()).expect("ca params");
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.distinguished_name.push(DnType::CommonName, cn);
    params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
    let cert = params.self_signed(&key).expect("self-sign ca");
    (cert, key)
}
