//! TLS configuration for the federated learning transport.
//!
//! The server requires mutual TLS by default: it presents an identity from
//! `QUANTUMACY_TLS_CERT` / `QUANTUMACY_TLS_KEY` and only accepts clients whose
//! certificates chain to `QUANTUMACY_TLS_CLIENT_CA`. Plaintext operation is
//! available solely for local demos behind `QUANTUMACY_INSECURE=1` and logs a
//! warning at serve time.

use crate::grpc_service::GrpcFederatedLearningService;
use crate::proto::federated_learning_server::FederatedLearningServer;
use crate::proto::key_exchange_server::KeyExchangeServer;
use crate::{TransportError, TransportResult};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::Path;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::{Certificate, ClientTlsConfig, Identity, Server, ServerTlsConfig};
use tracing::warn;

pub const ENV_TLS_CERT: &str = "QUANTUMACY_TLS_CERT";
pub const ENV_TLS_KEY: &str = "QUANTUMACY_TLS_KEY";
pub const ENV_TLS_CLIENT_CA: &str = "QUANTUMACY_TLS_CLIENT_CA";
pub const ENV_INSECURE: &str = "QUANTUMACY_INSECURE";

/// Transport security settings resolved from the environment.
#[derive(Debug, Clone)]
pub enum TlsSettings {
    /// Mutual TLS: server identity plus a required client CA.
    Mutual {
        cert_pem: Vec<u8>,
        key_pem: Vec<u8>,
        client_ca_pem: Vec<u8>,
    },
    /// Plaintext gRPC. Demo-only; serving in this mode logs a warning.
    Insecure,
}

impl TlsSettings {
    /// Resolve settings from an explicit variable map (testable without
    /// touching process environment). File contents are read eagerly so
    /// misconfiguration fails at startup, not on first connection.
    pub fn from_env_map(vars: &HashMap<String, String>) -> TransportResult<Self> {
        if vars.get(ENV_INSECURE).map(String::as_str) == Some("1") {
            return Ok(Self::Insecure);
        }

        let missing: Vec<&str> = [ENV_TLS_CERT, ENV_TLS_KEY, ENV_TLS_CLIENT_CA]
            .into_iter()
            .filter(|name| !vars.contains_key(*name))
            .collect();
        if !missing.is_empty() {
            return Err(TransportError::Config(format!(
                "missing required TLS environment variables: {} (set {ENV_INSECURE}=1 \
                 only for local plaintext demos)",
                missing.join(", ")
            )));
        }

        Ok(Self::Mutual {
            cert_pem: read_pem(&vars[ENV_TLS_CERT], ENV_TLS_CERT)?,
            key_pem: read_pem(&vars[ENV_TLS_KEY], ENV_TLS_KEY)?,
            client_ca_pem: read_pem(&vars[ENV_TLS_CLIENT_CA], ENV_TLS_CLIENT_CA)?,
        })
    }

    /// Resolve settings from the process environment.
    pub fn from_env() -> TransportResult<Self> {
        Self::from_env_map(&std::env::vars().collect())
    }

    /// Server-side TLS config. `None` means serve plaintext (insecure mode).
    pub fn server_tls_config(&self) -> Option<ServerTlsConfig> {
        match self {
            Self::Mutual {
                cert_pem,
                key_pem,
                client_ca_pem,
            } => Some(
                ServerTlsConfig::new()
                    .identity(Identity::from_pem(cert_pem, key_pem))
                    .client_ca_root(Certificate::from_pem(client_ca_pem)),
            ),
            Self::Insecure => {
                warn!("{ENV_INSECURE}=1: serving PLAINTEXT gRPC — local demos only");
                None
            }
        }
    }
}

/// Serve the FL + key-exchange services on `addr` with the given settings.
/// Runs until the server is shut down.
pub async fn serve(
    settings: &TlsSettings,
    addr: SocketAddr,
    fl_service: GrpcFederatedLearningService,
) -> TransportResult<()> {
    configured_builder(settings)?
        .add_service(FederatedLearningServer::new(fl_service.clone()))
        .add_service(KeyExchangeServer::new(fl_service.key_exchange_service()))
        .serve(addr)
        .await?;
    Ok(())
}

/// Serve on an already-bound listener (lets tests use an ephemeral port).
pub async fn serve_with_listener(
    settings: &TlsSettings,
    listener: tokio::net::TcpListener,
    fl_service: GrpcFederatedLearningService,
) -> TransportResult<()> {
    configured_builder(settings)?
        .add_service(FederatedLearningServer::new(fl_service.clone()))
        .add_service(KeyExchangeServer::new(fl_service.key_exchange_service()))
        .serve_with_incoming(TcpListenerStream::new(listener))
        .await?;
    Ok(())
}

fn configured_builder(settings: &TlsSettings) -> TransportResult<Server> {
    let builder = Server::builder();
    match settings.server_tls_config() {
        Some(tls) => Ok(builder.tls_config(tls)?),
        None => Ok(builder),
    }
}

/// Client-side mTLS config: trust `ca_pem`, present `cert_pem`/`key_pem`,
/// and validate the server certificate against `domain`.
pub fn client_tls_config(
    ca_pem: &[u8],
    cert_pem: &[u8],
    key_pem: &[u8],
    domain: &str,
) -> ClientTlsConfig {
    ClientTlsConfig::new()
        .ca_certificate(Certificate::from_pem(ca_pem))
        .identity(Identity::from_pem(cert_pem, key_pem))
        .domain_name(domain)
}

fn read_pem(path: &str, var: &str) -> TransportResult<Vec<u8>> {
    std::fs::read(Path::new(path))
        .map_err(|e| TransportError::Config(format!("unable to read {var} file {path}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_pem(name: &str) -> String {
        let path = std::env::temp_dir().join(format!(
            "quantumacy-tls-test-{}-{name}.pem",
            std::process::id()
        ));
        std::fs::write(
            &path,
            b"-----BEGIN TEST-----\nplaceholder\n-----END TEST-----\n",
        )
        .unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn test_mutual_settings_from_complete_env() {
        let vars: HashMap<String, String> = [
            (ENV_TLS_CERT.to_string(), temp_pem("cert")),
            (ENV_TLS_KEY.to_string(), temp_pem("key")),
            (ENV_TLS_CLIENT_CA.to_string(), temp_pem("ca")),
        ]
        .into();

        let settings = TlsSettings::from_env_map(&vars).unwrap();
        assert!(matches!(settings, TlsSettings::Mutual { .. }));
    }

    #[test]
    fn test_insecure_flag_short_circuits() {
        let vars: HashMap<String, String> = [(ENV_INSECURE.to_string(), "1".to_string())].into();

        let settings = TlsSettings::from_env_map(&vars).unwrap();
        assert!(matches!(settings, TlsSettings::Insecure));
    }

    #[test]
    fn test_partial_config_names_missing_variables() {
        let vars: HashMap<String, String> =
            [(ENV_TLS_CERT.to_string(), temp_pem("only-cert"))].into();

        let err = TlsSettings::from_env_map(&vars).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(ENV_TLS_KEY), "missing var not named: {msg}");
        assert!(
            msg.contains(ENV_TLS_CLIENT_CA),
            "missing var not named: {msg}"
        );
    }

    #[test]
    fn test_unreadable_path_errors_with_path() {
        let vars: HashMap<String, String> = [
            (
                ENV_TLS_CERT.to_string(),
                "/nonexistent/cert.pem".to_string(),
            ),
            (ENV_TLS_KEY.to_string(), temp_pem("key2")),
            (ENV_TLS_CLIENT_CA.to_string(), temp_pem("ca2")),
        ]
        .into();

        let err = TlsSettings::from_env_map(&vars).unwrap_err();
        assert!(err.to_string().contains("/nonexistent/cert.pem"));
    }
}
