//! What every HTTP request `stela` makes shares: a request that fails in
//! transit, or that a server refuses for now, is asked again with growing
//! waits, and a failure says its cause in words a user can act on (no
//! network, a name that does not resolve, a network that intercepts HTTPS).
use anyhow::{Result, anyhow};
use reqwest::{
    StatusCode,
    blocking::{RequestBuilder, Response},
};
use std::{error::Error as _, time::Duration};

/// Attempts per request: with waits of 1, 2, 4 and 8 s between them, a
/// network that drops for about 15 s costs a pause, not the setup.
pub(crate) const ATTEMPTS: u32 = 5;

/// The first wait between attempts; each later one doubles.
fn first_wait() -> Duration {
    if cfg!(test) {
        Duration::from_millis(10)
    } else {
        Duration::from_secs(1)
    }
}

/// A server's answer that the same request may not get again: a server
/// error, a timeout answer, or a rate limit.
pub(crate) fn transient_status(status: StatusCode) -> bool {
    status.is_server_error()
        || status == StatusCode::REQUEST_TIMEOUT
        || status == StatusCode::TOO_MANY_REQUESTS
}

/// A failure the same request may not meet again: the connection, a
/// timeout, or the transfer, never a refused redirect or a bad request.
pub(crate) fn transient(error: &reqwest::Error) -> bool {
    !error.is_redirect()
        && !error.is_builder()
        && (error.is_connect() || error.is_timeout() || error.is_request() || error.is_body())
}

/// Send the request `build` makes, asking again after a transient failure
/// ([`transient`], [`transient_status`]) up to [`ATTEMPTS`] times. Returns
/// the first answer that is not transient, whatever its status: callers
/// check it. `what` names the request in the error.
pub(crate) fn send(what: &str, build: impl Fn() -> RequestBuilder) -> Result<Response> {
    let mut wait = first_wait();
    let mut attempt = 1;
    loop {
        let failure = match build().send() {
            Ok(response) if transient_status(response.status()) && attempt < ATTEMPTS => {
                format!("{} answered {}", host_of(response.url()), response.status())
            }
            Ok(response) => return Ok(response),
            Err(error) if transient(&error) && attempt < ATTEMPTS => explain(&error),
            Err(error) => {
                let tries = match attempt {
                    1 => String::new(),
                    n => format!(" after {n} attempts"),
                };
                return Err(anyhow!("{what} failed{tries}: {}", explain(&error)));
            }
        };
        eprintln!(
            "note: {what}: {failure}; trying again in {} s",
            wait.as_secs().max(1)
        );
        std::thread::sleep(wait);
        wait *= 2;
        attempt += 1;
    }
}

fn host_of(url: &reqwest::Url) -> String {
    url.host_str().unwrap_or("the server").to_owned()
}

/// `error` with every cause under it, and what a user can do about the
/// kinds of failure a network causes.
pub(crate) fn explain(error: &reqwest::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        let cause_text = cause.to_string();
        if !text.contains(&cause_text) {
            text.push_str(": ");
            text.push_str(&cause_text);
        }
        source = cause.source();
    }
    let lower = text.to_ascii_lowercase();
    let hint = if lower.contains("dns error")
        || lower.contains("failed to lookup address")
        || lower.contains("name or service not known")
        || lower.contains("no such host")
        || lower.contains("nodename nor servname")
    {
        Some("this computer cannot look up the server's name: check its network connection")
    } else if lower.contains("certificate") || lower.contains("unknownissuer") {
        Some(
            "the server's HTTPS certificate is not one this computer trusts. A network that \
             intercepts HTTPS does this: sign in to the network (a hotel or guest network's \
             page), or set HTTPS_PROXY to the proxy that network requires",
        )
    } else if error.is_timeout() {
        Some("the server did not answer in time: check the network connection, or try again later")
    } else if lower.contains("connection refused")
        || lower.contains("actively refused")
        || lower.contains("network is unreachable")
        || lower.contains("unreachable network")
    {
        Some("this computer cannot reach the server: check its network connection and proxy")
    } else {
        None
    };
    match hint {
        Some(hint) => format!("{text} ({hint})"),
        None => text,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use reqwest::blocking::Client;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    /// Serve `replies` in order, one connection each, then close; returns
    /// the URL and how many requests arrived.
    pub(crate) fn serve(replies: Vec<String>) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let served = Arc::new(AtomicUsize::new(0));
        let count = served.clone();
        std::thread::spawn(move || {
            for reply in replies {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut request = Vec::new();
                let mut buffer = [0; 4096];
                while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                    match stream.read(&mut buffer) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => request.extend_from_slice(&buffer[..n]),
                    }
                }
                count.fetch_add(1, Ordering::SeqCst);
                let _ = stream.write_all(reply.as_bytes());
            }
        });
        (format!("http://{address}/file"), served)
    }

    fn reply(status: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    #[test]
    fn server_errors_and_rate_limits_are_asked_again() {
        let (url, served) = serve(vec![
            reply("503 Service Unavailable", ""),
            reply("429 Too Many Requests", ""),
            reply("200 OK", "done"),
        ]);
        let client = Client::new();
        let response = send("the test request", || client.get(&url)).unwrap();
        assert_eq!(response.text().unwrap(), "done");
        assert_eq!(served.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn a_refusal_is_returned_at_once_and_lasting_errors_stop() {
        let (url, served) = serve(vec![reply("404 Not Found", "")]);
        let client = Client::new();
        let response = send("the test request", || client.get(&url)).unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(served.load(Ordering::SeqCst), 1);
        // Every attempt answered 500: the last answer is returned.
        let (url, served) = serve(vec![reply("500 Internal Server Error", ""); 5]);
        let response = send("the test request", || client.get(&url)).unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(served.load(Ordering::SeqCst), ATTEMPTS as usize);
    }

    #[test]
    fn a_dropped_connection_is_asked_again_and_a_dead_server_named() {
        // The first connection closes without an answer.
        let (url, served) = serve(vec![String::new(), reply("200 OK", "done")]);
        let client = Client::new();
        let response = send("the test request", || client.get(&url)).unwrap();
        assert_eq!(response.text().unwrap(), "done");
        assert_eq!(served.load(Ordering::SeqCst), 2);
        // Nothing listens: every attempt fails, and the error says so.
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let error = send("the test request", || {
            client.get(format!("http://127.0.0.1:{port}/file"))
        })
        .unwrap_err()
        .to_string();
        assert!(
            error.starts_with("the test request failed after 5 attempts"),
            "{error}"
        );
        assert!(error.contains("cannot reach the server"), "{error}");
    }
}
