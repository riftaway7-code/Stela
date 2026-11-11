//! A remote install image (`payload::Layout::Remote`): the IPSW on Apple's
//! CDN, read by URL-aware byte ranges over the mount's own network. HTTPS
//! requires the native owner's admitted TLS time and entropy capabilities.
//!
//! Each request carries `If-Range` with the entity tag setup recorded, so a
//! changed object fails instead of mixing two versions; the restore store
//! then checks the bytes as the manifest says (`payload::Member::sha256`).
//! One range at a time, from where the restore reads to the image's end
//! (an ordinary download from there), streams into a read-ahead buffer: a
//! read at or just past the buffer continues it, and a read elsewhere
//! starts one there, abandoning only what the TCP window has in flight.
//! [`Remote::pump`] keeps the buffer filling while the engine runs the
//! guest, so the download overlaps the restore's own writes. A range that
//! fails resumes from its last byte received, a few times.
//!
//! The restore host reads it on a PE that waits for an interrupt
//! (`service`), whose guest CPU takes none while it reads: once
//! [`Remote::bound`] is set, a read waits at most [`BUDGET_MS`] for the
//! network and otherwise fails as blocked (`ReadAt::blocked`), and the
//! restore asks again at a later service, while the range goes on filling.
//! A read elsewhere keeps the buffer it leaves as a segment
//! ([`SEGMENT_BYTES`] in all), so a request that reads several members
//! again from its first finds those it already read.
use super::network::Shared;
use crate::Result;
use crate::output;
use alloc::{collections::VecDeque, string::String, vec::Vec};
use stela_contracts::refuse;
use {
    stela_formats::payload::Layout, stela_formats::payload::PayloadManifest,
    stela_formats::payload::ReadAt, stela_runtime::net::HttpTransfer, stela_runtime::net::NetError,
    stela_runtime::net::http::Ranged,
};

/// How far [`Remote::pump`] fills ahead of the restore.
const AHEAD: usize = 32 << 20;
/// A read up to this far past the buffered bytes continues the open range.
const SKIP: u64 = 8 << 20;
/// Bytes already read that stay buffered for a re-read.
const KEEP: usize = 8 << 20;
/// Consecutive failures of one range's progress before a read fails; the
/// waits between them double from a second, about a minute in all.
const ATTEMPTS: u32 = 7;
/// The longest one wait for an open range's next frame lasts.
const WAIT_US: u64 = 100_000;
/// The longest a bounded read waits for the network, and so holds the
/// guest CPU of the PE that serves it.
const BUDGET_MS: u64 = 20;
/// The bytes kept from earlier buffers for reads that come back to them.
const SEGMENT_BYTES: usize = 64 << 20;

pub struct Remote {
    network: Shared,
    url: String,
    etag: String,
    len: u64,
    /// Image bytes from `start`; the open range continues at their end.
    start: u64,
    buffer: VecDeque<u8>,
    range: Option<(HttpTransfer, Ranged)>,
    /// Ranges requested so far.
    ranges: u64,
    /// Whether [`Remote::pump`] reads ahead: from a read that succeeded
    /// until a read-ahead fails.
    ahead: bool,
    /// Earlier buffers, by their first byte, oldest first: each up to
    /// where its reads reached, without what was read ahead past them.
    segments: VecDeque<(u64, Vec<u8>)>,
    /// The farthest byte a read asked for since the buffer started.
    wanted: u64,
    /// Whether reads wait for the network at most [`BUDGET_MS`].
    bounded: bool,
    /// The last read failed only because its time ran out.
    blocked: bool,
    /// Consecutive failures of the open range's progress, and when the
    /// last one was, for a bounded read's backoff.
    failures: u32,
    failed_at: u64,
}

impl Remote {
    pub fn new(network: Shared, manifest: &PayloadManifest) -> Result<Self> {
        let Layout::Remote { url, etag } = &manifest.layout else {
            return Err(refuse!("the install payload is not a remote image"));
        };
        network
            .0
            .borrow()
            .check_url(url)
            .map_err(|error| refuse!("remote image URL: {error}", %error))?;
        let last = manifest
            .source
            .bytes
            .checked_sub(1)
            .ok_or_else(|| refuse!("the remote image is empty"))?;
        Ranged::with_identity(0, last, etag, manifest.source.bytes)
            .map_err(|error| refuse!("remote image range: {error}", %error))?;
        output::arguments(format_args!(
            "REMOTE IMAGE {url} bytes={} etag={etag}\n",
            manifest.source.bytes
        ));
        Ok(Self {
            network,
            url: url.clone(),
            etag: etag.clone(),
            len: manifest.source.bytes,
            start: 0,
            // Room for the read-ahead, what stays for a re-read and a
            // socket's worth more, so the ring rarely grows.
            buffer: VecDeque::with_capacity(AHEAD + KEEP + (8 << 20)),
            range: None,
            ranges: 0,
            ahead: false,
            segments: VecDeque::new(),
            wanted: 0,
            bounded: false,
            blocked: false,
            failures: 0,
            failed_at: 0,
        })
    }

    fn end(&self) -> u64 {
        self.start + self.buffer.len() as u64
    }

    /// Have every later read wait for the network at most [`BUDGET_MS`]:
    /// from when the restore host serves on a PE.
    pub fn bound(&mut self) {
        self.bounded = true;
    }

    /// Keep the buffer as a segment before it starts over elsewhere,
    /// dropping the oldest segments past [`SEGMENT_BYTES`].
    fn keep_segment(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let reached = self.wanted.saturating_sub(self.start) as usize;
        let mut bytes: Vec<u8> = self.buffer.drain(..).take(reached).collect();
        bytes.truncate(SEGMENT_BYTES);
        if bytes.is_empty() {
            return;
        }
        self.segments.push_back((self.start, bytes));
        let mut total: usize = self.segments.iter().map(|(_, b)| b.len()).sum();
        while total > SEGMENT_BYTES
            && let Some((_, dropped)) = self.segments.pop_front()
        {
            total -= dropped.len();
        }
    }

    /// Copy `out` from a kept segment that holds all of it.
    fn read_segment(&self, offset: u64, out: &mut [u8]) -> bool {
        let found = self.segments.iter().rev().find(|(start, bytes)| {
            offset >= *start && offset + out.len() as u64 <= *start + bytes.len() as u64
        });
        if let Some((start, bytes)) = found {
            let at = (offset - start) as usize;
            out.copy_from_slice(&bytes[at..at + out.len()]);
        }
        found.is_some()
    }

    fn stop(&mut self) {
        if let Some((transfer, _)) = self.range.take() {
            self.network.0.borrow_mut().close_download(transfer);
        }
    }

    /// Release the restore-only range and ring after the protocol has ended.
    pub(super) fn retire(&mut self) {
        self.stop();
        self.ahead = false;
        let bytes = self.buffer.capacity();
        self.buffer = VecDeque::new();
        self.segments = VecDeque::new();
        output::arguments(format_args!(
            "REMOTE IMAGE retired read-ahead capacity={bytes}\n"
        ));
    }

    /// Takes what has arrived, opening a range at the buffer's end first
    /// when none is open.
    fn step(&mut self) -> Result<(), NetError> {
        if self.range.is_none() {
            let (first, last) = (self.end(), self.len - 1);
            let ranged = Ranged::with_identity(first, last, &self.etag, self.len)?;
            let transfer = self
                .network
                .0
                .borrow_mut()
                .download(&self.url, first, last, &self.etag)?;
            self.range = Some((transfer, ranged));
            self.ranges += 1;
            output::arguments(format_args!(
                "REMOTE IMAGE range {} bytes={first}-{last}\n",
                self.ranges
            ));
        }
        let (transfer, ranged) = self.range.as_mut().expect("opened above");
        let buffer = &mut self.buffer;
        let done = self
            .network
            .0
            .borrow_mut()
            .read_download(transfer, |bytes, closed| {
                ranged.feed(bytes, closed, |body| buffer.extend(body))
            })?;
        if done {
            self.stop();
        }
        Ok(())
    }

    /// Fills the buffer up to `until`: bounded, within [`BUDGET_MS`], and
    /// after a failure not before its backoff ends; unbounded, waiting
    /// those out. A failure's count goes on across bounded reads.
    fn fill(&mut self, until: u64) -> Result<(), NetError> {
        let started = output::now();
        while self.end() < until {
            let spent = output::millis_since(started);
            if self.bounded && spent >= BUDGET_MS {
                return Err(self.block("the read's time ran out"));
            }
            if self.failures > 0 {
                let backoff = 1000 << (self.failures - 1);
                if output::millis_since(self.failed_at) < backoff {
                    if self.bounded {
                        return Err(self.block("the range waits before it is asked again"));
                    }
                    while output::millis_since(self.failed_at) < backoff {
                        core::hint::spin_loop();
                    }
                }
            }
            let before = self.end();
            if let Err(error) = self.step() {
                self.stop();
                if !error.retryable() {
                    self.failures = 0;
                    return Err(error);
                }
                self.failures += 1;
                output::arguments(format_args!(
                    "REMOTE IMAGE range at {before} failed ({}/{ATTEMPTS}): {error}\n",
                    self.failures
                ));
                if self.failures == ATTEMPTS {
                    self.failures = 0;
                    return Err(error);
                }
                self.failed_at = output::now();
            } else if self.end() > before {
                self.failures = 0;
            } else if self.range.is_some() {
                let wait = match self.bounded {
                    true => (BUDGET_MS - spent) * 1000,
                    false => WAIT_US,
                };
                let result = self.network.0.borrow_mut().wait(wait.min(WAIT_US));
                if let Err(error) = result {
                    self.stop();
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// A bounded read that ran out of time: the range keeps filling between
    /// the restore's requests, and the same read later goes on.
    fn block(&mut self, why: &'static str) -> NetError {
        self.blocked = true;
        self.ahead = true;
        NetError::Timeout(why)
    }

    /// Without waiting, takes what has arrived while less than [`AHEAD`] is
    /// buffered, opening the next range when one ends. A failure leaves the
    /// next read to retry.
    pub fn pump(&mut self) {
        if self.ahead && self.buffer.len() < AHEAD && self.end() < self.len && self.step().is_err()
        {
            self.stop();
            self.ahead = false;
        }
    }
}

impl ReadAt for Remote {
    type Error = NetError;

    fn len(&self) -> u64 {
        self.len
    }

    fn blocked(&mut self) -> bool {
        core::mem::take(&mut self.blocked)
    }

    fn read_at(&mut self, offset: u64, out: &mut [u8]) -> Result<(), NetError> {
        let until = offset
            .checked_add(out.len() as u64)
            .ok_or(NetError::Invalid("remote image offset overflow"))?;
        if until > self.len {
            return Err(NetError::Invalid("read past the remote image's end"));
        }
        if self.read_segment(offset, out) {
            return Ok(());
        }
        if offset < self.start || offset > self.end() + SKIP {
            self.stop();
            self.keep_segment();
            self.start = offset;
            self.wanted = offset;
        }
        self.wanted = self.wanted.max(until);
        self.fill(until)?;
        self.ahead = true;
        // The ring's two halves, from the read's first byte.
        let (mut at, mut filled) = ((offset - self.start) as usize, 0);
        for part in <[&[u8]; 2]>::from(self.buffer.as_slices()) {
            if at < part.len() {
                let take = (part.len() - at).min(out.len() - filled);
                out[filled..filled + take].copy_from_slice(&part[at..at + take]);
                filled += take;
                at = 0;
            } else {
                at -= part.len();
            }
        }
        // Bytes before this read are done with, but the last few stay for
        // a re-read.
        let done = (until - self.start) as usize;
        if done > KEEP {
            self.buffer.drain(..done - KEEP);
            self.start = until - KEEP as u64;
        }
        Ok(())
    }
}
