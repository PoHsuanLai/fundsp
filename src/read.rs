//! Symphonia integration for reading audio files.

use super::wave::*;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;
extern crate alloc;
use alloc::boxed::Box;
use symphonia::core::audio::{AudioBuffer, Signal};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::{Error, Result};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub type WaveResult<T> = Result<T>;
pub type WaveError = Error;

impl Wave {
    /// Load first track of audio file from the given path.
    /// Supported formats are anything that Symphonia can read.
    pub fn load<P: AsRef<Path>>(path: P) -> WaveResult<Wave> {
        Wave::load_track(path, None)
    }

    /// Load first track with a progress callback (0.0 to 1.0).
    /// Progress is reported roughly every 0.5 seconds of decoded audio.
    /// If the format doesn't report total frames, the callback is not invoked.
    pub fn load_with_progress<P: AsRef<Path>>(
        path: P,
        on_progress: impl Fn(f32),
    ) -> WaveResult<Wave> {
        Wave::load_track_with_progress(path, None, on_progress)
    }

    /// Load first track with progress and streaming peak callbacks.
    /// `on_peaks` is called with accumulated level-0 peaks (256 spp)
    /// each time progress is reported. Returns the wave and final peaks.
    pub fn load_with_peaks<P: AsRef<Path>>(
        path: P,
        on_progress: impl Fn(f32),
        on_peaks: impl Fn(Vec<(f32, f32)>),
        on_metadata: impl Fn(usize, u32),
        on_samples: impl Fn(&[f32]),
    ) -> WaveResult<(Wave, Vec<(f32, f32)>)> {
        let path = path.as_ref();
        let mut hint = Hint::new();

        if let Some(extension) = path.extension()
            && let Some(extension_str) = extension.to_str()
        {
            hint.with_extension(extension_str);
        }

        let source: Box<dyn MediaSource> = match File::open(path) {
            Ok(file) => Box::new(file),
            Err(error) => return Err(Error::IoError(error)),
        };

        Wave::decode_with_peaks(source, None, hint, &on_progress, &on_peaks as &dyn Fn(Vec<(f32, f32)>), &on_metadata, &on_samples)
    }

    /// Load first track of audio from the given slice.
    /// Supported formats are anything that Symphonia can read.
    pub fn load_slice<S: AsRef<[u8]> + Send + Sync + 'static>(slice: S) -> WaveResult<Wave> {
        Wave::load_slice_track(slice, None)
    }

    /// Load audio from the given slice. Track can be optionally selected.
    /// If not selected, the first track with a known codec will be loaded.
    /// Supported formats are anything that Symphonia can read.
    pub fn load_slice_track<S: AsRef<[u8]> + Send + Sync + 'static>(
        slice: S,
        track: Option<usize>,
    ) -> WaveResult<Wave> {
        let hint = Hint::new();
        let source: Box<dyn MediaSource> = Box::new(Cursor::new(slice));
        Wave::decode(source, track, hint, &|_| {})
    }

    /// Load audio file from the given path. Track can be optionally selected.
    /// If not selected, the first track with a known codec will be loaded.
    /// Supported formats are anything that Symphonia can read.
    pub fn load_track<P: AsRef<Path>>(path: P, track: Option<usize>) -> WaveResult<Wave> {
        Wave::load_track_with_progress(path, track, |_| {})
    }

    /// Load audio file with a progress callback (0.0 to 1.0).
    /// Track can be optionally selected.
    pub fn load_track_with_progress<P: AsRef<Path>>(
        path: P,
        track: Option<usize>,
        on_progress: impl Fn(f32),
    ) -> WaveResult<Wave> {
        let path = path.as_ref();
        let mut hint = Hint::new();

        if let Some(extension) = path.extension()
            && let Some(extension_str) = extension.to_str()
        {
            hint.with_extension(extension_str);
        }

        let source: Box<dyn MediaSource> = match File::open(path) {
            Ok(file) => Box::new(file),
            Err(error) => return Err(Error::IoError(error)),
        };

        Wave::decode(source, track, hint, &on_progress)
    }

    /// Decode track with streaming peak construction.
    fn decode_with_peaks(
        source: Box<dyn MediaSource>,
        track: Option<usize>,
        hint: Hint,
        on_progress: &dyn Fn(f32),
        on_peaks: &dyn Fn(Vec<(f32, f32)>),
        on_metadata: &dyn Fn(usize, u32),
        on_samples: &dyn Fn(&[f32]),
    ) -> WaveResult<(Wave, Vec<(f32, f32)>)> {
        use crate::peak_builder::StreamingPeakBuilder;

        let stream = MediaSourceStream::new(source, Default::default());
        let format_opts = FormatOptions { enable_gapless: false, ..Default::default() };
        let metadata_opts: MetadataOptions = Default::default();
        let mut wave: Option<Wave> = None;
        let mut peak_builder = StreamingPeakBuilder::new(256);

        match symphonia::default::get_probe().format(&hint, stream, &format_opts, &metadata_opts) {
            Ok(probed) => {
                let mut reader = probed.format;

                let track = track.and_then(|t| reader.tracks().get(t)).or_else(|| {
                    reader.tracks().iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
                });

                let track_id = match track {
                    Some(track) => track.id,
                    _ => return Err(Error::DecodeError("Could not find track.")),
                };

                let track = match reader.tracks().iter().find(|track| track.id == track_id) {
                    Some(track) => track,
                    _ => return Err(Error::DecodeError("Could not find track.")),
                };

                let total_frames = track.codec_params.n_frames;
                let sample_rate = track.codec_params.sample_rate.unwrap_or(44100) as f64;
                let progress_interval = (sample_rate * 0.5) as usize;

                // Send metadata early so the UI can size the clip before decode finishes
                if let Some(total) = total_frames {
                    on_metadata(total as usize, sample_rate as u32);
                }

                let decode_opts = DecoderOptions::default();
                let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decode_opts)?;

                let channels = track.codec_params.channels.map(|ch| ch.count()).unwrap_or(2);
                if let Some(total) = total_frames {
                    wave = Some(Wave::with_capacity(channels, sample_rate, total as usize));
                }

                let mut frames_decoded: usize = 0;
                let mut next_progress_at = progress_interval;
                let mut last_peaks_len: usize = 0;
                let mut last_sample_count: usize = 0;
                // Reuse a single decode buffer across all packets to avoid per-packet allocation.
                let mut dest: Option<AudioBuffer<f32>> = None;
                on_progress(0.0);

                loop {
                    let packet = match reader.next_packet() {
                        Ok(packet) => packet,
                        Err(err) => {
                            if let Some(ref wave_output) = wave {
                                // Send remaining ch-0 samples before finishing
                                let ch0 = wave_output.channel(0);
                                if ch0.len() > last_sample_count {
                                    on_samples(&ch0[last_sample_count..]);
                                }
                                on_progress(1.0);
                                let peaks = peak_builder.finish();
                                on_peaks(peaks.clone());
                                return Ok((wave.unwrap(), peaks));
                            } else {
                                return Err(err);
                            }
                        }
                    };

                    if packet.track_id() != track_id { continue; }

                    match decoder.decode(&packet) {
                        Ok(decoded) => {
                            if wave.is_none() {
                                let spec = *decoded.spec();
                                wave = Some(Wave::new(spec.channels.count(), spec.rate as f64));
                            }

                            if let Some(ref mut wave_output) = wave {
                                let buf = dest.get_or_insert_with(|| {
                                    AudioBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec())
                                });
                                buf.clear();
                                buf.render_silence(Some(decoded.frames()));
                                decoded.convert(buf);

                                let buffer_len = decoded.frames();
                                let num_ch = buf.spec().channels.count();

                                // Feed channel 0 to peak builder before appending to wave
                                peak_builder.push_samples(&buf.chan(0)[..buffer_len]);

                                let old_len = wave_output.len();
                                for ch in 0..num_ch {
                                    wave_output.channel_vec_mut(ch).extend_from_slice(&buf.chan(ch)[..buffer_len]);
                                }
                                wave_output.set_len(old_len + buffer_len);

                                frames_decoded += buffer_len;

                                // Send first snapshot immediately (don't wait for progress_interval)
                                if last_peaks_len == 0 && peak_builder.len() > 0 {
                                    on_peaks(peak_builder.snapshot());
                                    last_peaks_len = peak_builder.len();
                                    // Send ch-0 sample delta
                                    let ch0 = wave_output.channel(0);
                                    if ch0.len() > last_sample_count {
                                        on_samples(&ch0[last_sample_count..]);
                                        last_sample_count = ch0.len();
                                    }
                                } else if frames_decoded >= next_progress_at {
                                    if let Some(total) = total_frames {
                                        let ratio = frames_decoded as f32 / total as f32;
                                        on_progress(ratio.min(1.0));
                                    }
                                    if peak_builder.len() > last_peaks_len {
                                        on_peaks(peak_builder.snapshot());
                                        last_peaks_len = peak_builder.len();
                                    }
                                    // Send ch-0 sample delta
                                    let ch0 = wave_output.channel(0);
                                    if ch0.len() > last_sample_count {
                                        on_samples(&ch0[last_sample_count..]);
                                        last_sample_count = ch0.len();
                                    }
                                    next_progress_at = frames_decoded + progress_interval;
                                }
                            }
                        }
                        Err(err) => return Err(err),
                    }
                }
            }
            Err(err) => Err(err),
        }
    }

    /// Decode track from the given source.
    fn decode(
        source: Box<dyn MediaSource>,
        track: Option<usize>,
        hint: Hint,
        on_progress: &dyn Fn(f32),
    ) -> WaveResult<Wave> {
        let stream = MediaSourceStream::new(source, Default::default());

        let format_opts = FormatOptions {
            enable_gapless: false,
            ..Default::default()
        };

        let metadata_opts: MetadataOptions = Default::default();

        let mut wave: Option<Wave> = None;

        match symphonia::default::get_probe().format(&hint, stream, &format_opts, &metadata_opts) {
            Ok(probed) => {
                let mut reader = probed.format;

                // Select track if specified, otherwise select the first track with a known codec.
                let track = track.and_then(|t| reader.tracks().get(t)).or_else(|| {
                    reader
                        .tracks()
                        .iter()
                        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
                });

                let track_id = match track {
                    Some(track) => track.id,
                    _ => return Err(Error::DecodeError("Could not find track.")),
                };

                let track = match reader.tracks().iter().find(|track| track.id == track_id) {
                    Some(track) => track,
                    _ => return Err(Error::DecodeError("Could not find track.")),
                };

                // Read total frame count for progress reporting (available for most formats).
                let total_frames = track.codec_params.n_frames;
                let sample_rate = track.codec_params.sample_rate.unwrap_or(44100) as f64;
                // Report progress roughly every 0.5 seconds of decoded audio.
                let progress_interval = (sample_rate * 0.5) as usize;

                let decode_opts = DecoderOptions::default();

                let mut decoder =
                    symphonia::default::get_codecs().make(&track.codec_params, &decode_opts)?;

                let channels = track.codec_params.channels
                    .map(|ch| ch.count())
                    .unwrap_or(2);

                // Pre-allocate using metadata frame count when available (WAV, FLAC).
                if let Some(total) = total_frames {
                    wave = Some(Wave::with_capacity(channels, sample_rate, total as usize));
                }

                let mut frames_decoded: usize = 0;
                let mut next_progress_at = progress_interval;
                // Reuse a single decode buffer across all packets to avoid per-packet allocation.
                let mut dest: Option<AudioBuffer<f32>> = None;
                on_progress(0.0);

                loop {
                    let packet = match reader.next_packet() {
                        Ok(packet) => packet,
                        Err(err) => {
                            if let Some(wave_output) = wave {
                                on_progress(1.0);
                                return Ok(wave_output);
                            } else {
                                return Err(err);
                            }
                        }
                    };

                    // If the packet does not belong to the selected track, skip it.
                    if packet.track_id() != track_id {
                        continue;
                    }

                    match decoder.decode(&packet) {
                        Ok(decoded) => {
                            if wave.is_none() {
                                let spec = *decoded.spec();
                                wave = Some(Wave::new(spec.channels.count(), spec.rate as f64));
                            }

                            if let Some(ref mut wave_output) = wave {
                                let buf = dest.get_or_insert_with(|| {
                                    AudioBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec())
                                });
                                buf.clear();
                                buf.render_silence(Some(decoded.frames()));
                                decoded.convert(buf);

                                let buffer_len = decoded.frames();
                                let num_ch = buf.spec().channels.count();

                                // Batch-append all channels at once
                                let old_len = wave_output.len();
                                for ch in 0..num_ch {
                                    wave_output.channel_vec_mut(ch).extend_from_slice(&buf.chan(ch)[..buffer_len]);
                                }
                                wave_output.set_len(old_len + buffer_len);

                                frames_decoded += buffer_len;

                                // Report progress periodically.
                                if frames_decoded >= next_progress_at {
                                    if let Some(total) = total_frames {
                                        let ratio = frames_decoded as f32 / total as f32;
                                        on_progress(ratio.min(1.0));
                                    }
                                    next_progress_at = frames_decoded + progress_interval;
                                }
                            }
                        }
                        Err(err) => return Err(err),
                    }
                }
            }
            Err(err) => Err(err),
        }
    }
}
