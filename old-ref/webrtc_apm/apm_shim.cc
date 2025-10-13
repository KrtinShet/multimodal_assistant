// Lightweight shim around WebRTC AudioProcessing to expose a C API.

#include "apm_shim.h"

#include <memory>
#include <vector>

#include "api/audio/audio_frame.h"
#include "api/audio/audio_processing.h"
#include "modules/audio_processing/include/audio_processing.h"

using webrtc::AudioProcessing;

struct APMWrapper {
  std::unique_ptr<AudioProcessing> apm;
  int sample_rate_hz{16000};
  int channels{1};
  int frames_per_buffer{160};
};

extern "C" {

APMHandle apm_create(int sample_rate_hz, int num_channels,
                     int enable_aec3, int enable_hpf,
                     int enable_ns, int enable_agc) {
  webrtc::AudioProcessing::Config cfg;
  cfg.echo_canceller.enabled = enable_aec3 != 0;
  cfg.echo_canceller.mobile_mode = false;
  cfg.high_pass_filter.enabled = enable_hpf != 0;
  cfg.noise_suppression.enabled = enable_ns != 0;
  cfg.noise_suppression.level = webrtc::AudioProcessing::Config::NoiseSuppression::kModerate;
  cfg.gain_controller2.enabled = enable_agc != 0;
  cfg.gain_controller2.fixed_digital.gain_db = 0.0f;

  auto apm = AudioProcessingBuilder().Create();
  if (!apm) return nullptr;
  apm->ApplyConfig(cfg);

  auto* w = new APMWrapper();
  w->apm.reset(apm.release());
  w->sample_rate_hz = sample_rate_hz;
  w->channels = num_channels;
  w->frames_per_buffer = sample_rate_hz / 100;  // 10 ms
  return reinterpret_cast<APMHandle>(w);
}

void apm_destroy(APMHandle handle) {
  auto* w = reinterpret_cast<APMWrapper*>(handle);
  delete w;
}

int apm_set_stream_delay_ms(APMHandle handle, int delay_ms) {
  auto* w = reinterpret_cast<APMWrapper*>(handle);
  if (!w) return -1;
  w->apm->set_stream_delay_ms(delay_ms);
  return 0;
}

int apm_process_reverse(APMHandle handle, const float* interleaved, int frames) {
  auto* w = reinterpret_cast<APMWrapper*>(handle);
  if (!w || !interleaved) return -1;
  webrtc::AudioFrame frame;
  frame.num_channels_ = w->channels;
  frame.sample_rate_hz_ = w->sample_rate_hz;
  frame.samples_per_channel_ = frames;
  // Interleaved mono float to int16 expected by APM: use soft conversion.
  std::vector<int16_t> tmp(frames);
  for (int i = 0; i < frames; ++i) {
    float v = interleaved[i];
    if (v > 1.0f) v = 1.0f; else if (v < -1.0f) v = -1.0f;
    tmp[i] = static_cast<int16_t>(v * 32767.0f);
  }
  frame.UpdateFrame(0 /* timestamp */, tmp.data(), frames, w->sample_rate_hz,
                    webrtc::AudioFrame::kNormalSpeech, webrtc::AudioFrame::kVadUnknown, 1);
  return w->apm->ProcessReverseStream(&frame);
}

int apm_process_near(APMHandle handle, const float* interleaved_in, float* interleaved_out, int frames) {
  auto* w = reinterpret_cast<APMWrapper*>(handle);
  if (!w || !interleaved_in || !interleaved_out) return -1;

  webrtc::StreamConfig in_cfg(w->sample_rate_hz, w->channels);
  webrtc::StreamConfig out_cfg(w->sample_rate_hz, w->channels);

  std::vector<float> in_buf(interleaved_in, interleaved_in + frames);
  std::vector<float> out_buf(frames, 0.0f);

  int rv = w->apm->ProcessStream(in_buf.data(), in_cfg, out_cfg, out_buf.data());
  if (rv != 0) return rv;

  for (int i = 0; i < frames; ++i) interleaved_out[i] = out_buf[i];
  return 0;
}

}

