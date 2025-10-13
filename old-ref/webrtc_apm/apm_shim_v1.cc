// Lightweight shim around WebRTC AudioProcessing v1.3 to expose a C API.

#include "apm_shim.h"

#include <memory>
#include <vector>
#include <cstring>

#include "modules/audio_processing/include/audio_processing.h"
#include "api/audio/audio_frame.h"

using webrtc::AudioProcessing;
using webrtc::AudioFrame;
using webrtc::StreamConfig;

struct APMWrapper {
  std::unique_ptr<AudioProcessing> apm;
  int sample_rate_hz{16000};
  int channels{1};
  int frames_per_buffer{160};
  StreamConfig stream_config;
};

extern "C" {

APMHandle apm_create(int sample_rate_hz, int num_channels,
                     int enable_aec3, int enable_hpf,
                     int enable_ns, int enable_agc) {
  webrtc::AudioProcessingBuilder builder;
  std::unique_ptr<AudioProcessing> apm(builder.Create());
  if (!apm) return nullptr;

  // Configure using the new Config API
  AudioProcessing::Config config;
  config.echo_canceller.enabled = (enable_aec3 != 0);
  config.echo_canceller.mobile_mode = false;
  config.high_pass_filter.enabled = (enable_hpf != 0);
  config.noise_suppression.enabled = (enable_ns != 0);
  config.noise_suppression.level = AudioProcessing::Config::NoiseSuppression::kModerate;
  config.gain_controller2.enabled = (enable_agc != 0);

  apm->ApplyConfig(config);

  // Set stream format
  StreamConfig stream_config(sample_rate_hz, num_channels, false);
  apm->Initialize({stream_config, stream_config, stream_config, stream_config});

  auto* w = new APMWrapper();
  w->apm = std::move(apm);
  w->sample_rate_hz = sample_rate_hz;
  w->channels = num_channels;
  w->frames_per_buffer = sample_rate_hz / 100;  // 10 ms
  w->stream_config = stream_config;
  return reinterpret_cast<APMHandle>(w);
}

void apm_destroy(APMHandle handle) {
  auto* w = reinterpret_cast<APMWrapper*>(handle);
  delete w;
}

int apm_set_stream_delay_ms(APMHandle handle, int delay_ms) {
  auto* w = reinterpret_cast<APMWrapper*>(handle);
  if (!w) return -1;
  return w->apm->set_stream_delay_ms(delay_ms);
}

int apm_process_reverse(APMHandle handle, const float* interleaved, int frames) {
  auto* w = reinterpret_cast<APMWrapper*>(handle);
  if (!w || !interleaved) return -1;

  // Convert interleaved float to int16_t
  std::vector<int16_t> input(frames * w->channels);
  std::vector<int16_t> output(frames * w->channels);

  for (int i = 0; i < frames * w->channels; ++i) {
    float v = interleaved[i];
    if (v > 1.0f) v = 1.0f;
    else if (v < -1.0f) v = -1.0f;
    input[i] = static_cast<int16_t>(v * 32767.0f);
  }

  // Use ProcessReverseStream with int16 buffers
  return w->apm->ProcessReverseStream(
      input.data(), w->stream_config, w->stream_config, output.data());
}

int apm_process_near(APMHandle handle, const float* interleaved_in, float* interleaved_out, int frames) {
  auto* w = reinterpret_cast<APMWrapper*>(handle);
  if (!w || !interleaved_in || !interleaved_out) return -1;

  // Convert interleaved float to int16_t
  std::vector<int16_t> input(frames * w->channels);
  std::vector<int16_t> output(frames * w->channels);

  for (int i = 0; i < frames * w->channels; ++i) {
    float v = interleaved_in[i];
    if (v > 1.0f) v = 1.0f;
    else if (v < -1.0f) v = -1.0f;
    input[i] = static_cast<int16_t>(v * 32767.0f);
  }

  // Use ProcessStream with int16 buffers
  int rv = w->apm->ProcessStream(
      input.data(), w->stream_config, w->stream_config, output.data());
  if (rv != 0) return rv;

  // Convert back to float
  for (int i = 0; i < frames * w->channels; ++i) {
    interleaved_out[i] = output[i] / 32768.0f;
  }

  return 0;
}

}
