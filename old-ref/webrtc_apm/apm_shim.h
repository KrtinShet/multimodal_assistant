// Minimal C API around WebRTC AudioProcessing for AEC3.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void* APMHandle;

APMHandle apm_create(int sample_rate_hz, int num_channels,
                     int enable_aec3, int enable_hpf,
                     int enable_ns, int enable_agc);
void apm_destroy(APMHandle apm);

int apm_set_stream_delay_ms(APMHandle apm, int delay_ms);

int apm_process_reverse(APMHandle apm, const float* interleaved, int frames);
int apm_process_near(APMHandle apm, const float* interleaved_in, float* interleaved_out, int frames);

#ifdef __cplusplus
}
#endif

