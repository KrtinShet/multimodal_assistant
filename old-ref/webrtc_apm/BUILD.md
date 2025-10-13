WebRTC APM Shim Build (macOS/Linux)

Prerequisites (macOS):
- Homebrew packages: `brew install webrtc-audio-processing`
- Xcode command line tools (clang++)

Build:
```
cd native/webrtc_apm
mkdir -p dist
clang++ -std=c++17 -O3 -fPIC -shared \
  -I/usr/local/include \
  apm_shim.cc -o dist/libapm_shim.dylib \
  -lwebrtc_audio_processing
```

On Apple Silicon Homebrew might be under `/opt/homebrew`:
```
clang++ -std=c++17 -O3 -fPIC -shared \
  -I/opt/homebrew/include \
  apm_shim.cc -o dist/libapm_shim.dylib \
  -L/opt/homebrew/lib -lwebrtc_audio_processing
```

Linux (example):
```
sudo apt-get install -y libwebrtc-audio-processing-dev
clang++ -std=c++17 -O3 -fPIC -shared \
  apm_shim.cc -o dist/libapm_shim.so \
  -lwebrtc_audio_processing
```

The resulting `dist/libapm_shim.(dylib|so)` is loaded by
`src/multimodal_assistant/processors/webrtc_apm.py`.

